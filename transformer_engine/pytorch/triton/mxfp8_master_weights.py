# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Batched Triton cast of sharded master weights to MXFP8 primary weights.

Equivalent to ``_cast_master_weights_to_fp8_mxfp8_scaling`` in ``tensor/utils.py`` (bitwise
identical outputs), but every weight in the batch is handled by the same few launches:

1. ``_tile_kernel`` (mode 1) handles 32-row tiles that lie entirely inside this rank's shard:
   their amaxes are already global, so it writes the E8M0 scales and casts in one pass.
2. ``_tile_kernel`` (mode 0) writes the E8M0 scales of the partial amaxes of the remaining
   (shard-boundary) tiles.
3. One all-reduce (MAX) of the packed E8M0 scales across ``group``.
4. ``_scale_kernel`` copies the reduced scales into the model weights, either compact or in
   the GEMM-swizzled layout.
5. ``_tile_kernel`` (mode 2) casts the shard-boundary tiles with the reduced scales.

Master weights are read in their own dtype and rounded to the model dtype in registers, which
removes the temporary model-dtype copy the reference path makes. Launch tables depend only on
the tensors' addresses, shapes and offsets, so they are built once and cached.
"""

from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl
import transformer_engine_torch as tex

__all__ = ["cast_master_weights_to_mxfp8"]

_SCALE_BLOCK = 4096
_CACHE_SIZE = 8
_cache = {}


@triton.jit
def _e8m0(v):
    """Biased E8M0 exponent of a non-negative float, rounding up (cvt.rp.satfinite.ue8m0).

    fp32 subnormals give 0, as in the reference kernels (``amax * (1 / 448)`` is flushed to zero).
    """
    bits = v.to(tl.int32, bitcast=True)
    exp = (bits >> 23) & 0xFF
    man = bits & 0x7FFFFF
    inc = (man > 0) & (exp != 0xFE) & (exp != 0)
    e = exp + inc.to(tl.int32)
    return tl.where(exp == 0xFF, tl.where(man != 0, 0xFF, 0xFE), e)


@triton.jit
def _e8m0_rcp(e):
    """Reciprocal of the scale encoded by biased E8M0 exponent ``e``."""
    b = (254 - e) << 23
    b = tl.where(e == 254, 0x00400000, b)
    b = tl.where(e == 255, 0x7FFFFFFF, b)
    return b.to(tl.float32, bitcast=True)


@triton.jit
def _tile_kernel(
    tile_t,
    tile_y,
    tile_x,
    in_ptrs,
    srow_ptrs,
    scol_ptrs,
    orow_ptrs,
    ocol_ptrs,
    vin_ptrs,
    vorow_ptrs,
    vocol_ptrs,
    cols_a,
    start_a,
    end_a,
    srow_stride_a,
    scol_stride_a,
    has_col_a,
    IN_DTYPE: tl.constexpr,
    ROUND_DTYPE: tl.constexpr,
    BC: tl.constexpr,
    MAXRCP: tl.constexpr,
    MODE: tl.constexpr,
):
    # MODE 0: write E8M0 scales of the partial amaxes. 1: write scales and cast (the tile lies
    # entirely inside this rank's shard). 2: cast with the all-reduced scales.
    pid = tl.program_id(0)
    t = tl.load(tile_t + pid)
    ty = tl.load(tile_y + pid).to(tl.int64)
    tx = tl.load(tile_x + pid).to(tl.int64)
    w = tl.multiple_of(tl.load(cols_a + t), 32)
    has_col = tl.load(has_col_a + t) != 0
    r = ty * 32 + tl.arange(0, 32)[:, None]
    c1 = tx * BC + tl.arange(0, BC)
    c = c1[None, :]
    idx = r * w + c
    if MODE == 1:
        # No shard-boundary mask, and 16-byte aligned virtual bases (addresses of flat element
        # 0 of the full weight), so loads and stores vectorize.
        m = c < w
        inp = tl.multiple_of(tl.load(vin_ptrs + t), 16).to(tl.pointer_type(IN_DTYPE))
        orow = tl.multiple_of(tl.load(vorow_ptrs + t), 16).to(tl.pointer_type(tl.uint8))
        ocol = tl.multiple_of(tl.load(vocol_ptrs + t), 16).to(tl.pointer_type(tl.uint8))
        off = idx
    else:
        start = tl.load(start_a + t)
        end = tl.load(end_a + t)
        m = (c < w) & (idx >= start) & (idx < end)
        inp = tl.load(in_ptrs + t).to(tl.pointer_type(IN_DTYPE))
        orow = tl.load(orow_ptrs + t).to(tl.pointer_type(tl.uint8))
        ocol = tl.load(ocol_ptrs + t).to(tl.pointer_type(tl.uint8))
        off = idx - start
    x = tl.load(inp + off, mask=m, other=0.0, eviction_policy="evict_first")
    x = x.to(ROUND_DTYPE).to(tl.float32)
    scol = tl.load(scol_ptrs + t).to(tl.pointer_type(tl.uint8)) + ty * tl.load(scol_stride_a + t)
    j = tx * (BC // 32) + tl.arange(0, BC // 32)[None, :]
    srow = tl.load(srow_ptrs + t).to(tl.pointer_type(tl.uint8)) + r * tl.load(srow_stride_a + t)
    srow = srow + j
    mcol = (c1 < w) & has_col
    mrow = j < w // 32
    if MODE != 2:
        a = tl.abs(x)
        ec = _e8m0(tl.max(a, axis=0) * MAXRCP)
        er = _e8m0(tl.max(tl.reshape(a, (32, BC // 32, 32)), axis=2) * MAXRCP)
        tl.store(scol + c1, ec.to(tl.uint8), mask=mcol)
        tl.store(srow, er.to(tl.uint8), mask=mrow)
    else:
        ec = tl.load(scol + c1, mask=mcol, other=0).to(tl.int32)
        er = tl.load(srow, mask=mrow, other=0).to(tl.int32)
    if MODE != 0:
        yr = tl.reshape(tl.reshape(x, (32, BC // 32, 32)) * _e8m0_rcp(er)[:, :, None], (32, BC))
        tl.store(orow + off, yr.to(tl.float8e4nv).to(tl.uint8, bitcast=True), mask=m)
        yc = x * _e8m0_rcp(ec)[None, :]
        tl.store(ocol + off, yc.to(tl.float8e4nv).to(tl.uint8, bitcast=True), mask=m & has_col)


@triton.jit
def _scale_kernel(
    packed,
    prog_seg,
    prog_off,
    seg_src,
    seg_len,
    seg_dst,
    seg_kind,
    seg_rows,
    seg_cols,
    BLOCK: tl.constexpr,
    ALIGNED: tl.constexpr,
    SWIZZLE: tl.constexpr,
):
    pid = tl.program_id(0)
    s = tl.load(prog_seg + pid)
    off = tl.load(prog_off + pid)
    n = tl.load(seg_len + s)
    src = tl.load(seg_src + s)
    dst = tl.load(seg_dst + s)
    if ALIGNED:
        off = tl.multiple_of(off, 16)
        n = tl.multiple_of(n, 16)
        src = tl.multiple_of(src, 16)
        dst = tl.multiple_of(dst, 16)
    k = off + tl.arange(0, BLOCK)
    msk = k < n
    v = tl.load(packed + src + k, mask=msk, other=0)
    dst_ptr = dst.to(tl.pointer_type(tl.uint8))
    if SWIZZLE:
        # GEMM-swizzled layout (see gemm_swizzled_scale_idx in common/cast/mxfp8/swizzle.cuh):
        # 128 x 4 tiles of 512 bytes. Kind 1 swizzles rowwise scales [M, K/32], kind 2
        # columnwise scales [M/32, K] in their transposed view, kind 0 stays compact.
        kind = tl.load(seg_kind + s)
        nr = tl.load(seg_rows + s)
        nc = tl.load(seg_cols + s)
        i = k // nc
        j = k % nc
        mm = tl.where(kind == 2, j, i)
        kk = tl.where(kind == 2, i, j)
        kp = tl.where(kind == 2, nr, nc)
        sw = (
            ((mm // 128) * (kp // 4) + kk // 4) * 512
            + (mm % 32) * 16
            + ((mm // 32) % 4) * 4
            + kk % 4
        )
        k = tl.where(kind == 0, k, sw)
    tl.store(dst_ptr + k, v, mask=msk)


def _tl_dtype(dtype: torch.dtype):
    return {torch.float32: tl.float32, torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}[
        dtype
    ]


def _fragments(model_weight, start_offset, fragment, use_fsdp_shard_model_weights):
    has_col = model_weight._columnwise_scale_inv is not None
    if use_fsdp_shard_model_weights:
        return fragment[0], (fragment[1] if has_col else None)
    return (
        model_weight._rowwise_data.reshape(-1)[start_offset:],
        model_weight._columnwise_data.reshape(-1)[start_offset:] if has_col else None,
    )


def _supported(params, use_fsdp_shard_model_weights) -> Optional[tuple]:
    """Cache key of the batch, or None if the batch is outside what the kernels handle."""
    model_dtype = params[0][0].dtype
    if model_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        return None
    in_dtype = None
    key = [use_fsdp_shard_model_weights, model_dtype]
    for model_weight, master_weight, start_offset, fragment in params:
        if model_weight.dtype != model_dtype or len(model_weight.shape) != 2:
            return None
        if model_weight._fp8_dtype != tex.DType.kFloat8E4M3:
            return None
        rsi, csi = model_weight._rowwise_scale_inv, model_weight._columnwise_scale_inv
        if rsi is None or rsi.dim() != 2 or (csi is not None and csi.dim() != 2):
            return None
        h, w = model_weight.shape
        if h % 32 or w % 32:
            return None
        entry = [rsi.data_ptr(), tuple(rsi.shape), h, w]
        if csi is not None:
            entry += [csi.data_ptr(), tuple(csi.shape)]
        if master_weight is not None and master_weight.numel() > 0:
            if in_dtype is None:
                in_dtype = master_weight.dtype
            if master_weight.dtype != in_dtype or not master_weight.is_contiguous():
                return None
            rf, cf = _fragments(model_weight, start_offset, fragment, use_fsdp_shard_model_weights)
            if not rf.is_contiguous() or (cf is not None and not cf.is_contiguous()):
                return None
            entry += [master_weight.data_ptr(), master_weight.numel(), start_offset, rf.data_ptr()]
            if cf is not None:
                entry.append(cf.data_ptr())
        key.append(tuple(entry))
    if in_dtype not in (None, torch.float32, torch.bfloat16, torch.float16):
        return None
    key.append(in_dtype)
    return tuple(key), in_dtype or model_dtype


def _build_state(params, use_fsdp_shard_model_weights, gemm_swizzled_scales, block_cols):
    device = params[0][0].device
    seg_len, seg_dst, seg_kind, seg_rows, seg_cols = [], [], [], [], []
    srow_off, scol_off = [], []
    total = 0
    for (model_weight, _, _, _), swizzle in zip(params, gemm_swizzled_scales):
        for kind, si, offs in (
            (1, model_weight._rowwise_scale_inv, srow_off),
            (2, model_weight._columnwise_scale_inv, scol_off),
        ):
            offs.append(total)
            if si is None:
                continue
            total += si.numel()
            seg_len.append(si.numel())
            seg_dst.append(si.data_ptr())
            seg_kind.append(kind if swizzle else 0)
            seg_rows.append(si.shape[0])
            seg_cols.append(si.shape[1])
    seg_src = [0]
    for n in seg_len[:-1]:
        seg_src.append(seg_src[-1] + n)
    packed = torch.zeros(sum(seg_len), dtype=torch.uint8, device=device)

    prog_seg, prog_off = [], []
    for s, n in enumerate(seg_len):
        for o in range(0, n, _SCALE_BLOCK):
            prog_seg.append(s)
            prog_off.append(o)

    names = (
        "in_ptrs",
        "srow_ptrs",
        "scol_ptrs",
        "orow_ptrs",
        "ocol_ptrs",
        "vin_ptrs",
        "vorow_ptrs",
        "vocol_ptrs",
        "cols",
        "start",
        "end",
        "srow_stride",
        "scol_stride",
        "has_col",
    )
    cols = {k: [] for k in names}
    tiles = {"inner": ([], [], []), "edge": ([], [], [])}
    for i, (model_weight, master_weight, start_offset, fragment) in enumerate(params):
        if master_weight is None or master_weight.numel() == 0:
            continue
        h, w = model_weight.shape
        rf, cf = _fragments(model_weight, start_offset, fragment, use_fsdp_shard_model_weights)
        has_col = cf is not None
        t = len(cols["in_ptrs"])
        end = start_offset + master_weight.numel()
        vin = master_weight.data_ptr() - start_offset * master_weight.element_size()
        vorow = rf.data_ptr() - start_offset
        vocol = cf.data_ptr() - start_offset if has_col else vorow
        aligned = vin % 16 == 0 and vorow % 16 == 0 and vocol % 16 == 0
        cols["in_ptrs"].append(master_weight.data_ptr())
        cols["srow_ptrs"].append(packed.data_ptr() + srow_off[i])
        cols["scol_ptrs"].append(packed.data_ptr() + scol_off[i])
        cols["orow_ptrs"].append(rf.data_ptr())
        cols["ocol_ptrs"].append(cf.data_ptr() if has_col else rf.data_ptr())
        cols["vin_ptrs"].append(vin)
        cols["vorow_ptrs"].append(vorow)
        cols["vocol_ptrs"].append(vocol)
        cols["cols"].append(w)
        cols["start"].append(start_offset)
        cols["end"].append(end)
        cols["srow_stride"].append(model_weight._rowwise_scale_inv.shape[1])
        cols["scol_stride"].append(model_weight._columnwise_scale_inv.shape[1] if has_col else 0)
        cols["has_col"].append(int(has_col))
        ntx = (w + block_cols - 1) // block_cols
        for ty in range((start_offset // w) // 32, ((end - 1) // w) // 32 + 1):
            for tx in range(ntx):
                lo = ty * 32 * w + tx * block_cols
                hi = (ty * 32 + 31) * w + min(w, (tx + 1) * block_cols)
                inner = aligned and lo >= start_offset and hi <= end
                tt, ty_, tx_ = tiles["inner" if inner else "edge"]
                tt.append(t)
                ty_.append(ty)
                tx_.append(tx)

    def dev(x, dtype):
        return torch.tensor(x or [0], dtype=dtype).to(device)

    st = {"packed": packed, "nprogs": len(prog_seg)}
    st["scale_aligned"] = all(d % 16 == 0 for d in seg_dst) and all(n % 16 == 0 for n in seg_len)
    for name, vals, dtype in (
        ("prog_seg", prog_seg, torch.int32),
        ("prog_off", prog_off, torch.int64),
        ("seg_src", seg_src, torch.int64),
        ("seg_len", seg_len, torch.int64),
        ("seg_dst", seg_dst, torch.int64),
        ("seg_kind", seg_kind, torch.int32),
        ("seg_rows", seg_rows, torch.int64),
        ("seg_cols", seg_cols, torch.int64),
    ):
        st[name] = dev(vals, dtype)
    st["any_swizzled"] = any(gemm_swizzled_scales)
    for k, v in cols.items():
        st[k] = dev(v, torch.int32 if k == "has_col" else torch.int64)
    for name, (tt, ty_, tx_) in tiles.items():
        st[name + "_n"] = len(tt)
        st[name + "_t"] = dev(tt, torch.int32)
        st[name + "_y"] = dev(ty_, torch.int32)
        st[name + "_x"] = dev(tx_, torch.int32)
    return st


def cast_master_weights_to_mxfp8(
    params: List[Tuple],
    group,
    use_fsdp_shard_model_weights: bool,
    gemm_swizzled_scales: List[bool],
    block_cols: int = 128,
    num_warps: int = 4,
) -> bool:
    """Cast master-weight shards into MXFP8 model weights.

    ``params`` holds ``(model_weight, master_weight, start_offset, fsdp_shard_model_weight)``
    tuples as in ``quantize_master_weights``, with master weights in their own dtype. Model
    weights flagged in ``gemm_swizzled_scales`` receive GEMM-swizzled scales. Returns False,
    without touching any tensor, if the batch is not supported (callers then fall back to the
    reference path): model weights must be 2D E4M3 tensors with both dimensions divisible by 32,
    in one dtype, with contiguous master-weight shards of one dtype.
    """
    supported = _supported(params, use_fsdp_shard_model_weights)
    if supported is None:
        return False
    key, in_dtype = supported
    key = key + (tuple(gemm_swizzled_scales), block_cols)
    st = _cache.get(key)
    if st is None:
        st = _build_state(params, use_fsdp_shard_model_weights, gemm_swizzled_scales, block_cols)
        if len(_cache) >= _CACHE_SIZE:
            _cache.clear()
        _cache[key] = st

    model_dtype = params[0][0].dtype
    packed = st["packed"]
    packed.zero_()

    def launch(tiles, mode):
        n = st[tiles + "_n"]
        if n == 0:
            return
        _tile_kernel[(n,)](
            st[tiles + "_t"],
            st[tiles + "_y"],
            st[tiles + "_x"],
            st["in_ptrs"],
            st["srow_ptrs"],
            st["scol_ptrs"],
            st["orow_ptrs"],
            st["ocol_ptrs"],
            st["vin_ptrs"],
            st["vorow_ptrs"],
            st["vocol_ptrs"],
            st["cols"],
            st["start"],
            st["end"],
            st["srow_stride"],
            st["scol_stride"],
            st["has_col"],
            IN_DTYPE=_tl_dtype(in_dtype),
            ROUND_DTYPE=_tl_dtype(model_dtype),
            BC=block_cols,
            MAXRCP=1.0 / 448.0,
            MODE=mode,
            num_warps=num_warps,
        )

    launch("inner", 1)
    launch("edge", 0)
    if packed.numel() > 0:
        torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.MAX, group=group)
    if st["nprogs"] > 0:
        _scale_kernel[(st["nprogs"],)](
            packed,
            st["prog_seg"],
            st["prog_off"],
            st["seg_src"],
            st["seg_len"],
            st["seg_dst"],
            st["seg_kind"],
            st["seg_rows"],
            st["seg_cols"],
            BLOCK=_SCALE_BLOCK,
            ALIGNED=st["scale_aligned"],
            SWIZZLE=st["any_swizzled"],
            num_warps=4,
        )
    launch("edge", 2)
    for (model_weight, _, _, _), swizzle in zip(params, gemm_swizzled_scales):
        model_weight._with_gemm_swizzled_scales = bool(swizzle)
    return True
