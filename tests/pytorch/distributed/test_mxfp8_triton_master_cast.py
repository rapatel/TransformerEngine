# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""NVTE_MXFP8_TRITON_MASTER_CAST=1 matches the reference MXFP8 master-weight cast bit for bit."""

import argparse
import os
import pathlib
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist

import transformer_engine.pytorch as te
from transformer_engine.pytorch import MXFP8Quantizer, is_mxfp8_available
from transformer_engine.pytorch.tensor.utils import quantize_master_weights
from transformer_engine.pytorch.triton import mxfp8_master_weights

# (rows, cols, columnwise). 160 and 96 columns leave partial 128-column tiles.
_SHAPES = [
    (256, 512, True),
    (64, 7168, True),
    (96, 160, True),
    (2048, 64, True),
    (32, 32, True),
    (128, 96, False),
    (512, 384, True),
]


def _full_weights(seed, extreme):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    for h, w, _ in _SHAPES:
        x = torch.randn(h, w, device="cuda", generator=gen)
        if extreme:
            # Block amaxes across the E8M0 range, including fp32-subnormal amax / 448.
            e = torch.randint(-130, 120, (h, w // 32), device="cuda", generator=gen)
            x = x * torch.exp2(e.float()).repeat_interleave(32, dim=1)
        x[: min(h, 3)] = 0
        out.append(x)
    return out


def _make_model_weights():
    weights = []
    for h, w, col in _SHAPES:
        quantizer = MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E4M3, rowwise=True, columnwise=col)
        weight = quantizer.make_empty((h, w), dtype=torch.bfloat16, device="cuda")
        quantizer.update_quantized(torch.zeros(h, w, dtype=torch.bfloat16, device="cuda"), weight)
        weights.append(weight)
    return weights


def _shards(full, rank, world_size, master_dtype):
    """ZeRO-style split of the concatenated weights, with cuts at odd element offsets."""
    sizes = [x.numel() for x in full]
    total = sum(sizes)
    cuts = [0] + [k * total // world_size + 7 * k for k in range(1, world_size)] + [total]
    lo, hi = cuts[rank], cuts[rank + 1]
    masters, offsets = [], []
    base = 0
    for x, n in zip(full, sizes):
        a, b = max(lo, base), min(hi, base + n)
        if a < b:
            masters.append(x.flatten()[a - base : b - base].to(master_dtype).contiguous())
            offsets.append(a - base)
        else:
            masters.append(None)
            offsets.append(0)
        base += n
    return masters, offsets


def _fragments(weights, masters, offsets):
    frags = []
    for weight, master, off in zip(weights, masters, offsets):
        n = 0 if master is None else master.numel()
        row = weight._rowwise_data.flatten()[off : off + n]
        col = None
        if weight._columnwise_data is not None:
            col = weight._columnwise_data.flatten()[off : off + n]
        frags.append((row, col))
    return frags


def _cast(weights, masters, offsets, group, fsdp, swizzle, triton):
    os.environ["NVTE_MXFP8_TRITON_MASTER_CAST"] = "1" if triton else "0"
    frags = _fragments(weights, masters, offsets) if fsdp else None
    quantize_master_weights(weights, masters, offsets, group, frags, gemm_swizzled_scales=swizzle)
    if triton:
        # The Triton path ran (rather than fell back) and later steps reuse its tables.
        assert len(mxfp8_master_weights._cache) == 1


def _state(weight):
    names = ["_rowwise_data", "_rowwise_scale_inv"]
    if weight._columnwise_data is not None:
        names += ["_columnwise_data", "_columnwise_scale_inv"]
    out = [(name, getattr(weight, name).clone()) for name in names]
    return out + [("_with_gemm_swizzled_scales", weight._with_gemm_swizzled_scales)]


def _check_case(group, fsdp, master_dtype, swizzle, extreme):
    rank, world_size = dist.get_rank(group), dist.get_world_size(group)
    ref_weights, fast_weights = _make_model_weights(), _make_model_weights()
    mxfp8_master_weights._cache.clear()
    masters = None
    for step, seed in enumerate((1234, 5678)):
        full = _full_weights(seed, extreme)
        new_masters, offsets = _shards(full, rank, world_size, master_dtype)
        if masters is None:
            masters = new_masters
        else:
            # Update in place, like an optimizer step, so the cached launch tables are reused.
            for m, n in zip(masters, new_masters):
                if m is not None:
                    m.copy_(n)
        _cast(ref_weights, masters, offsets, group, fsdp, swizzle, triton=False)
        _cast(fast_weights, masters, offsets, group, fsdp, swizzle, triton=True)
        for i, (ref, fast) in enumerate(zip(ref_weights, fast_weights)):
            for (name, a), (_, b) in zip(_state(ref), _state(fast)):
                where = f"weight {i} {_SHAPES[i]} step {step} {name}"
                if isinstance(a, bool):
                    assert a == b, f"{where}: {a} vs {b}"
                    continue
                bad = (a != b).nonzero()
                assert bad.numel() == 0, (
                    f"{where}: {bad.shape[0]} mismatches, first at {bad[0].tolist()}: "
                    f"reference {a[tuple(bad[0])].item()} vs triton {b[tuple(bad[0])].item()}"
                )
        if extreme:
            continue
        # Compare with quantizing the full (rounded) weight in one shot.
        for j, ((h, w, col), x, weight, swz) in enumerate(
            zip(_SHAPES, full, fast_weights, swizzle)
        ):
            quantizer = MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E4M3, rowwise=True, columnwise=col)
            expected = quantizer(x.to(master_dtype).to(torch.bfloat16))
            assert weight._with_gemm_swizzled_scales == swz
            directions = [("rowwise", (h, w // 32))]
            if col:
                directions.append(("columnwise", (h // 32, w)))
            for d, (rows, cols) in directions:
                ref = getattr(expected, f"_{d}_scale_inv")
                padded = torch.zeros_like(ref)
                padded[:rows, :cols] = ref[:rows, :cols]
                if swz:
                    padded = _swizzle(padded if d == "rowwise" else padded.t())
                torch.testing.assert_close(
                    getattr(weight, f"_{d}_scale_inv").flatten(), padded.flatten(), rtol=0, atol=0
                )
                # Each rank wrote only its own shard; assemble the full weight.
                data = torch.zeros(h * w, dtype=torch.uint8, device="cuda")
                if masters[j] is not None:
                    lo, hi = offsets[j], offsets[j] + masters[j].numel()
                    data[lo:hi] = getattr(weight, f"_{d}_data").flatten()[lo:hi]
                dist.all_reduce(data, op=dist.ReduceOp.MAX, group=group)
                torch.testing.assert_close(
                    data.view(h, w), getattr(expected, f"_{d}_data"), rtol=0, atol=0
                )


def _swizzle(scales):
    """GEMM-swizzled layout of padded [M, K] scales: 128 x 4 tiles stored as [32][4][4]."""
    m, k = scales.shape
    return scales.reshape(m // 128, 4, 32, k // 4, 4).permute(0, 3, 2, 1, 4).contiguous()


def run_worker():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    group = dist.new_group(backend="nccl")
    alternate = [i % 2 == 0 for i in range(len(_SHAPES))]
    for fsdp in (False, True):
        for master_dtype in (torch.float32, torch.bfloat16):
            for swizzle in ([False] * len(_SHAPES), alternate, [True] * len(_SHAPES)):
                for extreme in (False, True):
                    _check_case(group, fsdp, master_dtype, swizzle, extreme)
                    if rank == 0:
                        print(
                            f"ok fsdp={fsdp} master={master_dtype} swizzle={swizzle} "
                            f"extreme={extreme}",
                            flush=True,
                        )
    dist.destroy_process_group()


@pytest.mark.skipif(not is_mxfp8_available(), reason="MXFP8 is not supported")
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_mxfp8_triton_master_cast(world_size: int) -> None:
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={world_size}",
        str(pathlib.Path(__file__).resolve()),
        "--worker",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    if parser.parse_args().worker:
        run_worker()
