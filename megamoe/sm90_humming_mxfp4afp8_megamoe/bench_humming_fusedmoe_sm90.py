#!/usr/bin/env python3
"""SM90 Humming MXFP4A-FP8 fused-MoE benchmark.

The timed path is:

    pre-quantized FP8 input
      -> Humming INDEXED MXFP4A-FP8 W13
      -> Triton SwiGLU, top-k weighting, and group-128 E4M3 quantization
      -> Humming INDEXED MXFP4A-FP8 W2
      -> Humming fused route reduction

This is a local fused-MoE benchmark.  Every rank holds the complete expert set;
there is no router, expert-parallel dispatch/combine, or other communication in
the timed region.  Input quantization, MXFP4/E8M0 weight generation and
transform, route metadata construction, allocations used only for setup, and
JIT warmup are outside the samples.

The workload shapes, random top-k construction, activation clamp, CUDA-event
sampling, cold-L2 option, and rank-0 maximum-rank-median reporting follow the
PR383-style HT benchmark convention.  Route weights are applied before W2
input quantization to match that benchmark's activation contract.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
from typing import Any, Callable, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    from humming import ops as humming_ops
    from humming.config import GemmType
    from humming.layer import HummingLayer
    from humming.ops.moe import moe_fused_mul_sum
    from humming.tune import get_heuristics_config
except Exception as exc:  # pragma: no cover - exercised on the GPU host
    humming_ops = None
    GemmType = None
    HummingLayer = None
    moe_fused_mul_sum = None
    get_heuristics_config = None
    _HUMMING_IMPORT_ERROR = exc
else:
    _HUMMING_IMPORT_ERROR = None


SHAPES = {
    "flash": {
        "hidden": 4096,
        "intermediate_hidden": 2048,
        "num_experts": 256,
        "num_topk": 6,
    },
    "pro": {
        "hidden": 7168,
        "intermediate_hidden": 3072,
        "num_experts": 384,
        "num_topk": 6,
    },
    "mimo_pro": {
        "hidden": 6144,
        "intermediate_hidden": 2048,
        "num_experts": 384,
        "num_topk": 8,
    },
}

BACKEND = "humming_fused_mxfp4afp8"
MODE = "fusedmoe_indexed"
SCOPE = "prequantized_input_w13_swiglu_requant_w2_reduce_no_router_no_comm"
ACT_SF_GRAN = 128
_FP8_E4M3_MAX_TL = tl.constexpr(448.0)


@triton.jit
def _swiglu_apply_weight_to_fp8_kernel(
    x_ptr,
    topk_w_ptr,
    y_ptr,
    y_sf_ptr,
    M,
    H,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_sfm,
    stride_sfk,
    clamp_value,
    HAS_CLAMP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M

    gate_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xn
    up_ptrs = x_ptr + offs_m[:, None] * stride_xm + (H + offs_k[None, :]) * stride_xn
    gate = tl.load(gate_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    if HAS_CLAMP:
        gate = tl.minimum(gate, clamp_value)
        up = tl.minimum(tl.maximum(up, -clamp_value), clamp_value)

    y = gate * tl.sigmoid(gate) * up
    topk_w = tl.load(topk_w_ptr + offs_m, mask=mask_m, other=1.0)
    y *= topk_w[:, None]
    sf = tl.maximum(tl.max(tl.abs(y), axis=1) / _FP8_E4M3_MAX_TL, 1.0e-30)
    y_fp8 = (y / sf[:, None]).to(tl.float8e4nv)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_k[None, :] * stride_yn
    tl.store(y_ptrs, y_fp8, mask=mask_m[:, None])
    sf_ptrs = y_sf_ptr + offs_m * stride_sfm + pid_k * stride_sfk
    tl.store(sf_ptrs, sf, mask=mask_m)


def swiglu_apply_weight_to_fp8(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    clamp_value: float | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous()
    m, two_h = x.shape
    h = two_h // 2
    assert two_h == 2 * h and h % ACT_SF_GRAN == 0
    topk_weights = topk_weights.reshape(-1)
    if topk_weights.numel() != m:
        raise RuntimeError(
            f"SwiGLU weight rows mismatch: x_rows={m}, "
            f"topk_weights={topk_weights.numel()}"
        )

    y = torch.empty((m, h), dtype=torch.float8_e4m3fn, device=x.device)
    y_sf = torch.empty((m, h // ACT_SF_GRAN), dtype=torch.float32, device=x.device)
    block_m = 16
    grid = (triton.cdiv(m, block_m), h // ACT_SF_GRAN)
    _swiglu_apply_weight_to_fp8_kernel[grid](
        x,
        topk_weights,
        y,
        y_sf,
        m,
        h,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        y_sf.stride(0),
        y_sf.stride(1),
        float(clamp_value) if clamp_value is not None else 0.0,
        HAS_CLAMP=clamp_value is not None,
        BLOCK_M=block_m,
        BLOCK_K=ACT_SF_GRAN,
    )
    return y, y_sf


def _init_dist(local_rank: int, world_size: int):
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", "29500")
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=local_rank,
            world_size=world_size,
        )
    return dist.get_rank(), dist.get_world_size()


def _parameter_seed(seed: int, layer_tag: int, name: str, expert_id: int) -> int:
    name_tag = sum((index + 1) * ord(char) for index, char in enumerate(name))
    return seed + layer_tag * 1_000_003 + name_tag * 101 + expert_id * 7_919


def _fill_parameter_for_expert(
    target: torch.Tensor,
    name: str,
    seed: int,
) -> None:
    generator = torch.Generator(device=target.device).manual_seed(seed)
    if "weight_scale" in name:
        if target.element_size() == 1:
            # Conservative E8M0 exponents keep the synthetic two-layer output finite.
            raw = torch.randint(
                114,
                128,
                target.view(torch.uint8).shape,
                dtype=torch.uint8,
                device=target.device,
                generator=generator,
            )
            target.view(torch.uint8).copy_(raw)
        elif target.is_floating_point():
            target.fill_(1.0)
        else:
            target.zero_()
    elif "weight" in name:
        raw = torch.randint(
            0,
            256,
            target.view(torch.uint8).shape,
            dtype=torch.uint8,
            device=target.device,
            generator=generator,
        )
        target.view(torch.uint8).copy_(raw)
    elif target.is_floating_point():
        target.normal_(mean=0.0, std=0.01, generator=generator)
    else:
        target.zero_()


def _make_humming_layer(
    shape_n: int,
    shape_k: int,
    num_experts: int,
    seed: int,
    layer_tag: int,
) -> HummingLayer:
    layer = HummingLayer(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=num_experts,
        weight_config={
            "dtype": "float4e2m1",
            "group_size": 32,
            "scale_dtype": "float8e8m0",
        },
        input_config={"dtype": "float8e4m3", "group_size": ACT_SF_GRAN},
        torch_dtype=torch.bfloat16,
    ).cuda()
    with torch.no_grad():
        for name, parameter in layer.named_parameters():
            if parameter.ndim == 0 or parameter.shape[0] != num_experts:
                raise RuntimeError(
                    f"unexpected Humming parameter shape for {name}: "
                    f"{tuple(parameter.shape)}, experts={num_experts}"
                )
            for expert_id in range(num_experts):
                _fill_parameter_for_expert(
                    parameter[expert_id],
                    name,
                    _parameter_seed(seed, layer_tag, name, expert_id),
                )
    layer.transform()
    return layer


def _compute_config() -> dict[str, Any]:
    return {
        "gemm_type": GemmType.INDEXED.value,
        "use_f16_accum": False,
        "use_m_major_input_scale": False,
    }


def _indexed_tuning(layer: HummingLayer) -> list[Any]:
    return get_heuristics_config(
        layer.humming_metas[""],
        gemm_type=GemmType.INDEXED,
        use_f16_accum=False,
    )


def _choose_indexed_block_size(
    tuning_config: list[Any], shape_m: int, top_k: int
) -> int:
    routed_shape_m = shape_m * top_k
    for minimum, maximum, config in tuning_config:
        if routed_shape_m > minimum and routed_shape_m <= maximum:
            return int(config["block_shape"][0])
    raise RuntimeError(f"no Humming indexed config for routed_shape_m={routed_shape_m}")


def _indexed_metadata(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_ids = topk_ids.reshape(-1)
    invalid_id = flat_ids.numel()
    sorted_chunks = []
    block_expert_ids = []
    for expert_id in range(num_experts):
        route_ids = torch.where(flat_ids == expert_id)[0].to(torch.int32)
        num_blocks = math.ceil(route_ids.numel() / block_size)
        padded_rows = num_blocks * block_size
        if padded_rows > route_ids.numel():
            route_ids = F.pad(
                route_ids,
                (0, padded_rows - route_ids.numel()),
                value=invalid_id,
            )
        sorted_chunks.append(route_ids)
        block_expert_ids.extend([expert_id] * num_blocks)

    sorted_ids = torch.cat(sorted_chunks).to(torch.int32).contiguous()
    expert_ids = torch.tensor(
        block_expert_ids,
        dtype=torch.int32,
        device=topk_ids.device,
    ).contiguous()
    num_tokens_padded = torch.tensor(
        sorted_ids.numel(),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    return sorted_ids, expert_ids, num_tokens_padded


def _route_summary(counts: list[int]) -> dict[str, Any]:
    active = [value for value in counts if value > 0]
    mean = sum(counts) / len(counts) if counts else 0.0
    return {
        "route_counts": counts,
        "route_total": sum(counts),
        "touched_experts": len(active),
        "min_active_expert_routes": min(active) if active else 0,
        "max_expert_routes": max(active) if active else 0,
        "mean_expert_routes": mean,
        "max_over_mean_routes": max(active) / mean if active and mean else 0.0,
    }


def _barrier() -> None:
    dist.barrier()


def _measure(
    fn: Callable[[], torch.Tensor],
    flush: torch.Tensor | None,
    warmups: int,
    samples: int,
) -> list[float]:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    _barrier()

    samples_us = []
    for _ in range(samples):
        if flush is not None:
            flush.zero_()
        torch.cuda.synchronize()
        _barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0)
    return samples_us


def _local_timing_row(
    rank: int,
    cfg: dict[str, Any],
    samples_us: list[float],
    route: dict[str, Any],
    padding: dict[str, Any],
    requested_flush: int,
    actual_flush: int,
    observation: int,
) -> dict[str, Any]:
    return {
        "backend": BACKEND,
        "mode": MODE,
        "scope": SCOPE,
        "shape": cfg["shape"],
        "m": cfg["m"],
        "seed": cfg["seed"],
        "weight_seed": cfg["weight_seed"],
        "observation": observation,
        "rank": rank,
        "num_samples": len(samples_us),
        "returned_us": statistics.median(samples_us),
        "mean_us": statistics.mean(samples_us),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
        "samples_us": samples_us,
        **route,
        **padding,
        "requested_flush_l2_bytes": requested_flush,
        "actual_flush_l2_bytes": actual_flush,
    }


def _aggregate_timing_rows(
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    observation: int,
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["rank"])
    max_row = max(ordered, key=lambda row: row["returned_us"])
    return {
        "backend": BACKEND,
        "mode": MODE,
        "scope": SCOPE,
        "shape": cfg["shape"],
        "m": cfg["m"],
        "seed": cfg["seed"],
        "weight_seed": cfg["weight_seed"],
        "observation": observation,
        "num_samples": cfg["samples"],
        "metric": "max_rank_of_per_rank_median_cuda_event_us",
        "max_rank": max_row["rank"],
        "max_rank_us": max_row["returned_us"],
        "per_rank_us": [row["returned_us"] for row in ordered],
        "route_total_per_rank": [row["route_total"] for row in ordered],
        "touched_experts_per_rank": [row["touched_experts"] for row in ordered],
        "w13_padded_rows_per_rank": [row["w13_padded_rows"] for row in ordered],
        "w2_padded_rows_per_rank": [row["w2_padded_rows"] for row in ordered],
        "w13_padding_efficiency_per_rank": [
            row["w13_padding_efficiency"] for row in ordered
        ],
        "w2_padding_efficiency_per_rank": [
            row["w2_padding_efficiency"] for row in ordered
        ],
        "requested_flush_l2_bytes": rows[0]["requested_flush_l2_bytes"],
        "actual_flush_l2_bytes_min": min(row["actual_flush_l2_bytes"] for row in rows),
    }


def _worker(local_rank: int, num_processes: int, cfg: dict[str, Any]):
    if HummingLayer is None or moe_fused_mul_sum is None:
        raise RuntimeError(f"Humming import failed: {_HUMMING_IMPORT_ERROR}")

    rank, num_ranks = _init_dist(local_rank, num_processes)
    completed = False
    try:
        major, minor = torch.cuda.get_device_capability()
        if major != 9:
            raise RuntimeError(f"SM90 required, got SM{major}{minor}")

        shape = SHAPES[cfg["shape"]]
        m = cfg["m"]
        hidden = shape["hidden"]
        intermediate = shape["intermediate_hidden"]
        num_experts = shape["num_experts"]
        num_topk = shape["num_topk"]

        torch.manual_seed(cfg["seed"] + rank)
        random.seed(cfg["seed"] + rank)
        x_bf16 = torch.randn((m, hidden), dtype=torch.bfloat16, device="cuda")
        scores = torch.randn((m, num_experts), dtype=torch.float32, device="cuda")
        topk_weights, topk_idx = torch.topk(
            scores,
            num_topk,
            dim=-1,
            largest=True,
            sorted=False,
        )
        topk_idx = topk_idx.to(torch.int64).contiguous()
        topk_weights = topk_weights.contiguous()
        route_counts = torch.bincount(
            topk_idx.reshape(-1),
            minlength=num_experts,
        ).tolist()

        x_fp8 = humming_ops.quant_input(
            x_bf16,
            "float8e4m3",
            group_size=ACT_SF_GRAN,
            scale_dtype="float32",
        )
        del x_bf16, scores

        l1_layer = _make_humming_layer(
            intermediate * 2,
            hidden,
            num_experts,
            cfg["weight_seed"],
            layer_tag=1,
        )
        l2_layer = _make_humming_layer(
            hidden,
            intermediate,
            num_experts,
            cfg["weight_seed"],
            layer_tag=2,
        )
        torch.cuda.empty_cache()

        l1_tuning = _indexed_tuning(l1_layer)
        l2_tuning = _indexed_tuning(l2_layer)
        l1_block = _choose_indexed_block_size(l1_tuning, m, num_topk)
        l1_sorted, l1_experts, l1_padded = _indexed_metadata(
            topk_idx,
            num_experts,
            l1_block,
        )
        l2_ids = topk_idx.reshape(-1, 1).contiguous()
        l2_block = _choose_indexed_block_size(l2_tuning, m * num_topk, 1)
        l2_sorted, l2_experts, l2_padded = _indexed_metadata(
            l2_ids,
            num_experts,
            l2_block,
        )
        compute_config = _compute_config()
        output_buffer = torch.empty(
            (m, hidden),
            dtype=torch.bfloat16,
            device="cuda",
        )
        # Router weights were applied before W2 quantization above, so reduction
        # uses unit weights while retaining Humming's fused reduction kernel.
        unit_route_weights = torch.ones_like(topk_weights)
        clamp = cfg["activation_clamp"]
        clamp_arg = clamp if math.isfinite(clamp) else None

        def run_humming_fusedmoe() -> torch.Tensor:
            l1_y = l1_layer(
                inputs=x_fp8[0],
                input_scale=x_fp8[1],
                sorted_ids=l1_sorted,
                expert_ids=l1_experts,
                num_tokens_padded=l1_padded,
                top_k=num_topk,
                compute_config=compute_config,
                tuning_config=l1_tuning,
            )
            l2_x, l2_scale = swiglu_apply_weight_to_fp8(
                l1_y,
                topk_weights,
                clamp_arg,
            )
            l2_y = l2_layer(
                inputs=l2_x,
                input_scale=l2_scale,
                sorted_ids=l2_sorted,
                expert_ids=l2_experts,
                num_tokens_padded=l2_padded,
                top_k=1,
                compute_config=compute_config,
                tuning_config=l2_tuning,
            )
            return moe_fused_mul_sum(
                l2_y.view(m, num_topk, hidden),
                unit_route_weights,
                outputs=output_buffer,
            )

        output = run_humming_fusedmoe()
        torch.cuda.synchronize()
        _barrier()
        if output.shape != (m, hidden) or output.dtype != torch.bfloat16:
            raise RuntimeError(
                f"invalid output: shape={tuple(output.shape)}, dtype={output.dtype}"
            )
        if not bool(torch.isfinite(output).all().item()):
            raise RuntimeError("Humming fused-MoE output contains non-finite values")

        free_bytes, _ = torch.cuda.mem_get_info()
        requested_flush = max(0, int(cfg["flush_l2_bytes"]))
        actual_flush = min(requested_flush, int(free_bytes * 0.5))
        actual_flush -= actual_flush % 4
        flush = (
            torch.empty(actual_flush // 4, dtype=torch.int32, device="cuda")
            if actual_flush
            else None
        )

        route = _route_summary(route_counts)
        logical_rows = m * num_topk
        l1_padded_rows = int(l1_padded.item())
        l2_padded_rows = int(l2_padded.item())
        padding = {
            "w13_block_m": l1_block,
            "w13_logical_rows": logical_rows,
            "w13_padded_rows": l1_padded_rows,
            "w13_padding_efficiency": (
                logical_rows / l1_padded_rows if l1_padded_rows else 1.0
            ),
            "w2_block_m": l2_block,
            "w2_logical_rows": logical_rows,
            "w2_padded_rows": l2_padded_rows,
            "w2_padding_efficiency": (
                logical_rows / l2_padded_rows if l2_padded_rows else 1.0
            ),
        }

        for observation in range(1, cfg["observations"] + 1):
            samples_us = _measure(
                run_humming_fusedmoe,
                flush,
                cfg["warmups"],
                cfg["samples"],
            )
            local_row = _local_timing_row(
                rank,
                cfg,
                samples_us,
                route,
                padding,
                requested_flush,
                actual_flush,
                observation,
            )
            print(
                "HUMMING_FUSEDMOE_STAT_JSON " + json.dumps(local_row, sort_keys=True),
                flush=True,
            )
            gathered = [None] * num_ranks
            dist.all_gather_object(gathered, local_row)
            if rank == 0:
                aggregate = _aggregate_timing_rows(gathered, cfg, observation)
                print(
                    "HUMMING_FUSEDMOE_OBSERVATION_JSON "
                    + json.dumps(aggregate, sort_keys=True),
                    flush=True,
                )
            _barrier()

        del flush
        completed = True
    finally:
        if dist.is_initialized():
            if completed:
                _barrier()
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", required=True, choices=tuple(SHAPES))
    parser.add_argument("--m", required=True, type=int)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--weight-seed", type=int, default=2027)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--observations", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--flush-l2-bytes", type=int, default=8_000_000_000)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    args = parser.parse_args()
    if args.m <= 0:
        parser.error("--m must be positive")
    if args.num_processes <= 0:
        parser.error("--num-processes must be positive")
    if args.observations <= 0 or args.warmups < 0 or args.samples <= 0:
        parser.error("observations/samples must be positive and warmups non-negative")
    if args.flush_l2_bytes < 0:
        parser.error("--flush-l2-bytes must be non-negative")

    torch.multiprocessing.spawn(
        _worker,
        args=(args.num_processes, vars(args)),
        nprocs=args.num_processes,
    )


if __name__ == "__main__":
    main()
