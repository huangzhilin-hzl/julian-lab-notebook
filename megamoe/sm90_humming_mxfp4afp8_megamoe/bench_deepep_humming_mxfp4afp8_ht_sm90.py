#!/usr/bin/env python3
"""SM90 DeepEP high-throughput + Humming MXFP4A-FP8 MoE benchmark.

The timed path is:

    DeepEP ElasticBuffer FP8 dispatch (expanded, contiguous experts)
      -> Humming MXFP4A-FP8 grouped-contiguous L1
      -> Triton SwiGLU, top-k weighting, and group-128 E4M3 quantization
      -> Humming MXFP4A-FP8 grouped-contiguous L2
      -> DeepEP combine

Humming weight quantization/transform and JIT warmup are outside the samples.
The observation value is the median sample on each rank; rank 0 reports the
maximum rank median, matching the PR383 HT benchmark convention.
"""

import argparse
import json
import math
import os
import random
import statistics
from typing import Tuple

import torch
import torch.distributed as dist
import triton
import triton.language as tl

try:
    import deep_ep
except Exception as exc:  # pragma: no cover - exercised on the GPU host
    deep_ep = None
    _DEEP_EP_IMPORT_ERROR = exc
else:
    _DEEP_EP_IMPORT_ERROR = None

try:
    from humming import ops as humming_ops
    from humming.config import GemmType
    from humming.layer import HummingLayer
except Exception as exc:  # pragma: no cover - exercised on the GPU host
    humming_ops = None
    GemmType = None
    HummingLayer = None
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

ACT_SF_GRAN = 128
EXPERT_ALIGNMENT = 128
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
    up_ptrs = (
        x_ptr
        + offs_m[:, None] * stride_xm
        + (H + offs_k[None, :]) * stride_xn
    )
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
    return dist.get_rank(), dist.get_world_size(), dist.group.WORLD


def _make_humming_layer(
    shape_n: int,
    shape_k: int,
    num_experts: int,
    unquantized_weight: torch.Tensor,
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
    layer.load_from_unquantized(unquantized_weight)
    layer.transform()
    return layer


def _compute_config(gemm_type) -> dict:
    return {
        "gemm_type": gemm_type.value,
        "use_f16_accum": False,
        "use_m_major_input_scale": False,
    }


def _expected_local_routes(
    topk_idx: torch.Tensor,
    num_experts: int,
    rank: int,
    num_ranks: int,
    group,
) -> torch.Tensor:
    counts = torch.bincount(topk_idx.reshape(-1), minlength=num_experts)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)
    experts_per_rank = num_experts // num_ranks
    return counts[
        rank * experts_per_rank : (rank + 1) * experts_per_rank
    ].to(torch.int)


def _aligned_expert_offsets(handle) -> torch.Tensor:
    psum = handle.psum_num_recv_tokens_per_expert
    aligned_ends = (psum + EXPERT_ALIGNMENT - 1) // EXPERT_ALIGNMENT
    aligned_ends *= EXPERT_ALIGNMENT
    offsets = torch.cat((torch.zeros_like(aligned_ends[:1]), aligned_ends))
    return offsets


def _barrier(buffer) -> None:
    buffer.barrier(use_comm_stream=False)


def _worker(local_rank: int, num_processes: int, cfg: dict):
    if deep_ep is None:
        raise RuntimeError(f"DeepEP import failed: {_DEEP_EP_IMPORT_ERROR}")
    if HummingLayer is None:
        raise RuntimeError(f"Humming import failed: {_HUMMING_IMPORT_ERROR}")

    rank, num_ranks, group = _init_dist(local_rank, num_processes)
    buffer = None
    completed = False
    try:
        major, minor = torch.cuda.get_device_capability()
        if major != 9:
            raise RuntimeError(f"SM90 required, got SM{major}{minor}")

        shape = SHAPES[cfg["shape"]]
        m = cfg["m"]
        cap = cfg["cap"]
        hidden = shape["hidden"]
        intermediate = shape["intermediate_hidden"]
        num_experts = shape["num_experts"]
        num_topk = shape["num_topk"]
        if num_experts % num_ranks != 0 or m > cap:
            raise RuntimeError("num_experts must divide world size and m must not exceed cap")
        experts_per_rank = num_experts // num_ranks

        torch.manual_seed(cfg["seed"] + rank)
        random.seed(cfg["seed"] + rank)
        x_bf16 = torch.randn((m, hidden), dtype=torch.bfloat16, device="cuda")
        l1_bf16 = torch.randn(
            (experts_per_rank, intermediate * 2, hidden),
            dtype=torch.bfloat16,
            device="cuda",
        ) * 0.05
        l2_bf16 = torch.randn(
            (experts_per_rank, hidden, intermediate),
            dtype=torch.bfloat16,
            device="cuda",
        ) * 0.05
        scores = torch.randn((m, num_experts), dtype=torch.float32, device="cuda")
        topk_weights, topk_idx = torch.topk(
            scores, num_topk, dim=-1, largest=True, sorted=False
        )
        topk_idx = topk_idx.to(torch.int64)
        expected_routes = _expected_local_routes(
            topk_idx, num_experts, rank, num_ranks, group
        )

        x_fp8 = humming_ops.quant_input(
            x_bf16,
            "float8e4m3",
            group_size=ACT_SF_GRAN,
            scale_dtype="float32",
        )
        l1_layer = _make_humming_layer(
            intermediate * 2, hidden, experts_per_rank, l1_bf16
        )
        l2_layer = _make_humming_layer(
            hidden, intermediate, experts_per_rank, l2_bf16
        )
        del x_bf16, l1_bf16, l2_bf16, scores
        torch.cuda.empty_cache()

        buffer = deep_ep.ElasticBuffer(
            group,
            num_max_tokens_per_rank=cap,
            hidden=hidden,
            num_topk=num_topk,
            use_fp8_dispatch=True,
            explicitly_destroy=True,
            allow_multiple_reduction=False,
            num_gpu_timeout_secs=10,
            num_cpu_timeout_secs=30,
        )
        cumulative_recv = torch.zeros(
            (experts_per_rank,), dtype=torch.int, device="cuda"
        )
        clamp = cfg["activation_clamp"]
        clamp_arg = clamp if math.isfinite(clamp) else None
        l1_compute = _compute_config(GemmType.GROUPED_CONTIGUOUS)
        l2_compute = _compute_config(GemmType.GROUPED_CONTIGUOUS)
        last_state = {}

        def run_humming_ht():
            recv_x, _, recv_topk_weights, handle, _ = buffer.dispatch(
                x_fp8,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                cumulative_local_expert_recv_stats=cumulative_recv,
                num_experts=num_experts,
                expert_alignment=EXPERT_ALIGNMENT,
                do_cpu_sync=True,
                do_handle_copy=False,
                do_expand=True,
                use_tma_aligned_col_major_sf=False,
            )
            recv_data, recv_scale = recv_x
            num_rows = recv_data.size(0)
            expert_offsets = _aligned_expert_offsets(handle)
            l1_y = l1_layer(
                inputs=recv_data,
                input_scale=recv_scale,
                expert_layout=expert_offsets,
                top_k=1,
                compute_config=l1_compute,
            )
            l2_x, l2_scale = swiglu_apply_weight_to_fp8(
                l1_y, recv_topk_weights, clamp_arg
            )
            l2_y = l2_layer(
                inputs=l2_x,
                input_scale=l2_scale,
                expert_layout=expert_offsets,
                top_k=1,
                compute_config=l2_compute,
            )
            last_state["handle"] = handle
            last_state["num_rows"] = num_rows
            last_state["expert_offsets"] = expert_offsets
            return buffer.combine(l2_y, handle=handle)[0]

        output = run_humming_ht()
        torch.cuda.synchronize()
        _barrier(buffer)
        if output.shape != (m, hidden) or output.dtype != torch.bfloat16:
            raise RuntimeError(
                f"invalid output shape/dtype: {tuple(output.shape)}, {output.dtype}"
            )
        if not bool(torch.isfinite(output).all().item()):
            raise RuntimeError("Humming HT output contains non-finite values")
        layout_end = int(last_state["expert_offsets"][-1].item())
        if layout_end != last_state["num_rows"]:
            raise RuntimeError(
                "DeepEP expanded rows/layout mismatch: "
                f"rows={last_state['num_rows']}, layout_end={layout_end}"
            )
        actual_routes = last_state["handle"].num_unaligned_recv_tokens_per_expert
        actual_routes = actual_routes.to(torch.int)
        if not torch.equal(actual_routes, expected_routes):
            raise RuntimeError(
                f"route mismatch: actual={actual_routes.tolist()}, "
                f"expected={expected_routes.tolist()}"
            )

        free_bytes, _ = torch.cuda.mem_get_info()
        requested_flush = max(0, int(cfg["flush_l2_bytes"]))
        actual_flush = min(requested_flush, int(free_bytes * 0.5))
        actual_flush -= actual_flush % 4
        flush = (
            torch.empty(actual_flush // 4, dtype=torch.int32, device="cuda")
            if actual_flush
            else None
        )

        route_list = actual_routes.tolist()
        for observation in range(1, cfg["observations"] + 1):
            for _ in range(cfg["warmups"]):
                run_humming_ht()
            torch.cuda.synchronize()
            _barrier(buffer)

            samples_us = []
            for _ in range(cfg["samples"]):
                if flush is not None:
                    flush.zero_()
                torch.cuda.synchronize()
                _barrier(buffer)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                run_humming_ht()
                end.record()
                end.synchronize()
                samples_us.append(start.elapsed_time(end) * 1000.0)

            observed_routes = (
                last_state["handle"]
                .num_unaligned_recv_tokens_per_expert.to(torch.int)
                .tolist()
            )
            if observed_routes != route_list:
                raise RuntimeError("HT routes changed across calls")
            returned_us = statistics.median(samples_us)
            offsets = last_state["expert_offsets"].tolist()
            local_row = {
                "backend": "humming_mxfp4afp8",
                "mode": "ht_grouped_contiguous",
                "shape": cfg["shape"],
                "m": m,
                "cap": cap,
                "seed": cfg["seed"],
                "observation": observation,
                "rank": rank,
                "num_samples": len(samples_us),
                "returned_us": returned_us,
                "mean_us": statistics.mean(samples_us),
                "min_us": min(samples_us),
                "max_us": max(samples_us),
                "samples_us": samples_us,
                "route_counts": route_list,
                "route_total": sum(route_list),
                "touched_experts": sum(value > 0 for value in route_list),
                "expanded_buffer_rows": last_state["num_rows"],
                "padded_expert_rows": offsets[-1],
                "expert_alignment": EXPERT_ALIGNMENT,
                "requested_flush_l2_bytes": requested_flush,
                "actual_flush_l2_bytes": actual_flush,
            }
            print(
                "HUMMING_HT_STAT_JSON " + json.dumps(local_row, sort_keys=True),
                flush=True,
            )

            gathered = [None] * num_ranks
            dist.all_gather_object(gathered, local_row)
            if rank == 0:
                max_row = max(gathered, key=lambda row: row["returned_us"])
                aggregate = {
                    "backend": "humming_mxfp4afp8",
                    "mode": "ht_grouped_contiguous",
                    "shape": cfg["shape"],
                    "m": m,
                    "seed": cfg["seed"],
                    "observation": observation,
                    "num_samples": cfg["samples"],
                    "max_rank": max_row["rank"],
                    "max_rank_us": max_row["returned_us"],
                    "per_rank_us": [
                        row["returned_us"]
                        for row in sorted(gathered, key=lambda row: row["rank"])
                    ],
                    "routes": [
                        {
                            "rank": row["rank"],
                            "route_counts": row["route_counts"],
                            "route_total": row["route_total"],
                            "touched_experts": row["touched_experts"],
                            "expanded_buffer_rows": row["expanded_buffer_rows"],
                        }
                        for row in sorted(gathered, key=lambda row: row["rank"])
                    ],
                    "requested_flush_l2_bytes": requested_flush,
                    "actual_flush_l2_bytes_min": min(
                        row["actual_flush_l2_bytes"] for row in gathered
                    ),
                }
                print(
                    "HUMMING_HT_OBSERVATION_JSON "
                    + json.dumps(aggregate, sort_keys=True),
                    flush=True,
                )
            dist.barrier()

        del flush
        completed = True
    finally:
        if buffer is not None:
            buffer.destroy()
        if dist.is_initialized():
            if completed:
                dist.barrier()
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", required=True, choices=tuple(SHAPES))
    parser.add_argument("--m", required=True, type=int)
    parser.add_argument("--cap", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--observations", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--flush-l2-bytes", type=int, default=8_000_000_000)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    args = parser.parse_args()
    if args.m <= 0 or args.cap < args.m:
        parser.error("--m must be positive and --cap must be at least --m")
    if args.observations <= 0 or args.warmups < 0 or args.samples <= 0:
        parser.error("observations/samples must be positive and warmups non-negative")

    torch.multiprocessing.spawn(
        _worker,
        args=(args.num_processes, vars(args)),
        nprocs=args.num_processes,
    )


if __name__ == "__main__":
    main()
