#!/usr/bin/env python3
"""SM90 DeepEP low-latency + Humming MXFP4A-FP8 MoE benchmark.

The timed path is:

    DeepEP low_latency_dispatch (FP8, fixed per-expert buffers)
      -> Humming MXFP4A-FP8 grouped-masked L1
      -> Triton masked SwiGLU and group-128 E4M3 quantization
      -> Humming MXFP4A-FP8 grouped-masked L2
      -> DeepEP low_latency_combine (top-k weighting)

Humming weight quantization/transform and JIT warmup are outside the samples.
The observation value is the median sample on each rank; rank 0 reports the
maximum rank median, matching the PR383 LL benchmark convention.
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
    from humming.config import GemmType
    from humming.layer import HummingLayer
except Exception as exc:  # pragma: no cover - exercised on the GPU host
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
_FP8_E4M3_MAX_TL = tl.constexpr(448.0)


@triton.jit
def _swiglu_masked_post_quant_kernel(
    x_ptr,
    stride_x_e,
    stride_x_m,
    stride_x_n,
    y_ptr,
    stride_y_e,
    stride_y_m,
    stride_y_n,
    y_sf_ptr,
    stride_sf_e,
    stride_sf_m,
    stride_sf_k,
    masked_m_ptr,
    H,
    clamp_value,
    HAS_CLAMP: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_e = tl.program_id(2)
    num_token_stripes = tl.num_programs(1)
    num_valid_tokens = tl.load(masked_m_ptr + pid_e)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    x_base = x_ptr + pid_e * stride_x_e + offs_k * stride_x_n
    y_base = y_ptr + pid_e * stride_y_e + offs_k * stride_y_n
    sf_base = y_sf_ptr + pid_e * stride_sf_e + pid_k * stride_sf_k
    for token in tl.range(
        pid_m, num_valid_tokens, num_token_stripes, num_stages=NUM_STAGES
    ):
        gate = tl.load(x_base + token * stride_x_m).to(tl.float32)
        up = tl.load(x_base + token * stride_x_m + H * stride_x_n).to(tl.float32)
        if HAS_CLAMP:
            gate = tl.minimum(gate, clamp_value)
            up = tl.minimum(tl.maximum(up, -clamp_value), clamp_value)
        y = gate * tl.sigmoid(gate) * up
        sf = tl.maximum(tl.max(tl.abs(y)) / _FP8_E4M3_MAX_TL, 1.0e-30)
        tl.store(y_base + token * stride_y_m, (y / sf).to(tl.float8e4nv))
        tl.store(sf_base + token * stride_sf_m, sf)


def swiglu_masked_post_quant_to_fp8(
    x: torch.Tensor,
    masked_m: torch.Tensor,
    clamp_value: float | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous()
    assert x.dim() == 3 and x.shape[-1] % 2 == 0
    num_experts, max_m, two_h = x.shape
    hidden = two_h // 2
    assert hidden % ACT_SF_GRAN == 0
    assert masked_m.shape == (num_experts,)

    y = torch.empty(
        (num_experts, max_m, hidden),
        dtype=torch.float8_e4m3fn,
        device=x.device,
    )
    y_sf = torch.empty(
        (num_experts, max_m, hidden // ACT_SF_GRAN),
        dtype=torch.float32,
        device=x.device,
    )
    token_stripes = 64 if num_experts < 4 else 32
    grid = (hidden // ACT_SF_GRAN, token_stripes, num_experts)
    _swiglu_masked_post_quant_kernel[grid](
        x,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        y,
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y_sf,
        y_sf.stride(0),
        y_sf.stride(1),
        y_sf.stride(2),
        masked_m,
        hidden,
        float(clamp_value) if clamp_value is not None else 0.0,
        HAS_CLAMP=clamp_value is not None,
        BLOCK_K=ACT_SF_GRAN,
        NUM_STAGES=4,
        num_warps=1,
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

        l1_layer = _make_humming_layer(
            intermediate * 2, hidden, experts_per_rank, l1_bf16
        )
        l2_layer = _make_humming_layer(
            hidden, intermediate, experts_per_rank, l2_bf16
        )
        del l1_bf16, l2_bf16, scores
        torch.cuda.empty_cache()

        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            cap, hidden, num_ranks, num_experts
        )
        buffer = deep_ep.Buffer(
            group,
            num_nvl_bytes=0,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=experts_per_rank,
            allow_nvlink_for_low_latency_mode=True,
            explicitly_destroy=True,
        )
        buffer.clean_low_latency_buffer(cap, hidden, num_experts)
        torch.cuda.synchronize()
        dist.barrier()

        max_m = cap * num_ranks
        l1_y = torch.empty(
            (experts_per_rank, max_m, intermediate * 2),
            dtype=torch.bfloat16,
            device="cuda",
        )
        l2_y = torch.empty(
            (experts_per_rank, max_m, hidden),
            dtype=torch.bfloat16,
            device="cuda",
        )
        combined = torch.empty((m, hidden), dtype=torch.bfloat16, device="cuda")
        clamp = cfg["activation_clamp"]
        clamp_arg = clamp if math.isfinite(clamp) else None
        l1_compute = _compute_config(GemmType.GROUPED_MASKED)
        l2_compute = _compute_config(GemmType.GROUPED_MASKED)
        valid_shape_m = m * num_topk
        last_masked_m = [None]

        def run_humming_ll():
            (recv_data, recv_scale), masked_m, handle, _, _ = (
                buffer.low_latency_dispatch(
                    x_bf16,
                    topk_idx,
                    cap,
                    num_experts,
                    use_fp8=True,
                    round_scale=False,
                    use_ue8m0=False,
                    async_finish=False,
                    return_recv_hook=False,
                )
            )
            last_masked_m[0] = masked_m
            l1_layer(
                inputs=recv_data.flatten(0, 1),
                outputs=l1_y.flatten(0, 1),
                input_scale=recv_scale.flatten(0, 1),
                expert_layout=masked_m,
                top_k=1,
                valid_shape_m=valid_shape_m,
                compute_config=l1_compute,
            )
            l2_data, l2_scale = swiglu_masked_post_quant_to_fp8(
                l1_y, masked_m, clamp_arg
            )
            l2_layer(
                inputs=l2_data.flatten(0, 1),
                outputs=l2_y.flatten(0, 1),
                input_scale=l2_scale.flatten(0, 1),
                expert_layout=masked_m,
                top_k=1,
                valid_shape_m=valid_shape_m,
                compute_config=l2_compute,
            )
            output, _, _ = buffer.low_latency_combine(
                l2_y,
                topk_idx,
                topk_weights,
                handle,
                use_logfmt=False,
                zero_copy=False,
                async_finish=False,
                return_recv_hook=False,
                out=combined,
            )
            return output

        output = run_humming_ll()
        torch.cuda.synchronize()
        dist.barrier()
        if output.shape != (m, hidden) or output.dtype != torch.bfloat16:
            raise RuntimeError(
                f"invalid output shape/dtype: {tuple(output.shape)}, {output.dtype}"
            )
        if not bool(torch.isfinite(output).all().item()):
            raise RuntimeError("Humming LL output contains non-finite values")
        route_counts = last_masked_m[0].to(torch.int).clone()
        if not torch.equal(route_counts, expected_routes):
            raise RuntimeError(
                f"route mismatch: actual={route_counts.tolist()}, "
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

        route_list = route_counts.tolist()
        for observation in range(1, cfg["observations"] + 1):
            for _ in range(cfg["warmups"]):
                run_humming_ll()
            torch.cuda.synchronize()
            dist.barrier()

            samples_us = []
            for _ in range(cfg["samples"]):
                if flush is not None:
                    flush.zero_()
                torch.cuda.synchronize()
                dist.barrier()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                run_humming_ll()
                end.record()
                end.synchronize()
                samples_us.append(start.elapsed_time(end) * 1000.0)

            observed_routes = last_masked_m[0].to(torch.int).tolist()
            if observed_routes != route_list:
                raise RuntimeError("LL routes changed across calls")
            returned_us = statistics.median(samples_us)
            local_row = {
                "backend": "humming_mxfp4afp8",
                "mode": "ll_grouped_masked",
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
                "max_m": max_m,
                "valid_shape_m": valid_shape_m,
                "route_counts": route_list,
                "route_total": sum(route_list),
                "touched_experts": sum(value > 0 for value in route_list),
                "requested_flush_l2_bytes": requested_flush,
                "actual_flush_l2_bytes": actual_flush,
            }
            print(
                "HUMMING_LL_STAT_JSON " + json.dumps(local_row, sort_keys=True),
                flush=True,
            )

            gathered = [None] * num_ranks
            dist.all_gather_object(gathered, local_row)
            if rank == 0:
                max_row = max(gathered, key=lambda row: row["returned_us"])
                aggregate = {
                    "backend": "humming_mxfp4afp8",
                    "mode": "ll_grouped_masked",
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
                        }
                        for row in sorted(gathered, key=lambda row: row["rank"])
                    ],
                    "requested_flush_l2_bytes": requested_flush,
                    "actual_flush_l2_bytes_min": min(
                        row["actual_flush_l2_bytes"] for row in gathered
                    ),
                }
                print(
                    "HUMMING_LL_OBSERVATION_JSON "
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
    parser.add_argument("--cap", required=True, type=int)
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
