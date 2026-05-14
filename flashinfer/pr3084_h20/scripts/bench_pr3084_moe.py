#!/usr/bin/env python3
"""Benchmark FlashInfer PR #3084 SM90 mixed-input MoE path on H20.

The script intentionally adapts to the imported FlashInfer package:

- pre-PR packages do not expose SM90 interleave helpers, so it benchmarks the
  legacy W4A16 layout directly.
- PR #3084 and newer packages expose the helpers, so it applies weight and
  scale interleave before calling cutlass_fused_moe.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def compute_routing(router_logits: torch.Tensor, top_k: int):
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    return routing_weights.float(), selected_experts


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def build_case(batch: int, hidden: int, intermediate: int, experts: int, topk: int):
    torch.manual_seed(42)
    device = torch.device("cuda")
    x = torch.randn(batch, hidden, dtype=torch.bfloat16, device=device)
    w1 = torch.randint(
        0, 256, (experts, 2 * intermediate, hidden // 2), device=device, dtype=torch.uint8
    )
    w2 = torch.randint(
        0, 256, (experts, hidden, intermediate // 2), device=device, dtype=torch.uint8
    )
    w1_scale = torch.randint(
        118, 123, (experts, 2 * intermediate, hidden // 32), device=device, dtype=torch.uint8
    )
    w2_scale = torch.randint(
        118, 123, (experts, hidden, intermediate // 32), device=device, dtype=torch.uint8
    )
    router_logits = torch.randn(batch, experts, dtype=torch.bfloat16, device=device)
    routing_weights, selected_experts = compute_routing(router_logits, topk)
    return x, w1, w2, w1_scale, w2_scale, routing_weights, selected_experts


def maybe_apply_pr3084_layout(fused_moe, w1, w2, w1_scale, w2_scale):
    has_weight_helper = hasattr(fused_moe, "interleave_moe_weights_for_sm90_mixed_gemm")
    has_scale_helper = hasattr(fused_moe, "interleave_moe_scales_for_sm90_mixed_gemm")
    if has_weight_helper and has_scale_helper:
        return (
            fused_moe.interleave_moe_weights_for_sm90_mixed_gemm(w1, "fp4"),
            fused_moe.interleave_moe_weights_for_sm90_mixed_gemm(w2, "fp4"),
            fused_moe.interleave_moe_scales_for_sm90_mixed_gemm(w1_scale),
            fused_moe.interleave_moe_scales_for_sm90_mixed_gemm(w2_scale),
            True,
        )
    return w1, w2, w1_scale, w2_scale, False


def run_case(args, fused_moe, bench_gpu_time, autotune_ctx, batch: int, use_autotune: bool):
    hidden = args.hidden
    intermediate = args.intermediate
    experts = args.experts
    topk = args.topk

    x, w1, w2, w1_scale, w2_scale, routing_weights, selected_experts = build_case(
        batch, hidden, intermediate, experts, topk
    )
    w1_run, w2_run, w1_scale_run, w2_scale_run, used_interleave = maybe_apply_pr3084_layout(
        fused_moe, w1, w2, w1_scale, w2_scale
    )
    output = torch.zeros_like(x)
    quant_scales = [w1_scale_run.view(torch.int32), w2_scale_run.view(torch.int32)]

    def kernel():
        return fused_moe.cutlass_fused_moe(
            x,
            selected_experts.to(torch.int),
            routing_weights,
            w1_run,
            w2_run,
            torch.bfloat16,
            quant_scales=quant_scales,
            use_w4_group_scaling=True,
            output=output,
            tune_max_num_tokens=args.tune_max_num_tokens,
        )

    # Compile and warm the JIT module outside measured timing.
    kernel()
    torch.cuda.synchronize()

    if use_autotune:
        with torch.inference_mode(), autotune_ctx(True):
            kernel()
        torch.cuda.synchronize()

    times = bench_gpu_time(
        kernel,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.num_iters,
        enable_cupti=args.use_cupti,
        use_cuda_graph=False,
        cold_l2_cache=args.cold_l2,
    )
    median_ms = float(statistics.median(times))
    std_ms = float(statistics.pstdev(times)) if len(times) > 1 else 0.0
    flops = 6.0 * batch * topk * hidden * intermediate
    tflops = flops / (median_ms * 1e9)
    torch.cuda.synchronize()
    max_abs = float(output.float().abs().max().item())
    return {
        "batch": batch,
        "hidden": hidden,
        "intermediate": intermediate,
        "experts": experts,
        "topk": topk,
        "autotune": use_autotune,
        "used_pr3084_interleave": used_interleave,
        "median_ms": median_ms,
        "std_ms": std_ms,
        "p20_ms": quantile([float(x) for x in times], 0.20),
        "p80_ms": quantile([float(x) for x in times], 0.80),
        "min_ms": min(float(x) for x in times),
        "max_ms": max(float(x) for x in times),
        "approx_tflops": tflops,
        "output_max_abs": max_abs,
        "num_iters": len(times),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="flashinfer")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--num-iters", type=int, default=30)
    parser.add_argument("--dry-run-iters", type=int, default=5)
    parser.add_argument("--tune-max-num-tokens", type=int, default=16384)
    parser.add_argument("--use-cupti", action="store_true")
    parser.add_argument("--cold-l2", action="store_true")
    parser.add_argument("--no-autotune", action="store_true")
    parser.add_argument("--autotune-only", action="store_true")
    args = parser.parse_args()

    import flashinfer
    import flashinfer.fused_moe as fused_moe
    from flashinfer.autotuner import autotune
    from flashinfer.testing import bench_gpu_time

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "label": args.label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "flashinfer_version": getattr(flashinfer, "__version__", None),
        "flashinfer_file": getattr(flashinfer, "__file__", None),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": torch.cuda.get_device_capability(0),
        "env": {
            "FLASHINFER_WORKSPACE_BASE": os.getenv("FLASHINFER_WORKSPACE_BASE"),
            "FLASHINFER_CUDA_ARCH_LIST": os.getenv("FLASHINFER_CUDA_ARCH_LIST"),
            "FLASHINFER_NVCC_THREADS": os.getenv("FLASHINFER_NVCC_THREADS"),
            "FLASHINFER_DISABLE_VERSION_CHECK": os.getenv("FLASHINFER_DISABLE_VERSION_CHECK"),
        },
        "has_weight_interleave_helper": hasattr(
            fused_moe, "interleave_moe_weights_for_sm90_mixed_gemm"
        ),
        "has_scale_interleave_helper": hasattr(
            fused_moe, "interleave_moe_scales_for_sm90_mixed_gemm"
        ),
    }

    modes = []
    if not args.autotune_only:
        modes.append(False)
    if not args.no_autotune:
        modes.append(True)

    rows = []
    for batch in args.batches:
        for use_autotune in modes:
            print(
                f"[RUN] label={args.label} batch={batch} autotune={use_autotune}",
                flush=True,
            )
            row = run_case(args, fused_moe, bench_gpu_time, autotune, batch, use_autotune)
            row["label"] = args.label
            rows.append(row)
            print(
                "[RESULT] "
                f"batch={batch} autotune={use_autotune} "
                f"median_ms={row['median_ms']:.4f} "
                f"approx_tflops={row['approx_tflops']:.2f} "
                f"interleave={row['used_pr3084_interleave']}",
                flush=True,
            )

    (output_dir / f"{args.label}_env.json").write_text(json.dumps(env, indent=2) + "\n")
    (output_dir / f"{args.label}_results.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    with (output_dir / f"{args.label}_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
