#!/usr/bin/env python3
"""PR383-style workload benchmark for DeepGEMM PR #36 SM90 MegaMoE.

This benchmark adapts the fused-only path from sgl-project/DeepGEMM PR #36
(``tests/test_mega_moe_hopper.py``) to the local MegaMoE workload contract:

* EP8 by default, with ``M`` interpreted as tokens per rank;
* Flash: H=4096, I=2048, E=256, top-k=6;
* Pro: H=7168, I=3072, E=384, top-k=6;
* M sweep: 8,16,32,64,128,256,512,1024,2048,4096,8192;
* workspace capacity ``cap=M`` at every point by default;
* seed 101, 3 observations, 5 warmups, 20 cold-L2 samples per observation,
  and slowest-rank aggregation.

PR #36 is an FP8 x FP8 SM90 fused MegaMoE implementation. It is not the
FP8 x MXFP4 Humming path despite this file living beside that study.

The default ``kernel`` timing scope matches the PR's fused-only benchmark:
FP8 input/weight preparation, weight transformation, SymmBuffer allocation,
and copies into the SymmBuffer are outside timing. CUDA events cover the full
``deep_gemm.fp8_mega_moe(...)`` call, whose persistent kernel includes token
dispatch, FC1, SwiGLU, FC2, and combine. Use ``--timing-scope forward`` to
include the four local input copies before every fused call.

Requirements:

* sgl-project/DeepGEMM PR #36 head
  ``3f9268b5c15d4b939957051a1b5d22d2ef3dcf4e`` built with
  ``bash build_sgl_deep_gemm.sh``;
* Hopper SM90 GPUs and a working single-node symmetric-memory/NVLink setup;
* launch with torchrun, for example:

  ``torchrun --standalone --nproc_per_node=8 bench_deepgemm_pr36_megamoe_sm90.py``

Useful shorter run:

  ``torchrun --standalone --nproc_per_node=8 bench_deepgemm_pr36_megamoe_sm90.py --model-config flash --batches 8 16 32 --observations 1``
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import statistics
from typing import Callable


PR36_HEAD = "3f9268b5c15d4b939957051a1b5d22d2ef3dcf4e"
BACKEND = "deepgemm_pr36_sm90_fp8_megamoe"
KERNEL_NAME = "sm90_fp8_mega_moe_impl"
FP8_E4M3_MAX = 448.0
RECIPE = (128, 128, 128)

MODEL_CONFIGS = {
    "flash": {
        "hidden": 4096,
        "intermediate": 2048,
        "num_experts": 256,
        "top_k": 6,
    },
    "pro": {
        "hidden": 7168,
        "intermediate": 3072,
        "num_experts": 384,
        "top_k": 6,
    },
}

DEFAULT_BATCHES = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model-config",
        nargs="+",
        choices=tuple(MODEL_CONFIGS),
        default=list(MODEL_CONFIGS),
    )
    parser.add_argument("--batches", nargs="+", type=int, default=list(DEFAULT_BATCHES))
    parser.add_argument(
        "--num-max-tokens-per-rank",
        type=int,
        default=8192,
        help="fixed requested capacity used only with --no-match-cap-to-m",
    )
    parser.add_argument(
        "--match-cap-to-m",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="request cap=M at each point; PR36 may align the internal capacity",
    )
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--observations", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--num-tests", type=int, default=20)
    parser.add_argument("--flush-l2-bytes", type=int, default=8_000_000_000)
    parser.add_argument(
        "--barrier-sleep-cycles",
        type=int,
        default=20_000_000,
        help="GPU sleep before each rank barrier, matching bench_kineto discipline",
    )
    parser.add_argument(
        "--timing-scope",
        choices=("kernel", "forward"),
        default="kernel",
        help="kernel times fp8_mega_moe only; forward also times SymmBuffer input copies",
    )
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--fast-math", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--masked-ratio",
        type=float,
        default=0.0,
        help="randomly mask this fraction of top-k routes",
    )
    args = parser.parse_args()

    if not args.batches or min(args.batches) <= 0:
        parser.error("--batches must contain positive values")
    if not args.match_cap_to_m and args.num_max_tokens_per_rank < max(args.batches):
        parser.error("--num-max-tokens-per-rank must cover the largest batch")
    if args.expected_world_size <= 0:
        parser.error("--expected-world-size must be positive")
    if min(args.observations, args.num_tests) <= 0:
        parser.error("--observations and --num-tests must be positive")
    if min(args.warmups, args.flush_l2_bytes, args.barrier_sleep_cycles) < 0:
        parser.error("warmups, flush size, and sleep cycles must be non-negative")
    if not 0.0 <= args.masked_ratio < 1.0:
        parser.error("--masked-ratio must be in [0, 1)")
    if args.timing_scope == "kernel" and os.getenv("DG_COMM_KERNEL_DEBUG", "0") == "1":
        parser.error(
            "kernel scope is incompatible with DG_COMM_KERNEL_DEBUG=1 because "
            "the debug kernel clears staged inputs; use --timing-scope forward"
        )
    return args


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_identity(deep_gemm) -> dict:
    module_file = os.path.realpath(deep_gemm.__file__)
    package_dir = os.path.dirname(module_file)
    artifacts = {}
    for label, path in (
        ("python_module", module_file),
        ("native_extension", os.path.join(package_dir, "_C.so")),
    ):
        if os.path.isfile(path):
            artifacts[label] = {
                "filename": os.path.basename(path),
                "sha256": _sha256_file(path),
            }

    declared_revision = getattr(deep_gemm, "__git_commit__", None)
    revision_source = "deep_gemm.__git_commit__"
    if not declared_revision:
        declared_revision = os.getenv("DEEPGEMM_BUILD_REVISION")
        revision_source = "DEEPGEMM_BUILD_REVISION" if declared_revision else None
    if declared_revision is not None:
        declared_revision = str(declared_revision)

    return {
        "version": getattr(deep_gemm, "__version__", "unknown"),
        "declared_revision": declared_revision,
        "revision_source": revision_source,
        "revision_matches_target": (
            declared_revision == PR36_HEAD if declared_revision is not None else None
        ),
        "artifacts": artifacts,
    }


def _quantize_grouped_fp8_block_128_128(w):
    """Quantize (G, N, K) BF16 weights using PR36's block-FP8 layout."""
    import torch

    g, n, k = w.shape
    if n % 128 or k % 128:
        raise ValueError(f"weight N={n}, K={k} must be multiples of 128")
    w_view = w.view(g, n // 128, 128, k // 128, 128).float()
    amax = w_view.abs().amax(dim=(-1, -3)).clamp(1e-4)
    scale = amax / FP8_E4M3_MAX
    w_fp8 = (w_view / scale.unsqueeze(-1).unsqueeze(-3)).to(torch.float8_e4m3fn)
    return w_fp8.view(g, n, k).contiguous(), scale.contiguous()


def _make_case_data(args, model: dict, m: int, rank: int, world_size: int, device):
    import torch
    from deep_gemm.utils import per_token_cast_to_fp8

    local_experts = model["num_experts"] // world_size
    generator = torch.Generator(device=device).manual_seed(args.seed + rank)

    # Match PR383's per-point RNG order: x, W1, W2, then routing scores.
    x_bf16 = torch.randn(
        (m, model["hidden"]),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    l1_bf16 = torch.randn(
        (local_experts, 2 * model["intermediate"], model["hidden"]),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.05)
    l2_bf16 = torch.randn(
        (local_experts, model["hidden"], model["intermediate"]),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.05)
    scores = torch.randn(
        (m, model["num_experts"]),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    topk_weights, topk_ids = torch.topk(
        scores,
        model["top_k"],
        dim=-1,
        largest=True,
        sorted=False,
    )
    if args.masked_ratio:
        route_mask = (
            torch.rand(
                topk_ids.shape,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            < args.masked_ratio
        )
        topk_ids.masked_fill_(route_mask, -1)
        topk_weights.masked_fill_(route_mask, 0.0)

    x_fp8 = per_token_cast_to_fp8(
        x_bf16,
        use_ue8m0=False,
        gran_k=128,
        use_packed_ue8m0=False,
    )
    l1_weights = _quantize_grouped_fp8_block_128_128(l1_bf16)
    del l1_bf16
    l2_weights = _quantize_grouped_fp8_block_128_128(l2_bf16)
    del x_bf16, l2_bf16, scores

    return x_fp8, topk_ids, topk_weights, l1_weights, l2_weights


def _expected_local_routes(topk_ids, num_experts: int, rank: int, world_size: int):
    import torch
    import torch.distributed as dist

    valid_ids = topk_ids[topk_ids >= 0]
    counts = torch.bincount(valid_ids, minlength=num_experts)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    local_experts = num_experts // world_size
    start = rank * local_experts
    return counts[start : start + local_experts].to(torch.int)


def _allocate_flush(requested_bytes: int):
    import torch

    free_bytes, _ = torch.cuda.mem_get_info()
    actual_bytes = min(requested_bytes, int(free_bytes * 0.5))
    actual_bytes -= actual_bytes % 4
    tensor = (
        torch.empty(actual_bytes // 4, dtype=torch.int32, device="cuda")
        if actual_bytes
        else None
    )
    return tensor, actual_bytes


def _time_samples(
    call: Callable,
    *,
    warmups: int,
    num_tests: int,
    flush,
    barrier_sleep_cycles: int,
) -> list[float]:
    import torch
    import torch.distributed as dist

    for _ in range(warmups):
        call()
    torch.cuda.synchronize()
    dist.barrier()

    samples: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(num_tests):
        if flush is not None:
            flush.zero_()
        if barrier_sleep_cycles:
            torch.cuda._sleep(barrier_sleep_cycles)
        torch.cuda.synchronize()
        dist.barrier()
        start.record()
        call()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) * 1_000.0)
    return samples


def _emit_observation(
    *,
    args,
    model_name: str,
    model: dict,
    m: int,
    requested_cap: int,
    actual_cap: int,
    observation: int,
    samples_us: list[float],
    routes,
    actual_flush_bytes: int,
    rank: int,
    world_size: int,
) -> float | None:
    import torch.distributed as dist

    route_list = routes.tolist()
    local_row = {
        "backend": BACKEND,
        "kernel_name": KERNEL_NAME,
        "series": args.timing_scope,
        "model": model_name,
        "m": m,
        "requested_cap": requested_cap,
        "actual_cap": actual_cap,
        "rank": rank,
        "world_size": world_size,
        "hidden": model["hidden"],
        "intermediate": model["intermediate"],
        "num_experts": model["num_experts"],
        "top_k": model["top_k"],
        "seed": args.seed,
        "observation": observation,
        "num_samples": len(samples_us),
        "returned_us": statistics.median(samples_us),
        "mean_us": statistics.mean(samples_us),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
        "samples_us": samples_us,
        "route_counts": route_list,
        "route_counts_source": "input_all_reduce_verified_against_kernel_stats",
        "route_total": sum(route_list),
        "touched_experts": sum(value > 0 for value in route_list),
        "recipe": list(RECIPE),
        "activation": "swiglu",
        "activation_clamp": args.activation_clamp,
        "fast_math": bool(args.fast_math),
        "masked_ratio": args.masked_ratio,
        "requested_flush_l2_bytes": args.flush_l2_bytes,
        "actual_flush_l2_bytes": actual_flush_bytes,
    }
    print(
        "DEEPGEMM_PR36_STAT_JSON " + json.dumps(local_row, sort_keys=True), flush=True
    )

    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_row)
    if rank != 0:
        return None

    max_row = max(gathered, key=lambda row: row["returned_us"])
    aggregate = {
        "backend": BACKEND,
        "kernel_name": KERNEL_NAME,
        "series": args.timing_scope,
        "model": model_name,
        "m": m,
        "requested_cap": requested_cap,
        "actual_cap": actual_cap,
        "world_size": world_size,
        "seed": args.seed,
        "observation": observation,
        "num_samples": len(samples_us),
        "max_rank": max_row["rank"],
        "max_rank_us": max_row["returned_us"],
        "per_rank_us": [
            row["returned_us"] for row in sorted(gathered, key=lambda row: row["rank"])
        ],
        "routes": [
            {
                "rank": row["rank"],
                "route_counts": row["route_counts"],
                "route_counts_source": row["route_counts_source"],
                "route_total": row["route_total"],
                "touched_experts": row["touched_experts"],
            }
            for row in sorted(gathered, key=lambda row: row["rank"])
        ],
        "recipe": list(RECIPE),
        "requested_flush_l2_bytes": args.flush_l2_bytes,
        "actual_flush_l2_bytes_min": min(
            row["actual_flush_l2_bytes"] for row in gathered
        ),
    }
    print(
        "DEEPGEMM_PR36_OBSERVATION_JSON " + json.dumps(aggregate, sort_keys=True),
        flush=True,
    )
    return aggregate["max_rank_us"]


def _run_series(
    call: Callable,
    *,
    args,
    model_name: str,
    model: dict,
    m: int,
    requested_cap: int,
    actual_cap: int,
    routes,
    cumulative_stats,
    rank: int,
    world_size: int,
) -> None:
    import torch
    import torch.distributed as dist

    output = call()
    torch.cuda.synchronize()
    dist.barrier()
    if tuple(output.shape) != (m, model["hidden"]):
        raise RuntimeError(
            f"invalid output shape {tuple(output.shape)}, expected {(m, model['hidden'])}"
        )
    if output.dtype != torch.bfloat16:
        raise RuntimeError(f"invalid output dtype {output.dtype}")
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("output contains non-finite values")
    _validate_cumulative_routes(cumulative_stats, routes, 1, phase="sanity")
    cumulative_stats.zero_()

    flush, actual_flush_bytes = _allocate_flush(args.flush_l2_bytes)
    observation_max_rank_us: list[float] = []
    try:
        for observation in range(1, args.observations + 1):
            samples_us = _time_samples(
                call,
                warmups=args.warmups,
                num_tests=args.num_tests,
                flush=flush,
                barrier_sleep_cycles=args.barrier_sleep_cycles,
            )
            max_rank_us = _emit_observation(
                args=args,
                model_name=model_name,
                model=model,
                m=m,
                requested_cap=requested_cap,
                actual_cap=actual_cap,
                observation=observation,
                samples_us=samples_us,
                routes=routes,
                actual_flush_bytes=actual_flush_bytes,
                rank=rank,
                world_size=world_size,
            )
            if rank == 0:
                assert max_rank_us is not None
                observation_max_rank_us.append(max_rank_us)
            dist.barrier()
    finally:
        del flush

    timed_invocations = args.observations * (args.warmups + args.num_tests)
    _validate_cumulative_routes(
        cumulative_stats, routes, timed_invocations, phase="timed series"
    )

    if rank == 0:
        summary = {
            "backend": BACKEND,
            "kernel_name": KERNEL_NAME,
            "series": args.timing_scope,
            "model": model_name,
            "m": m,
            "requested_cap": requested_cap,
            "actual_cap": actual_cap,
            "world_size": world_size,
            "observations": args.observations,
            "num_tests": args.num_tests,
            "max_rank_median_us": statistics.median(observation_max_rank_us),
            "max_rank_min_us": min(observation_max_rank_us),
            "max_rank_max_us": max(observation_max_rank_us),
            "observation_max_rank_us": observation_max_rank_us,
            "timed_kernel_invocations": timed_invocations,
            "validation": "shape_dtype_finite_and_kernel_route_counts",
            "numerical_reference": False,
            "recipe": list(RECIPE),
            "activation": "swiglu",
            "activation_clamp": args.activation_clamp,
            "fast_math": bool(args.fast_math),
            "masked_ratio": args.masked_ratio,
        }
        print(
            "DEEPGEMM_PR36_SUMMARY_JSON " + json.dumps(summary, sort_keys=True),
            flush=True,
        )
    return None


def _validate_cumulative_routes(
    cumulative_stats, expected_routes, invocations: int, *, phase: str
) -> None:
    import torch

    expected = expected_routes * invocations
    if torch.equal(cumulative_stats, expected):
        return
    mismatches = torch.nonzero(cumulative_stats != expected).flatten()[:8]
    details = [
        {
            "local_expert": int(index),
            "actual": int(cumulative_stats[index].item()),
            "expected": int(expected[index].item()),
        }
        for index in mismatches
    ]
    raise RuntimeError(f"{phase} route-stat mismatch: {details}")


def _run_point(
    args,
    model_name: str,
    m: int,
    rank: int,
    world_size: int,
    group,
) -> None:
    import torch
    import deep_gemm

    model = MODEL_CONFIGS[model_name]
    requested_cap = m if args.match_cap_to_m else args.num_max_tokens_per_rank
    device = torch.device("cuda", torch.cuda.current_device())
    buffer = None
    completed = False
    try:
        x_fp8, topk_ids, topk_weights, l1_weights, l2_weights = _make_case_data(
            args, model, m, rank, world_size, device
        )
        routes = _expected_local_routes(
            topk_ids, model["num_experts"], rank, world_size
        )
        transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
            l1_weights, l2_weights
        )
        del l1_weights, l2_weights

        buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            model["num_experts"],
            requested_cap,
            model["top_k"],
            model["hidden"],
            model["intermediate"],
        )
        actual_cap = int(getattr(buffer, "num_max_tokens_per_rank", requested_cap))
        output = torch.empty((m, model["hidden"]), dtype=torch.bfloat16, device=device)
        cumulative_stats = torch.zeros(
            (model["num_experts"] // world_size,),
            dtype=torch.int32,
            device=device,
        )
        clamp_arg = (
            args.activation_clamp if math.isfinite(args.activation_clamp) else None
        )

        def stage_inputs() -> None:
            buffer.x[:m].copy_(x_fp8[0])
            buffer.x_sf[:m].copy_(x_fp8[1])
            buffer.topk_idx[:m].copy_(topk_ids)
            buffer.topk_weights[:m].copy_(topk_weights)

        def kernel_call():
            deep_gemm.fp8_mega_moe(
                output,
                transformed_l1,
                transformed_l2,
                buffer,
                cumulative_local_expert_recv_stats=cumulative_stats,
                recipe=RECIPE,
                activation="swiglu",
                activation_clamp=clamp_arg,
                fast_math=bool(args.fast_math),
            )
            return output

        def forward_call():
            stage_inputs()
            return kernel_call()

        if args.timing_scope == "kernel":
            stage_inputs()
            timed_call = kernel_call
        else:
            timed_call = forward_call

        _run_series(
            timed_call,
            args=args,
            model_name=model_name,
            model=model,
            m=m,
            requested_cap=requested_cap,
            actual_cap=actual_cap,
            routes=routes,
            cumulative_stats=cumulative_stats,
            rank=rank,
            world_size=world_size,
        )
        completed = True
    except Exception as exc:
        point = {
            "backend": BACKEND,
            "kernel_name": KERNEL_NAME,
            "model": model_name,
            "m": m,
            "requested_cap": requested_cap,
            "rank": rank,
            "status": "fatal",
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(
            "DEEPGEMM_PR36_POINT_JSON " + json.dumps(point, sort_keys=True),
            flush=True,
        )
        raise
    finally:
        # On failure, let process teardown reclaim symmetric memory. Destroying a
        # buffer while peer ranks are still inside the kernel is unsafe.
        if completed and buffer is not None:
            with contextlib.suppress(Exception):
                buffer.destroy()
            gc.collect()
            torch.cuda.empty_cache()


def main() -> int:
    args = _parse_args()

    try:
        import torch
        import torch.distributed as dist
        import deep_gemm
    except ImportError as exc:
        raise SystemExit(
            "DeepGEMM PR36 runtime is unavailable. Build the PR with "
            "`bash build_sgl_deep_gemm.sh` and install the resulting wheel. "
            f"Original import error: {exc}"
        ) from exc

    required_symbols = (
        "fp8_mega_moe",
        "get_symm_buffer_for_mega_moe",
        "transform_weights_for_mega_moe_sm90",
    )
    missing = [name for name in required_symbols if not hasattr(deep_gemm, name)]
    if missing:
        raise SystemExit(
            "installed deep_gemm does not contain PR36 SM90 MegaMoE symbols: "
            + ", ".join(missing)
        )
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise SystemExit("launch this multi-rank benchmark with torchrun")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    group = dist.group.WORLD

    try:
        major, minor = torch.cuda.get_device_capability()
        if major != 9:
            raise RuntimeError(f"SM90 required, got SM{major}{minor}")
        if world_size != args.expected_world_size:
            raise RuntimeError(
                f"expected world_size={args.expected_world_size}, got {world_size}"
            )
        for name in args.model_config:
            model = MODEL_CONFIGS[name]
            if model["num_experts"] % world_size:
                raise RuntimeError(
                    f"{name} experts={model['num_experts']} must divide "
                    f"world_size={world_size}"
                )
            if model["hidden"] % 128 or model["intermediate"] % 128:
                raise RuntimeError(f"{name} hidden dimensions must align to 128")
            if model["intermediate"] > 4096:
                raise RuntimeError(
                    f"{name} intermediate={model['intermediate']} exceeds PR36 limit 4096"
                )

        if rank == 0:
            runtime_identity = _runtime_identity(deep_gemm)
            plan = {
                "backend": BACKEND,
                "kernel_name": KERNEL_NAME,
                "target_pr36_head": PR36_HEAD,
                "runtime_identity": runtime_identity,
                "models": args.model_config,
                "batches": args.batches,
                "world_size": world_size,
                "num_max_tokens_per_rank": args.num_max_tokens_per_rank,
                "match_cap_to_m": args.match_cap_to_m,
                "series": args.timing_scope,
                "seed": args.seed,
                "observations": args.observations,
                "warmups": args.warmups,
                "num_tests": args.num_tests,
                "flush_l2_bytes": args.flush_l2_bytes,
                "recipe": list(RECIPE),
                "activation": "swiglu",
                "activation_clamp": args.activation_clamp,
                "fast_math": bool(args.fast_math),
                "masked_ratio": args.masked_ratio,
            }
            print(
                "DEEPGEMM_PR36_PLAN_JSON " + json.dumps(plan, sort_keys=True),
                flush=True,
            )

        for model_name in args.model_config:
            for m in args.batches:
                if rank == 0:
                    print(
                        f"# DeepGEMM PR36 model={model_name} M={m} "
                        f"world={world_size} scope={args.timing_scope}",
                        flush=True,
                    )
                _run_point(args, model_name, m, rank, world_size, group)
        return 0
    finally:
        with contextlib.suppress(Exception):
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
