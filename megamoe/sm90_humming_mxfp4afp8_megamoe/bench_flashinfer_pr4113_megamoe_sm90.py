#!/usr/bin/env python3
"""PR383-style workload benchmark for FlashInfer PR #4113 SM90 MegaMoE.

This benchmark keeps the DeepGEMM PR #383 model/workload contract:

* EP8 by default, with ``M`` interpreted as tokens per rank;
* Flash: H=4096, I=2048, E=256, top-k=6;
* Pro: H=7168, I=3072, E=384, top-k=6;
* M sweep: 8,16,32,64,128,256,512,1024,2048,4096,8192;
* cold-L2 samples, slowest-rank aggregation, 50 observations for M<=128
  and 3 observations for larger M, with 20 samples per observation.

The implementation is FlashInfer PR #4113's public ``sm90_pull_fp8`` backend,
not DeepGEMM's workspace ABI.  This compute-only benchmark stages inputs once
outside timing and measures ``MegaKernelBackend.compute(output=None)``:
the fused dispatch + FC1 + SwiGLU + FC2 launch plus the default TopkReduce.

CUDA events time the complete FlashInfer call.  This is deliberately not
presented as identical to PR383's Kineto sum of two named DeepGEMM kernels:
PR #4113 has a different launch topology and includes TopkReduce in the
compute call.

Requirements:

* FlashInfer at PR #4113 head ``28483960`` or a descendant containing the
  ``sm90_pull_fp8`` backend;
* Hopper SM90 GPUs, CuTeDSL, NVSHMEM, and working multi-rank peer access;
* launch with torchrun, for example:

  ``torchrun --standalone --nproc_per_node=8 bench_flashinfer_pr4113_megamoe_sm90.py``

Useful shorter runs:

  ``torchrun --standalone --nproc_per_node=8 bench_flashinfer_pr4113_megamoe_sm90.py --model-config flash --batches 8 16 32 --observations 3``
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import statistics
import sys
from dataclasses import dataclass
from typing import Callable

# PR #4113's own benchmark uses these defaults for multirank Hopper.
os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
os.environ.setdefault("NVSHMEM_DISABLE_NVLS", "1")


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
DEFAULT_TILES = {
    "non_swap_ab": (64, 128, 128),
    "swap_ab": (256, 32, 128),
}
E4M3_MAX = 448.0
FC1_ACT_SCALE = 8.0 / (0.95 * E4M3_MAX)
FC2_ACT_SCALE = 8.0 / (0.95 * E4M3_MAX)


@dataclass
class SeriesResult:
    status: str
    observation_max_rank_us: list[float]
    error: str = ""


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
        help="fixed workspace capacity; PR383 defaults to 8192",
    )
    parser.add_argument(
        "--match-cap-to-m",
        action="store_true",
        help="allocate each point with cap=M instead of PR383's fixed cap",
    )
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--observations",
        type=int,
        default=None,
        help="override PR383's 50 observations for M<=128 and 3 otherwise",
    )
    parser.add_argument("--small-observations", type=int, default=50)
    parser.add_argument("--large-observations", type=int, default=3)
    parser.add_argument("--small-m-threshold", type=int, default=128)
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
        "--fp8-scale-mode",
        choices=("blockwise", "per_tensor"),
        default="blockwise",
        help="blockwise is the closest match to PR383's FP32 block scales",
    )
    parser.add_argument(
        "--operand-order",
        choices=("swap_ab", "non_swap_ab"),
        default="swap_ab",
    )
    parser.add_argument(
        "--mma-tiler",
        type=str,
        default=None,
        metavar="M,N,K",
        help="override PR4113's layout-specific default tile",
    )
    parser.add_argument("--kind", choices=("fp8_e4m3", "fp8_e5m2"), default="fp8_e4m3")
    parser.add_argument("--fp8-accum-mode", choices=("1xacc", "2xacc"), default="1xacc")
    parser.add_argument(
        "--load-balance-mode",
        choices=("static", "atomic_counter"),
        default="atomic_counter",
    )
    parser.add_argument(
        "--token-back",
        choices=("reuse_dispatch_warps", "epi_warps"),
        default="reuse_dispatch_warps",
        help="PR4113 perf default; use epi_warps for its correctness-tested path",
    )
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--fast-math", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="emit a failed point and continue instead of stopping",
    )
    args = parser.parse_args()

    if not args.batches or min(args.batches) <= 0:
        parser.error("--batches must contain positive values")
    if not args.match_cap_to_m and args.num_max_tokens_per_rank < max(args.batches):
        parser.error("--num-max-tokens-per-rank must cover the largest batch")
    if args.expected_world_size <= 0:
        parser.error("--expected-world-size must be positive")
    if args.observations is not None and args.observations <= 0:
        parser.error("--observations must be positive")
    if min(args.small_observations, args.large_observations, args.num_tests) <= 0:
        parser.error("observation and test counts must be positive")
    if args.warmups < 0 or args.flush_l2_bytes < 0:
        parser.error("warmups and flush size must be non-negative")
    return args


def _stable_seed(name: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(name)) & 0x7FFFFFFF


def _case_seed(base_seed: int, model: str, m: int, rank: int) -> int:
    return base_seed + rank * 1_000_003 + _stable_seed(f"{model}:{m}")


def _tile(args: argparse.Namespace) -> tuple[int, int, int]:
    if args.mma_tiler is None:
        return DEFAULT_TILES[args.operand_order]
    values = tuple(int(value) for value in args.mma_tiler.split(","))
    if len(values) != 3 or min(values) <= 0:
        raise ValueError("--mma-tiler must be three positive integers M,N,K")
    return values


def _make_config(args: argparse.Namespace, model: dict):
    from flashinfer.moe_ep import Sm90PullFp8MegaMoeConfig

    return Sm90PullFp8MegaMoeConfig(
        intermediate_size=model["intermediate"],
        top_k=model["top_k"],
        kind=args.kind,
        fp8_scale_mode=args.fp8_scale_mode,
        fp8_accum_mode=args.fp8_accum_mode,
        swap_ab=args.operand_order == "swap_ab",
        mma_tiler_mnk=_tile(args),
        load_balance_mode=args.load_balance_mode,
        gate_up_clamp=args.activation_clamp,
        fast_math=bool(args.fast_math),
        in_kernel_fc2_reduce=False,
        token_back_by_dispatch=args.token_back == "reuse_dispatch_warps",
        fc1_activation_dequant_scale=FC1_ACT_SCALE,
        fc2_activation_dequant_scale=FC2_ACT_SCALE,
    )


def _make_inputs(
    args: argparse.Namespace,
    model_name: str,
    model: dict,
    m: int,
    rank: int,
    device,
):
    import torch

    seed = _case_seed(args.seed, model_name, m, rank)
    generator = torch.Generator(device=device).manual_seed(seed)
    hidden_states = torch.randn(
        (m, model["hidden"]),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
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
    return hidden_states, topk_ids.to(torch.int64), topk_weights


def _make_weights(
    args: argparse.Namespace,
    model_name: str,
    model: dict,
    m: int,
    rank: int,
    world_size: int,
    device,
):
    import torch

    from flashinfer.moe_ep import preprocess_sm90_pull_fp8_mega_weights
    from flashinfer.moe_ep.weights import MoEWeightPack

    local_experts = model["num_experts"] // world_size
    seed = _case_seed(args.seed, model_name, m, rank)
    generator = torch.Generator(device=device).manual_seed(seed + 17)
    if args.fp8_scale_mode == "blockwise":
        w13_scale = 0.05
        w2_scale = 0.05
    else:
        w13_scale = model["hidden"] ** -0.5
        w2_scale = model["intermediate"] ** -0.5

    w13 = (
        torch.randn(
            (local_experts, 2 * model["intermediate"], model["hidden"]),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * w13_scale
    )
    w2 = (
        torch.randn(
            (local_experts, model["hidden"], model["intermediate"]),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * w2_scale
    )
    transformed = preprocess_sm90_pull_fp8_mega_weights(
        MoEWeightPack(w13=w13, w2=w2),
        intermediate_size=model["intermediate"],
        hidden_size=model["hidden"],
        kind=args.kind,
        fp8_scale_mode=args.fp8_scale_mode,
        fc1_activation_dequant_scale=FC1_ACT_SCALE,
        fc2_activation_dequant_scale=FC2_ACT_SCALE,
    )
    del w13, w2
    return transformed


def _expected_local_routes(topk_ids, num_experts: int, rank: int, world_size: int):
    import torch
    import torch.distributed as dist

    counts = torch.bincount(topk_ids.reshape(-1), minlength=num_experts)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    local_experts = num_experts // world_size
    return counts[rank * local_experts : (rank + 1) * local_experts].to(torch.int)


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
    args: argparse.Namespace,
    model_name: str,
    model: dict,
    m: int,
    cap: int,
    series: str,
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
        "backend": "flashinfer_pr4113_sm90_pull_fp8",
        "series": series,
        "model": model_name,
        "m": m,
        "cap": cap,
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
        "route_total": sum(route_list),
        "touched_experts": sum(value > 0 for value in route_list),
        "fp8_scale_mode": args.fp8_scale_mode,
        "operand_order": args.operand_order,
        "mma_tiler_mnk": list(_tile(args)),
        "requested_flush_l2_bytes": args.flush_l2_bytes,
        "actual_flush_l2_bytes": actual_flush_bytes,
    }
    print(
        "FLASHINFER_4113_STAT_JSON " + json.dumps(local_row, sort_keys=True),
        flush=True,
    )

    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_row)
    if rank != 0:
        return None

    max_row = max(gathered, key=lambda row: row["returned_us"])
    aggregate = {
        "backend": local_row["backend"],
        "series": series,
        "model": model_name,
        "m": m,
        "cap": cap,
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
                "route_total": row["route_total"],
                "touched_experts": row["touched_experts"],
            }
            for row in sorted(gathered, key=lambda row: row["rank"])
        ],
        "fp8_scale_mode": args.fp8_scale_mode,
        "operand_order": args.operand_order,
        "mma_tiler_mnk": list(_tile(args)),
        "requested_flush_l2_bytes": args.flush_l2_bytes,
        "actual_flush_l2_bytes_min": min(
            row["actual_flush_l2_bytes"] for row in gathered
        ),
    }
    print(
        "FLASHINFER_4113_OBSERVATION_JSON " + json.dumps(aggregate, sort_keys=True),
        flush=True,
    )
    return aggregate["max_rank_us"]


def _run_series(
    call: Callable,
    *,
    args: argparse.Namespace,
    model_name: str,
    model: dict,
    m: int,
    cap: int,
    series: str,
    routes,
    observations: int,
    rank: int,
    world_size: int,
) -> SeriesResult:
    import torch
    import torch.distributed as dist

    output = call()
    torch.cuda.synchronize()
    dist.barrier()
    if tuple(output.shape) != (m, model["hidden"]):
        raise RuntimeError(
            f"invalid {series} output shape {tuple(output.shape)}, "
            f"expected {(m, model['hidden'])}"
        )
    if output.dtype != torch.bfloat16:
        raise RuntimeError(f"invalid {series} output dtype {output.dtype}")
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError(f"{series} output contains non-finite values")

    flush, actual_flush_bytes = _allocate_flush(args.flush_l2_bytes)
    observation_max_rank_us: list[float] = []
    try:
        for observation in range(1, observations + 1):
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
                cap=cap,
                series=series,
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

    if rank == 0:
        summary = {
            "backend": "flashinfer_pr4113_sm90_pull_fp8",
            "series": series,
            "model": model_name,
            "m": m,
            "cap": cap,
            "world_size": world_size,
            "observations": observations,
            "num_tests": args.num_tests,
            "max_rank_median_us": statistics.median(observation_max_rank_us),
            "max_rank_min_us": min(observation_max_rank_us),
            "max_rank_max_us": max(observation_max_rank_us),
            "observation_max_rank_us": observation_max_rank_us,
            "fp8_scale_mode": args.fp8_scale_mode,
            "operand_order": args.operand_order,
            "mma_tiler_mnk": list(_tile(args)),
        }
        print(
            "FLASHINFER_4113_SUMMARY_JSON " + json.dumps(summary, sort_keys=True),
            flush=True,
        )
    return SeriesResult("pass", observation_max_rank_us)


def _is_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    message = str(exc).lower()
    return "out of memory" in message or "oom" in message or "nvshmem_malloc" in message


def _run_point(
    args: argparse.Namespace,
    model_name: str,
    m: int,
    rank: int,
    world_size: int,
    runtime_bootstrap,
) -> bool:
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import FleetParams, MoEEpTensors
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    model = MODEL_CONFIGS[model_name]
    cap = m if args.match_cap_to_m else args.num_max_tokens_per_rank
    observations = (
        args.observations
        if args.observations is not None
        else (
            args.small_observations
            if m <= args.small_m_threshold
            else args.large_observations
        )
    )
    device = torch.device("cuda", torch.cuda.current_device())
    backend = None
    workspace = None
    status = "pass"
    error = ""
    try:
        hidden_states, topk_ids, topk_weights = _make_inputs(
            args, model_name, model, m, rank, device
        )
        routes = _expected_local_routes(
            topk_ids, model["num_experts"], rank, world_size
        )
        transformed = _make_weights(
            args, model_name, model, m, rank, world_size, device
        )
        kernel_config = _make_config(args, model)
        fleet_params = FleetParams(
            num_experts=model["num_experts"],
            max_tokens_per_rank=cap,
            token_hidden_size=model["hidden"],
        )
        tensors = MoEEpTensors(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

        backend = create_mega_kernel(kernel_config)
        backend.bind_ep_bootstrap(runtime_bootstrap)
        workspace = backend.prepare_workspace(runtime_bootstrap, fleet_params)
        backend.stage_inputs(tensors, workspace, quantize_input=True)

        def compute_call():
            return backend.compute(workspace, transformed, output=None)

        _run_series(
            compute_call,
            args=args,
            model_name=model_name,
            model=model,
            m=m,
            cap=cap,
            series="compute",
            routes=routes,
            observations=observations,
            rank=rank,
            world_size=world_size,
        )
    except Exception as exc:  # noqa: BLE001 - collective point reporting below
        status = "skip_oom" if _is_oom(exc) else "failed"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if backend is not None and workspace is not None:
            with contextlib.suppress(Exception):
                backend.destroy(workspace)
        gc.collect()
        torch.cuda.empty_cache()

    all_status = [None] * world_size
    dist.all_gather_object(all_status, (status, error))
    dist.barrier()
    passed = all(item[0] == "pass" for item in all_status)
    if rank == 0 and not passed:
        point = {
            "backend": "flashinfer_pr4113_sm90_pull_fp8",
            "model": model_name,
            "m": m,
            "cap": cap,
            "status": "skip_oom"
            if any(item[0] == "skip_oom" for item in all_status)
            else "failed",
            "errors": [
                {"rank": idx, "error": item[1]}
                for idx, item in enumerate(all_status)
                if item[1]
            ],
        }
        print(
            "FLASHINFER_4113_POINT_JSON " + json.dumps(point, sort_keys=True),
            flush=True,
        )
    return passed


def main() -> int:
    args = _parse_args()

    try:
        import torch
        import torch.distributed as dist
        import flashinfer
        from flashinfer.moe_ep import (
            BootstrapConfig,
            bootstrap_moe_ep_runtime,
            ensure_moe_ep_cuda_device,
            finalize_moe_ep_runtime,
        )
        from flashinfer.moe_ep.core.runtime import sm90_pull_fp8_runtime_requirements
    except ImportError as exc:
        raise SystemExit(
            "FlashInfer PR4113 runtime is unavailable. Install a build containing "
            "the sm90_pull_fp8 backend, CuTeDSL, and NVSHMEM. "
            f"Original import error: {exc}"
        ) from exc

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise SystemExit(
            "multi-rank PR4113 benchmark must be launched with torchrun; "
            "see this file's module docstring"
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    runtime = None
    completed = False
    try:
        major, minor = torch.cuda.get_device_capability()
        if major != 9:
            raise RuntimeError(f"SM90 required, got SM{major}{minor}")
        if world_size != args.expected_world_size:
            raise RuntimeError(
                f"expected world_size={args.expected_world_size}, got {world_size}"
            )
        for name in args.model_config:
            if MODEL_CONFIGS[name]["num_experts"] % world_size != 0:
                raise RuntimeError(
                    f"{name} experts={MODEL_CONFIGS[name]['num_experts']} "
                    f"must divide world_size={world_size}"
                )

        bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
        ensure_moe_ep_cuda_device(bootstrap)
        runtime = bootstrap_moe_ep_runtime(
            bootstrap, sm90_pull_fp8_runtime_requirements(bootstrap)
        )
        session_bootstrap = BootstrapConfig(
            world_size=world_size,
            rank=rank,
            auto_bootstrap=False,
        )

        if rank == 0:
            plan = {
                "backend": "flashinfer_pr4113_sm90_pull_fp8",
                "flashinfer_version": getattr(flashinfer, "__version__", "unknown"),
                "pr4113_head": "28483960d7a56dd6a77e735f2c874b8e4dbd9d44",
                "models": args.model_config,
                "batches": args.batches,
                "world_size": world_size,
                "num_max_tokens_per_rank": args.num_max_tokens_per_rank,
                "match_cap_to_m": args.match_cap_to_m,
                "series": "compute",
                "seed": args.seed,
                "observations_override": args.observations,
                "small_observations": args.small_observations,
                "large_observations": args.large_observations,
                "warmups": args.warmups,
                "num_tests": args.num_tests,
                "flush_l2_bytes": args.flush_l2_bytes,
                "fp8_scale_mode": args.fp8_scale_mode,
                "operand_order": args.operand_order,
                "mma_tiler_mnk": list(_tile(args)),
                "load_balance_mode": args.load_balance_mode,
                "token_back": args.token_back,
                "fast_math": bool(args.fast_math),
            }
            print(
                "FLASHINFER_4113_PLAN_JSON " + json.dumps(plan, sort_keys=True),
                flush=True,
            )

        for model_name in args.model_config:
            for m in args.batches:
                if rank == 0:
                    print(
                        f"# FlashInfer PR4113 model={model_name} M={m} "
                        f"world={world_size}",
                        flush=True,
                    )
                passed = _run_point(
                    args,
                    model_name,
                    m,
                    rank,
                    world_size,
                    session_bootstrap,
                )
                if not passed and not args.continue_on_error:
                    raise RuntimeError(f"failed point model={model_name} M={m}")
        completed = True
    finally:
        if runtime is not None:
            finalize_moe_ep_runtime(runtime)
        if dist.is_initialized():
            if completed:
                dist.barrier()
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
