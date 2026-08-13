#!/usr/bin/env python3
"""PR383-style workload benchmark for FlashInfer PR #4069 SM90 MegaMoE.

This benchmark intentionally keeps the same model/workload contract used by the
local PR383-aligned FlashInfer PR4113 study, but swaps the implementation to
FlashInfer's merged ``sm90_push_fp8`` Hopper backend from PR #4069:

* EP8 by default, with ``M`` interpreted as tokens per rank;
* Flash: H=4096, I=2048, E=256, top-k=6;
* Pro: H=7168, I=3072, E=384, top-k=6;
* M sweep: 8,16,32,64,128,256,512,1024,2048,4096,8192;
* workspace capacity ``cap=M`` at every point by default;
* seed 101, 3 observations, 5 warmups, 20 cold-L2 samples per observation,
  and slowest-rank aggregation.

Unlike the upstream PR4069 benchmark, this file does not use the PR's SMALL or
DSV3 canned shapes.  It is designed to produce numbers that are directly
comparable to the local PR383-style notes and tables.

Timing boundary:

* timed region: ``backend.stage_inputs(...)`` followed by
  ``backend.compute(workspace, transformed, output=output)``;
* outside timing: static weight preprocessing, workspace allocation, JIT warmup,
  and output allocation.

PR4069 starts token dispatch inside ``stage_inputs``, so both calls must be in
the timed region to measure the complete push-style MoE round and avoid omitting
dispatch latency.

The benchmark defaults to PR4069's optimized path:

* ``dedup_dispatch=True``
* ``grouped_combine=True``
* ``fuse_fc1_epilogue=True``

Use ``--no-dedup-dispatch``, ``--no-grouped-combine``, or
``--no-fuse-fc1-epilogue`` for ablations.

Requirements:

* FlashInfer build that contains the merged ``sm90_push_fp8`` backend from
  PR #4069 (merged on 2026-08-12);
* Hopper SM90 GPUs on a single-node NVLink topology;
* launch with torchrun, for example:

  ``torchrun --standalone --nproc_per_node=8 bench_flashinfer_pr4069_megamoe_sm90.py``
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
        help="fixed workspace capacity used only with --no-match-cap-to-m",
    )
    parser.add_argument(
        "--match-cap-to-m",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="allocate each point with cap=M, matching the PR383 workload",
    )
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--observations",
        type=int,
        default=3,
        help="independent observations per point; PR383 workload uses 3",
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--num-tests", type=int, default=20)
    parser.add_argument("--flush-l2-bytes", type=int, default=8_000_000_000)
    parser.add_argument(
        "--barrier-sleep-cycles",
        type=int,
        default=20_000_000,
        help="GPU sleep before each rank barrier, matching bench_kineto discipline",
    )
    parser.add_argument("--payload-dtype", choices=("fp8", "bf16"), default="fp8")
    parser.add_argument("--combine-dtype", choices=("fp8", "bf16"), default="fp8")
    parser.add_argument(
        "--capacity-factor",
        type=float,
        default=1.0,
        help="workspace headroom bound used by the sm90_push_fp8 protocol",
    )
    parser.add_argument(
        "--dedup-dispatch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="store one payload per token and destination rank",
    )
    parser.add_argument(
        "--grouped-combine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="owner-side grouped FP8 combine; requires --combine-dtype fp8",
    )
    parser.add_argument(
        "--fuse-fc1-epilogue",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable PR4069's fused SwiGLU + activation-quantization FC1 epilogue",
    )
    parser.add_argument(
        "--allow-unverified-p2p",
        action="store_true",
        help="forward allow_unverified_p2p to the backend",
    )
    parser.add_argument(
        "--init-timeout-s",
        type=float,
        default=600.0,
        help="process-group timeout used during push-backend runner construction",
    )
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
    if args.observations <= 0:
        parser.error("--observations must be positive")
    if args.num_tests <= 0:
        parser.error("--num-tests must be positive")
    if args.warmups < 0 or args.flush_l2_bytes < 0:
        parser.error("warmups and flush size must be non-negative")
    if not (0.0 < args.capacity_factor <= 1.0):
        parser.error("--capacity-factor must be in (0, 1]")
    if args.grouped_combine and args.combine_dtype != "fp8":
        parser.error("--grouped-combine requires --combine-dtype fp8")
    if args.init_timeout_s <= 0.0:
        parser.error("--init-timeout-s must be positive")
    return args


def _make_config(args: argparse.Namespace, model: dict):
    from flashinfer.moe_ep import Sm90PushFp8MegaMoeConfig

    return Sm90PushFp8MegaMoeConfig(
        intermediate_size=model["intermediate"],
        top_k=model["top_k"],
        capacity_factor=args.capacity_factor,
        dedup_dispatch=args.dedup_dispatch,
        grouped_combine=args.grouped_combine,
        fuse_fc1_epilogue=args.fuse_fc1_epilogue,
        payload_dtype=args.payload_dtype,
        combine_dtype=args.combine_dtype,
        allow_unverified_p2p=args.allow_unverified_p2p,
        init_timeout_s=args.init_timeout_s,
    )


def _make_case_data(
    args: argparse.Namespace,
    model: dict,
    m: int,
    rank: int,
    world_size: int,
    device,
):
    import torch

    from flashinfer.moe_ep.weights import MoEWeightPack

    # Reset RNG per point because this driver keeps one long-lived process while
    # the original PR383 discipline reseeds each point independently.
    generator = torch.Generator(device=device).manual_seed(args.seed + rank)
    hidden_states = torch.randn(
        (m, model["hidden"]),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    local_experts = model["num_experts"] // world_size
    w13 = (
        torch.randn(
            (local_experts, 2 * model["intermediate"], model["hidden"]),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * 0.05
    )
    w2 = (
        torch.randn(
            (local_experts, model["hidden"], model["intermediate"]),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * 0.05
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
    weights = MoEWeightPack(w13=w13.contiguous(), w2=w2.contiguous())
    output = torch.empty((m, model["hidden"]), dtype=torch.bfloat16, device=device)
    return (
        hidden_states,
        topk_ids.to(torch.int32).contiguous(),
        topk_weights.contiguous(),
        weights,
        output,
    )


def _expected_local_routes(topk_ids, num_experts: int, rank: int, world_size: int):
    import torch
    import torch.distributed as dist

    counts = torch.bincount(topk_ids.reshape(-1).to(torch.int64), minlength=num_experts)
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
    stage: Callable[[], None],
    compute: Callable[[], object],
    *,
    warmups: int,
    num_tests: int,
    flush,
    barrier_sleep_cycles: int,
) -> list[float]:
    import torch
    import torch.distributed as dist

    for _ in range(warmups):
        stage()
        compute()
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
        stage()
        compute()
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
        "backend": "flashinfer_pr4069_sm90_push_fp8",
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
        "payload_dtype": args.payload_dtype,
        "combine_dtype": args.combine_dtype,
        "capacity_factor": args.capacity_factor,
        "dedup_dispatch": args.dedup_dispatch,
        "grouped_combine": args.grouped_combine,
        "fuse_fc1_epilogue": args.fuse_fc1_epilogue,
        "requested_flush_l2_bytes": args.flush_l2_bytes,
        "actual_flush_l2_bytes": actual_flush_bytes,
    }
    print(
        "FLASHINFER_4069_STAT_JSON " + json.dumps(local_row, sort_keys=True),
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
        "payload_dtype": args.payload_dtype,
        "combine_dtype": args.combine_dtype,
        "capacity_factor": args.capacity_factor,
        "dedup_dispatch": args.dedup_dispatch,
        "grouped_combine": args.grouped_combine,
        "fuse_fc1_epilogue": args.fuse_fc1_epilogue,
        "requested_flush_l2_bytes": args.flush_l2_bytes,
        "actual_flush_l2_bytes_min": min(
            row["actual_flush_l2_bytes"] for row in gathered
        ),
    }
    print(
        "FLASHINFER_4069_OBSERVATION_JSON " + json.dumps(aggregate, sort_keys=True),
        flush=True,
    )
    return aggregate["max_rank_us"]


def _run_series(
    stage: Callable[[], None],
    compute: Callable[[], object],
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

    stage()
    output = compute()
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
                stage,
                compute,
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
            "backend": "flashinfer_pr4069_sm90_push_fp8",
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
            "payload_dtype": args.payload_dtype,
            "combine_dtype": args.combine_dtype,
            "capacity_factor": args.capacity_factor,
            "dedup_dispatch": args.dedup_dispatch,
            "grouped_combine": args.grouped_combine,
            "fuse_fc1_epilogue": args.fuse_fc1_epilogue,
        }
        print(
            "FLASHINFER_4069_SUMMARY_JSON " + json.dumps(summary, sort_keys=True),
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
    bootstrap,
) -> bool:
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import FleetParams, MoEEpTensors
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    model = MODEL_CONFIGS[model_name]
    cap = m if args.match_cap_to_m else args.num_max_tokens_per_rank
    device = torch.device("cuda", torch.cuda.current_device())
    backend = None
    workspace = None
    status = "pass"
    error = ""
    try:
        hidden_states, topk_ids, topk_weights, weights, output = _make_case_data(
            args, model, m, rank, world_size, device
        )
        routes = _expected_local_routes(
            topk_ids, model["num_experts"], rank, world_size
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
        backend.bind_ep_bootstrap(bootstrap)
        backend.validate_init(bootstrap, fleet_params)
        transformed = backend.preprocess_weights(weights, fleet_params)
        workspace = backend.prepare_workspace(bootstrap, fleet_params)

        def stage_call():
            backend.stage_inputs(tensors, workspace, quantize_input=True)

        def compute_call():
            return backend.compute(workspace, transformed, output=output)

        _run_series(
            stage_call,
            compute_call,
            args=args,
            model_name=model_name,
            model=model,
            m=m,
            cap=cap,
            series="forward",
            routes=routes,
            observations=args.observations,
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
            "backend": "flashinfer_pr4069_sm90_push_fp8",
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
            "FLASHINFER_4069_POINT_JSON " + json.dumps(point, sort_keys=True),
            flush=True,
        )
    return passed


def main() -> int:
    args = _parse_args()

    try:
        import flashinfer
        import torch
        import torch.distributed as dist
        from flashinfer.moe_ep import BootstrapConfig
    except ImportError as exc:
        raise SystemExit(
            "FlashInfer PR4069 runtime is unavailable. Install a build containing "
            "the merged sm90_push_fp8 backend. "
            f"Original import error: {exc}"
        ) from exc

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise SystemExit(
            "multi-rank PR4069 benchmark must be launched with torchrun; "
            "see this file's module docstring"
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

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

        bootstrap = BootstrapConfig(
            world_size=world_size,
            rank=rank,
            process_group=dist.group.WORLD,
            device=torch.cuda.current_device(),
        )

        if rank == 0:
            plan = {
                "backend": "flashinfer_pr4069_sm90_push_fp8",
                "kernel_name": "sm90_push_fp8",
                "flashinfer_version": getattr(flashinfer, "__version__", "unknown"),
                "pr4069_merged_date": "2026-08-12",
                "models": args.model_config,
                "batches": args.batches,
                "world_size": world_size,
                "num_max_tokens_per_rank": args.num_max_tokens_per_rank,
                "match_cap_to_m": args.match_cap_to_m,
                "series": "forward",
                "seed": args.seed,
                "observations": args.observations,
                "warmups": args.warmups,
                "num_tests": args.num_tests,
                "flush_l2_bytes": args.flush_l2_bytes,
                "payload_dtype": args.payload_dtype,
                "combine_dtype": args.combine_dtype,
                "capacity_factor": args.capacity_factor,
                "dedup_dispatch": args.dedup_dispatch,
                "grouped_combine": args.grouped_combine,
                "fuse_fc1_epilogue": args.fuse_fc1_epilogue,
                "allow_unverified_p2p": args.allow_unverified_p2p,
                "init_timeout_s": args.init_timeout_s,
            }
            print(
                "FLASHINFER_4069_PLAN_JSON " + json.dumps(plan, sort_keys=True),
                flush=True,
            )

        for model_name in args.model_config:
            for m in args.batches:
                if rank == 0:
                    print(
                        f"# FlashInfer PR4069 model={model_name} M={m} "
                        f"world={world_size}",
                        flush=True,
                    )
                passed = _run_point(
                    args,
                    model_name,
                    m,
                    rank,
                    world_size,
                    bootstrap,
                )
                if not passed and not args.continue_on_error:
                    raise RuntimeError(f"failed point model={model_name} M={m}")
        completed = True
    finally:
        if dist.is_initialized():
            if completed:
                dist.barrier()
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
