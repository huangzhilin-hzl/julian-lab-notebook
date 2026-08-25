#!/usr/bin/env python3
"""Auditable SM90 Gluon MegaMoE benchmark for the DSV4 Pro EP8 workload.

The default workload reproduces the Gluon run documented at Triton-distributed
commit ``ed0be56e967474c289ab0b48097f9f33c030b994``:

* EP8, with ``M`` interpreted as tokens per rank;
* Pro: H=7168, I=3072, E=384, top-k=6;
* M sweep: 1,2,4,8,16,32,64 with a fixed 64-token/rank capacity;
* random-score top-k routes, 48 experts per scheduler wave;
* 30 warmups and 100 active cold-L2 CUDA Event samples;
* per-iteration maximum latency across all ranks, followed by the median of
  those critical-path samples.

The benchmark loads the vendored ``mega_moe_gluon.py`` implementation and
``mega_moe_gluon_reference.py`` helper from its own directory.  FP8 inputs,
blockwise-FP8 weights, weight transformation, symmetric-memory allocation, and
JIT compilation are outside the timed interval.  The default ``e2e`` scope
measures Gluon pre-dispatch plus the persistent fused kernel, from
already-quantized FP8 source inputs and routes to token-major BF16 output.
``kernel`` measures the persistent kernel with fixed registered inputs;
``both`` records both boundaries.

Every rank emits ``GLUON_STAT_JSON`` with its raw samples.  Rank zero also emits
``GLUON_OBSERVATION_JSON`` and ``GLUON_SUMMARY_JSON`` records compatible with
the JSONL-oriented evaluation style used by the neighboring MegaMoE scripts.

Example::

    env -u PYTHONPATH torchrun --standalone --nproc-per-node=8 \
      bench_gluon_megamoe_sm90.py
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Callable


SOURCE_SNAPSHOT = "Triton-distributed@ed0be56e967474c289ab0b48097f9f33c030b994"
SOURCE_SHA256 = {
    "mega_moe_gluon.py": "6ade1ba123ff4c25ba5b3d2f1118963124546fa0451e103b9f85788e7b7ff5a8",
    "mega_moe_gluon_reference.py": "8dc9264fde9f40d94abd7e39040bfb6d0c23caf4d8b1231b13c6955aeff77153",
}
BACKEND = "triton_dist_gluon_sm90_fp8_megamoe"
KERNEL_NAME = "run_sm90_fused_dispatch_1d2d_compact_symmetric"
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
DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32, 64)
BLOCK_M = 64
GROUP_N = 128
GROUP_K = 128


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model-config",
        nargs="+",
        choices=tuple(MODEL_CONFIGS),
        default=["pro"],
    )
    parser.add_argument("--batches", nargs="+", type=int, default=list(DEFAULT_BATCHES))
    parser.add_argument(
        "--num-max-tokens-per-rank",
        type=int,
        default=64,
        help="fixed symmetric-buffer capacity used unless --match-cap-to-m is set",
    )
    parser.add_argument(
        "--match-cap-to-m",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="allocate each point with cap=M instead of the documented fixed cap=64",
    )
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="offset added to the documented per-input RNG seeds",
    )
    parser.add_argument("--observations", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=30)
    parser.add_argument("--num-tests", type=int, default=100)
    parser.add_argument("--flush-l2-bytes", type=int, default=8_000_000_000)
    parser.add_argument(
        "--barrier-sleep-cycles",
        type=int,
        default=0,
        help="optional GPU sleep before each measured cross-rank barrier",
    )
    parser.add_argument(
        "--timing-scope",
        choices=("e2e", "kernel", "both"),
        default="e2e",
        help=(
            "e2e includes pre-dispatch registration; kernel uses fixed registered "
            "inputs; both records each boundary independently"
        ),
    )
    parser.add_argument(
        "--num-experts-per-wave",
        type=int,
        help=(
            "scheduler experts per FC1/FC2 wave; defaults to all local experts "
            "(48 for Pro EP8 and 32 for Flash EP8)"
        ),
    )
    parser.add_argument("--num-sms", type=int)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--fast-math", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--use-swap-ab",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if not args.batches or any(m <= 0 for m in args.batches):
        parser.error("--batches must contain positive values")
    if len(set(args.batches)) != len(args.batches):
        parser.error("--batches must not contain duplicates")
    if args.num_max_tokens_per_rank <= 0:
        parser.error("--num-max-tokens-per-rank must be positive")
    if not args.match_cap_to_m and args.num_max_tokens_per_rank < max(args.batches):
        parser.error("--num-max-tokens-per-rank must cover the largest batch")
    if args.expected_world_size <= 0:
        parser.error("--expected-world-size must be positive")
    if min(args.observations, args.num_tests) <= 0:
        parser.error("--observations and --num-tests must be positive")
    if min(args.warmups, args.flush_l2_bytes, args.barrier_sleep_cycles) < 0:
        parser.error("warmups, flush size, and sleep cycles must be non-negative")
    if args.num_experts_per_wave is not None and args.num_experts_per_wave <= 0:
        parser.error("--num-experts-per-wave must be positive")
    if math.isnan(args.activation_clamp) or args.activation_clamp <= 0:
        parser.error("--activation-clamp must be positive or infinity")
    return args


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source_paths() -> dict[str, Path]:
    source_dir = Path(__file__).resolve().parent
    return {name: source_dir / name for name in SOURCE_SHA256}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source_snapshot() -> dict:
    files = {}
    mismatches = []
    for name, path in _source_paths().items():
        if not path.is_file():
            raise FileNotFoundError(f"missing vendored Gluon source: {path}")
        actual = _sha256_file(path)
        expected = SOURCE_SHA256[name]
        matches = actual == expected
        files[name] = {
            "path": str(path),
            "sha256": actual,
            "expected_sha256": expected,
            "matches_snapshot": matches,
        }
        if not matches:
            mismatches.append(f"{name}: expected {expected}, got {actual}")
    if mismatches:
        raise RuntimeError(
            f"vendored Gluon sources do not match {SOURCE_SNAPSHOT}: "
            + "; ".join(mismatches)
        )
    return {
        "snapshot": SOURCE_SNAPSHOT,
        "files": files,
        "matches_snapshot": True,
    }


def _load_gluon_sources():
    paths = _source_paths()
    gluon = _load_module(
        "notebook_mega_moe_gluon",
        paths["mega_moe_gluon.py"],
    )
    reference = _load_module(
        "notebook_mega_moe_gluon_reference",
        paths["mega_moe_gluon_reference.py"],
    )
    return gluon, reference


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _make_random_topk_routes(args, model: dict, m: int, rank: int, device):
    import torch

    generator = torch.Generator(device=device).manual_seed(
        2903 + args.seed_offset + rank
    )
    scores = torch.randn(
        (m, model["num_experts"]),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    topk_weights, topk_idx = torch.topk(
        scores,
        model["top_k"],
        dim=-1,
        largest=True,
        sorted=False,
    )
    return topk_idx.contiguous(), topk_weights.contiguous()


def _make_case_data(args, reference, model: dict, m: int, rank: int, device):
    import torch

    x_generator = torch.Generator(device=device).manual_seed(
        1103 + args.seed_offset + rank
    )
    x_bf16 = torch.randn(
        (m, model["hidden"]),
        dtype=torch.bfloat16,
        device=device,
        generator=x_generator,
    ).mul_(0.1)
    x_fp8, x_sf = reference.quantize_per_token_per_128(x_bf16)
    del x_bf16

    topk_idx, topk_weights = _make_random_topk_routes(args, model, m, rank, device)
    local_experts = model["num_experts"] // args.expected_world_size

    l1_generator = torch.Generator(device=device).manual_seed(
        7701 + args.seed_offset + rank
    )
    l1_float = torch.randn(
        (local_experts, 2 * model["intermediate"], model["hidden"]),
        dtype=torch.float32,
        device=device,
        generator=l1_generator,
    ).mul_(0.05)
    l1_weight, l1_weight_sf = reference.quantize_weight_block_128x128(l1_float)
    del l1_float

    l2_generator = torch.Generator(device=device).manual_seed(
        8801 + args.seed_offset + rank
    )
    l2_float = torch.randn(
        (local_experts, model["hidden"], model["intermediate"]),
        dtype=torch.float32,
        device=device,
        generator=l2_generator,
    ).mul_(0.05)
    l2_weight, l2_weight_sf = reference.quantize_weight_block_128x128(l2_float)
    del l2_float

    l1_interleaved = reference.interleave_l1_weight_for_sm90(l1_weight)
    del l1_weight
    return (
        x_fp8,
        x_sf,
        topk_idx,
        topk_weights,
        l1_interleaved,
        l1_weight_sf,
        l2_weight,
        l2_weight_sf,
    )


def _expected_local_routes(topk_idx, num_experts: int, rank: int, world_size: int):
    import torch
    import torch.distributed as dist

    counts = torch.bincount(topk_idx.flatten(), minlength=num_experts)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    local_experts = num_experts // world_size
    start = rank * local_experts
    return counts[start : start + local_experts].to(torch.int)


def _route_distribution(topk_idx, num_experts: int, world_size: int) -> dict:
    import torch
    import torch.distributed as dist

    counts = torch.bincount(topk_idx.flatten(), minlength=num_experts).to(torch.int64)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    experts_per_rank = num_experts // world_size
    rows_per_rank = counts.view(world_size, experts_per_rank).sum(dim=1)
    active = counts[counts > 0]
    return {
        "active_experts": int(active.numel()),
        "min_active_expert_rows": int(active.min().item()),
        "max_active_expert_rows": int(active.max().item()),
        "rows_per_destination_rank": rows_per_rank.cpu().tolist(),
    }


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


def _validate_output(output, model: dict, m: int, series: str) -> None:
    import torch

    expected_shape = (m, model["hidden"])
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"invalid {series} output shape {tuple(output.shape)}, "
            f"expected {expected_shape}"
        )
    if output.dtype != torch.bfloat16:
        raise RuntimeError(f"invalid {series} output dtype {output.dtype}")
    if not bool(torch.isfinite(output.float()).all().item()):
        raise RuntimeError(f"{series} output contains non-finite values")


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

    dist.barrier()
    for _ in range(warmups):
        call()
    torch.cuda.synchronize()
    dist.barrier()

    samples_us = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(num_tests):
        if barrier_sleep_cycles:
            torch.cuda._sleep(barrier_sleep_cycles)
            torch.cuda.synchronize()
        dist.barrier()
        if flush is not None:
            flush.zero_()
        start.record()
        call()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1_000.0)
    return samples_us


def _emit_observation(
    *,
    args,
    model_name: str,
    model: dict,
    m: int,
    cap: int,
    series: str,
    observation: int,
    samples_us: list[float],
    routes,
    route_distribution: dict,
    actual_flush_bytes: int,
    rank: int,
    world_size: int,
) -> dict | None:
    import torch
    import torch.distributed as dist

    route_list = routes.cpu().tolist()
    local_row = {
        "backend": BACKEND,
        "kernel_name": KERNEL_NAME,
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
        "seed_offset": args.seed_offset,
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
        "requested_flush_l2_bytes": args.flush_l2_bytes,
        "actual_flush_l2_bytes": actual_flush_bytes,
    }
    print("GLUON_STAT_JSON " + json.dumps(local_row, sort_keys=True), flush=True)

    local_tensor = torch.tensor(samples_us, dtype=torch.float64, device="cuda")
    gathered_samples = [torch.empty_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_samples, local_tensor)
    gathered_rows = [None] * world_size
    dist.all_gather_object(
        gathered_rows,
        {key: value for key, value in local_row.items() if key != "samples_us"},
    )
    if rank != 0:
        return None

    rank_samples = torch.stack(gathered_samples).cpu()
    critical_samples = rank_samples.max(dim=0).values.tolist()
    per_rank_medians = [statistics.median(row.tolist()) for row in rank_samples]
    slowest_rank = max(range(world_size), key=per_rank_medians.__getitem__)
    critical_median = statistics.median(critical_samples)
    aggregate = {
        "backend": BACKEND,
        "kernel_name": KERNEL_NAME,
        "series": series,
        "model": model_name,
        "m": m,
        "cap": cap,
        "world_size": world_size,
        "seed_offset": args.seed_offset,
        "observation": observation,
        "num_samples": len(samples_us),
        "critical_path_median_us": critical_median,
        "critical_path_mean_us": statistics.mean(critical_samples),
        "critical_path_min_us": min(critical_samples),
        "critical_path_max_us": max(critical_samples),
        "critical_path_samples_us": critical_samples,
        "per_rank_median_us": per_rank_medians,
        "slowest_rank_by_median": slowest_rank,
        "slowest_rank_median_us": per_rank_medians[slowest_rank],
        "max_rank": slowest_rank,
        "max_rank_us": per_rank_medians[slowest_rank],
        "per_rank_us": per_rank_medians,
        "rank_samples_us": rank_samples.tolist(),
        "routes": [
            {
                "rank": row["rank"],
                "route_counts": row["route_counts"],
                "route_total": row["route_total"],
                "touched_experts": row["touched_experts"],
            }
            for row in sorted(gathered_rows, key=lambda row: row["rank"])
        ],
        "route_distribution": route_distribution,
        "requested_flush_l2_bytes": args.flush_l2_bytes,
        "actual_flush_l2_bytes_min": min(
            row["actual_flush_l2_bytes"] for row in gathered_rows
        ),
        "analysis_rule": (
            "CUDA Event per active iteration; maximum across ranks per iteration; "
            "median across critical-path samples"
        ),
        "max_rank_analysis_rule": (
            "median per rank across active samples; maximum rank median"
        ),
    }
    print(
        "GLUON_OBSERVATION_JSON " + json.dumps(aggregate, sort_keys=True),
        flush=True,
    )
    return {
        "critical_path_median_us": critical_median,
        "max_rank": slowest_rank,
        "max_rank_us": per_rank_medians[slowest_rank],
    }


def _run_series(
    call: Callable,
    *,
    prepare: Callable | None,
    args,
    model_name: str,
    model: dict,
    m: int,
    cap: int,
    series: str,
    routes,
    route_distribution: dict,
    num_experts_per_wave: int,
    rank: int,
    world_size: int,
) -> None:
    import torch
    import torch.distributed as dist

    if prepare is not None:
        prepare()
        torch.cuda.synchronize()
        dist.barrier()
    output = call()
    torch.cuda.synchronize()
    dist.barrier()
    _validate_output(output, model, m, series)

    flush, actual_flush_bytes = _allocate_flush(args.flush_l2_bytes)
    observation_critical_path_medians = []
    observation_max_rank_us = []
    try:
        for observation in range(1, args.observations + 1):
            samples_us = _time_samples(
                call,
                warmups=args.warmups,
                num_tests=args.num_tests,
                flush=flush,
                barrier_sleep_cycles=args.barrier_sleep_cycles,
            )
            observation_result = _emit_observation(
                args=args,
                model_name=model_name,
                model=model,
                m=m,
                cap=cap,
                series=series,
                observation=observation,
                samples_us=samples_us,
                routes=routes,
                route_distribution=route_distribution,
                actual_flush_bytes=actual_flush_bytes,
                rank=rank,
                world_size=world_size,
            )
            if rank == 0:
                assert observation_result is not None
                observation_critical_path_medians.append(
                    observation_result["critical_path_median_us"]
                )
                observation_max_rank_us.append(observation_result["max_rank_us"])
            dist.barrier()
    finally:
        del flush

    if rank == 0:
        summary = {
            "backend": BACKEND,
            "kernel_name": KERNEL_NAME,
            "series": series,
            "model": model_name,
            "m": m,
            "cap": cap,
            "world_size": world_size,
            "observations": args.observations,
            "warmups": args.warmups,
            "num_tests": args.num_tests,
            "max_rank_median_us": statistics.median(observation_max_rank_us),
            "max_rank_min_us": min(observation_max_rank_us),
            "max_rank_max_us": max(observation_max_rank_us),
            "observation_max_rank_us": observation_max_rank_us,
            "critical_path_median_us": statistics.median(
                observation_critical_path_medians
            ),
            "critical_path_min_us": min(observation_critical_path_medians),
            "critical_path_max_us": max(observation_critical_path_medians),
            "observation_critical_path_median_us": (observation_critical_path_medians),
            "route_distribution": route_distribution,
            "num_experts_per_wave": num_experts_per_wave,
            "num_sms": args.num_sms,
            "activation": "swiglu",
            "activation_clamp": args.activation_clamp,
            "fast_math": bool(args.fast_math),
            "use_swap_ab": args.use_swap_ab,
            "critical_path_analysis_rule": (
                "median across observations of the median per-iteration "
                "max-rank CUDA Event latency"
            ),
            "max_rank_analysis_rule": (
                "median across observations of the maximum per-rank median"
            ),
        }
        print("GLUON_SUMMARY_JSON " + json.dumps(summary, sort_keys=True), flush=True)


def _execute_point(
    args,
    gluon,
    reference,
    model_name: str,
    m: int,
    rank: int,
    world_size: int,
) -> None:
    import torch
    import torch.distributed as dist

    model = MODEL_CONFIGS[model_name]
    cap = m if args.match_cap_to_m else args.num_max_tokens_per_rank
    num_experts_per_wave = args.num_experts_per_wave or (
        model["num_experts"] // world_size
    )
    device = torch.device("cuda", torch.cuda.current_device())
    (
        x_fp8,
        x_sf,
        topk_idx,
        topk_weights,
        l1_weight,
        l1_weight_sf,
        l2_weight,
        l2_weight_sf,
    ) = _make_case_data(args, reference, model, m, rank, device)
    routes = _expected_local_routes(topk_idx, model["num_experts"], rank, world_size)
    distribution = _route_distribution(topk_idx, model["num_experts"], world_size)
    context = gluon.create_sm90_mega_moe_symmetric_context(
        max_tokens=cap,
        hidden=model["hidden"],
        num_experts=model["num_experts"],
        topk=model["top_k"],
    )
    workspace = None
    registered_holder = [None]

    def launch(*, registered=None):
        return gluon.run_sm90_fused_dispatch_1d2d_compact_symmetric(
            context,
            x_fp8,
            x_sf,
            topk_idx,
            topk_weights,
            l1_weight,
            l1_weight_sf,
            l2_weight,
            l2_weight_sf,
            block_m=BLOCK_M,
            group_n=GROUP_N,
            group_k=GROUP_K,
            num_experts_per_wave=num_experts_per_wave,
            num_sms=args.num_sms,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
            workspace=workspace,
            pre_dispatch_result=registered,
            use_swap_ab=args.use_swap_ab,
        )

    first = launch()
    torch.cuda.synchronize()
    workspace = first.workspace
    registered_holder[0] = first.pre_dispatch
    _validate_output(first.output, model, m, "initial")

    def run_e2e():
        return launch().output

    def prepare_kernel():
        registered_holder[0] = gluon.run_sm90_mega_moe_pre_dispatch(
            context,
            x_fp8,
            topk_idx,
            topk_weights,
            x_sf=x_sf,
        )
        return registered_holder[0]

    def run_kernel():
        return launch(registered=registered_holder[0]).output

    series = []
    if args.timing_scope in ("e2e", "both"):
        series.append(("e2e", run_e2e, None))
    if args.timing_scope in ("kernel", "both"):
        series.append(("kernel", run_kernel, prepare_kernel))
    for series_name, call, prepare in series:
        _run_series(
            call,
            prepare=prepare,
            args=args,
            model_name=model_name,
            model=model,
            m=m,
            cap=cap,
            series=series_name,
            routes=routes,
            route_distribution=distribution,
            num_experts_per_wave=num_experts_per_wave,
            rank=rank,
            world_size=world_size,
        )
    dist.barrier()


def _run_point(
    args,
    gluon,
    reference,
    model_name: str,
    m: int,
    rank: int,
    world_size: int,
) -> None:
    import torch
    import torch.distributed as dist

    # Keep all CUDA and symmetric-memory owners inside _execute_point.  Once it
    # returns, no closure or result object retains the workspace, registered
    # inputs, context handles, activations, or weights, so cache cleanup is real.
    # Exceptions deliberately propagate: torchrun terminates peer ranks instead
    # of letting one failed rank enter a different collective sequence.
    _execute_point(args, gluon, reference, model_name, m, rank, world_size)
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()


def main() -> int:
    args = _parse_args()

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise SystemExit(
            "multi-rank Gluon benchmark must be launched with torchrun; "
            "see this file's module docstring"
        )
    source_identity = _validate_source_snapshot()

    try:
        import torch
        import torch.distributed as dist
        import triton
    except ImportError as exc:
        raise SystemExit(
            "PyTorch, Triton >= 3.6, and the Gluon runtime are required. "
            f"Original import error: {exc}"
        ) from exc

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    completed = False
    try:
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) != (9, 0):
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
            if model["hidden"] % GROUP_K or model["intermediate"] % GROUP_N:
                raise RuntimeError(f"{name} hidden dimensions must align to 128")
            experts_per_rank = model["num_experts"] // world_size
            if (
                args.num_experts_per_wave is not None
                and args.num_experts_per_wave > experts_per_rank
            ):
                raise RuntimeError(
                    "--num-experts-per-wave must not exceed experts per rank "
                    f"({experts_per_rank})"
                )

        physical_sms = torch.cuda.get_device_properties(
            local_rank
        ).multi_processor_count
        if args.num_sms is None:
            args.num_sms = physical_sms
        if not 0 < args.num_sms <= physical_sms:
            raise RuntimeError(
                f"--num-sms must be in [1, {physical_sms}], got {args.num_sms}"
            )

        gluon, reference = _load_gluon_sources()

        def alloc_fn(size: int, alignment: int, stream):
            del alignment, stream
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

        if rank == 0:
            plan = {
                "backend": BACKEND,
                "kernel_name": KERNEL_NAME,
                "source_identity": source_identity,
                "torch": torch.__version__,
                "triton": triton.__version__,
                "triton_kernels": _package_version("triton-kernels"),
                "models": args.model_config,
                "batches": args.batches,
                "world_size": world_size,
                "num_max_tokens_per_rank": args.num_max_tokens_per_rank,
                "match_cap_to_m": args.match_cap_to_m,
                "series": args.timing_scope,
                "seed_offset": args.seed_offset,
                "rng_seeds": {
                    "x": "1103 + seed_offset + rank",
                    "routes": "2903 + seed_offset + rank",
                    "l1": "7701 + seed_offset + rank",
                    "l2": "8801 + seed_offset + rank",
                },
                "observations": args.observations,
                "warmups": args.warmups,
                "num_tests": args.num_tests,
                "flush_l2_bytes": args.flush_l2_bytes,
                "num_experts_per_wave": args.num_experts_per_wave,
                "num_experts_per_wave_policy": (
                    "explicit"
                    if args.num_experts_per_wave is not None
                    else "all_local_experts"
                ),
                "num_sms": args.num_sms,
                "activation": "swiglu",
                "activation_clamp": args.activation_clamp,
                "fast_math": bool(args.fast_math),
                "use_swap_ab": args.use_swap_ab,
                "analysis_rule": (
                    "CUDA Event per active iteration; maximum across ranks per "
                    "iteration; median across critical-path samples"
                ),
            }
            print("GLUON_PLAN_JSON " + json.dumps(plan, sort_keys=True), flush=True)

        for model_name in args.model_config:
            for m in args.batches:
                if rank == 0:
                    cap = m if args.match_cap_to_m else args.num_max_tokens_per_rank
                    print(
                        f"# Gluon model={model_name} M={m} cap={cap} "
                        f"world={world_size} scope={args.timing_scope}",
                        flush=True,
                    )
                _run_point(
                    args,
                    gluon,
                    reference,
                    model_name,
                    m,
                    rank,
                    world_size,
                )
        completed = True
        return 0
    finally:
        if dist.is_initialized():
            if completed:
                dist.barrier()
            with contextlib.suppress(Exception):
                dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
