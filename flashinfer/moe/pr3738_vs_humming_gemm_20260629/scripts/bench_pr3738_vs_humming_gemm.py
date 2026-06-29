#!/usr/bin/env python3
"""FlashInfer PR #3738 GEMM profiler vs upstream Humming indexed GEMM.

This benchmark intentionally measures GEMM scope only:
- FlashInfer calls the PR #3738 CUTLASS fused-MoE runner's run_gemm_profile
  with gemm_idx=1 for w13/GEMM1 and gemm_idx=2 for w2/GEMM2.
- Humming calls HummingLayer with GemmType.INDEXED for w13 and w2 separately.

The script is intentionally environment-agnostic. It records only benchmark
metadata needed for reproducibility; deployment and installation details belong
in the companion README for the benchmark run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int

    def local_intermediate(self, tp_size: int) -> int:
        if self.intermediate_size % tp_size != 0:
            raise ValueError(f"{self.name} intermediate {self.intermediate_size} not divisible by TP{tp_size}")
        return self.intermediate_size // tp_size

    def local_experts(self, ep_size: int) -> int:
        if self.num_experts % ep_size != 0:
            raise ValueError(f"{self.name} experts {self.num_experts} not divisible by EP{ep_size}")
        return self.num_experts // ep_size


WORKLOADS = {
    "dsv4-flash": WorkloadSpec("dsv4-flash", 4096, 2048, 256, 6),
    "dsv4-pro": WorkloadSpec("dsv4-pro", 7168, 3072, 384, 6),
}

DEFAULT_TOPOLOGIES = [(1, 8), (2, 4), (4, 2), (8, 1)]
DEFAULT_BATCHES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
FLASHINFER_BACKEND = "flashinfer_pr3738"
HUMMING_BACKEND = "humming_indexed"


def log(msg: str, logfile: Path | None = None) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if logfile is not None:
        with logfile.open("a") as f:
            f.write(line + "\n")


def repo_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        marker = path / ".commit"
        return marker.read_text().strip() if marker.exists() else ""


def stable_seed(seed: int, workload: str, tp_size: int, ep_size: int, batch: int) -> int:
    tag = sum((i + 1) * ord(ch) for i, ch in enumerate(workload))
    return seed + tag + tp_size * 1009 + ep_size * 917 + batch * 65537


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[idx])


def cuda_event_bench(fn, warmup: int, repeat: int) -> list[float]:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return times


def make_topk_ids(batch: int, num_experts: int, top_k: int, seed: int):
    import torch

    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    logits = torch.randn((batch, num_experts), dtype=torch.float32, device="cuda", generator=gen)
    return torch.topk(logits, top_k, dim=1).indices.to(torch.int32).contiguous()


def route_stats(topk_ids) -> dict[str, Any]:
    import torch

    flat = topk_ids.reshape(-1).to(torch.int64)
    if flat.numel() == 0:
        return {
            "active_experts": 0,
            "min_group_m": 0,
            "max_group_m": 0,
            "mean_group_m": 0.0,
            "std_group_m": 0.0,
        }
    counts = torch.bincount(flat).detach().cpu().tolist()
    counts = [int(x) for x in counts if int(x) > 0]
    return {
        "active_experts": len(counts),
        "min_group_m": min(counts),
        "max_group_m": max(counts),
        "mean_group_m": float(statistics.mean(counts)),
        "std_group_m": float(statistics.pstdev(counts)) if len(counts) > 1 else 0.0,
    }


def indexed_moe_tensors_from_topk_ids(topk_ids, num_experts: int, block_size: int):
    import torch

    flat = topk_ids.reshape(-1).to(torch.int32)
    part_token_ids_list = []
    expert_id_list = []
    for expert_id in range(num_experts):
        part_token_ids = torch.where(flat == expert_id)[0]
        num_blocks = math.ceil(part_token_ids.size(0) / block_size)
        if num_blocks == 0:
            continue
        padded_size = num_blocks * block_size
        part_token_ids = torch.nn.functional.pad(
            part_token_ids,
            pad=(0, padded_size - part_token_ids.size(0)),
            value=flat.nelement(),
        )
        part_token_ids_list.append(part_token_ids)
        expert_id_list += [expert_id] * num_blocks
    if not part_token_ids_list:
        raise RuntimeError("indexed route has no active expert blocks")
    sorted_ids = torch.cat(part_token_ids_list).to(torch.int32)
    expert_ids = torch.tensor(expert_id_list, dtype=torch.int32, device=flat.device)
    num_tokens_padded = torch.tensor(sorted_ids.size(0), dtype=torch.int32, device=flat.device)
    return sorted_ids, expert_ids, num_tokens_padded


def choose_indexed_block_size(tuning_config, shape_m: int, top_k: int) -> int:
    routed_shape_m = shape_m * top_k
    for min_shape_m, max_shape_m, config in tuning_config:
        if routed_shape_m > min_shape_m and routed_shape_m <= max_shape_m:
            return int(config["block_shape"][0])
    raise RuntimeError(f"No Humming indexed block config for routed_shape_m={routed_shape_m}")


def reset_imports(prefixes: tuple[str, ...]) -> None:
    for name in list(sys.modules):
        if name in prefixes or any(name.startswith(prefix + ".") for prefix in prefixes):
            del sys.modules[name]


def patch_cutlass_dsl_import_compat() -> None:
    import pkgutil
    import types

    try:
        import cutlass.cute as cute
    except Exception:
        return

    if not hasattr(cute.nvgpu, "OperandMajorMode") and hasattr(cute.nvgpu, "tcgen05"):
        cute.nvgpu.OperandMajorMode = cute.nvgpu.tcgen05.OperandMajorMode
    if "cutlass.cute.experimental" not in sys.modules:
        experimental = types.ModuleType("cutlass.cute.experimental")
        experimental.__path__ = []
        sys.modules["cutlass.cute.experimental"] = experimental
        setattr(cute, "experimental", experimental)
    if not getattr(pkgutil.walk_packages, "_flashinfer_pr3738_filtered", False):
        original_walk_packages = pkgutil.walk_packages

        def filtered_walk_packages(path=None, prefix="", onerror=None):
            for info in original_walk_packages(path, prefix, onerror):
                if info.name.startswith("cutlass.cute.experimental"):
                    continue
                yield info

        filtered_walk_packages._flashinfer_pr3738_filtered = True
        pkgutil.walk_packages = filtered_walk_packages


def configure_flashinfer_repo(repo: Path, output_dir: Path):
    reset_imports(("flashinfer",))
    patch_cutlass_dsl_import_compat()
    sys.path.insert(0, str(repo))
    import torch
    from flashinfer.jit import env as jit_env

    workspace = Path(
        os.environ.get(
            "FLASHINFER_BENCH_JIT_WORKSPACE",
            str(output_dir / "flashinfer_jit_workspace"),
        )
    )
    jit_env.FLASHINFER_WORKSPACE_DIR = workspace
    jit_env.FLASHINFER_JIT_DIR = workspace / "cached_ops"
    jit_env.FLASHINFER_GEN_SRC_DIR = workspace / "generated"
    jit_env.FLASHINFER_AOT_DIR = output_dir / "flashinfer_aot"
    jit_env.FLASHINFER_INCLUDE_DIR = repo / "include"
    jit_env.FLASHINFER_CSRC_DIR = repo / "csrc"
    jit_env.FLASHINFER_DATA = repo
    jit_env.CUTLASS_INCLUDE_DIRS = [
        repo / "3rdparty" / "cutlass" / "include",
        repo / "3rdparty" / "cutlass" / "tools" / "util" / "include",
    ]
    jit_env.SPDLOG_INCLUDE_DIR = repo / "3rdparty" / "spdlog" / "include"
    jit_env.CCCL_INCLUDE_DIRS = [
        repo / "3rdparty" / "cccl" / "cub",
        repo / "3rdparty" / "cccl" / "libcudacxx" / "include",
        repo / "3rdparty" / "cccl" / "thrust",
    ]

    from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm90_module
    from flashinfer.fused_moe import (
        ActivationType,
        interleave_moe_scales_for_sm90_mixed_gemm,
        interleave_moe_weights_for_sm90_mixed_gemm,
        preprocess_moe_weights_for_sm90_mixed_gemm_humming,
    )

    module = gen_cutlass_fused_moe_sm90_module().build_and_load()
    module.set_deepgemm_jit_include_dirs([str(jit_env.FLASHINFER_CSRC_DIR / "nv_internal" / "tensorrt_llm")])
    runner = module.init(
        torch.bfloat16,
        torch.uint8,
        torch.bfloat16,
        False,
        True,
        False,
        False,
        True,
    )
    return {
        "torch": torch,
        "runner": runner,
        "ActivationType": ActivationType,
        "interleave_weight": interleave_moe_weights_for_sm90_mixed_gemm,
        "interleave_scale": interleave_moe_scales_for_sm90_mixed_gemm,
        "preprocess_humming": preprocess_moe_weights_for_sm90_mixed_gemm_humming,
        "jit_workspace": workspace,
    }


def configure_humming_repo(repo: Path):
    reset_imports(("humming",))
    sys.path.insert(0, str(repo))
    from humming import ops
    from humming.config import GemmType
    from humming.jit.compiler import NVRTCCompiler
    from humming.layer import HummingLayer
    from humming.tune import get_heuristics_config
    import humming.utils.cuda as humming_cuda_utils
    from humming.utils.test import random_fill_tensor

    cuda_target_include_candidates = []
    if os.environ.get("CUDA_TARGET_INCLUDE_PATH"):
        cuda_target_include_candidates.append(Path(os.environ["CUDA_TARGET_INCLUDE_PATH"]))
    if os.environ.get("CUDA_HOME"):
        cuda_target_include_candidates.append(Path(os.environ["CUDA_HOME"]) / "targets/x86_64-linux/include")
    cuda_target_include_candidates.append(
        Path(os.sep) / "usr" / "local" / "cuda" / "targets/x86_64-linux/include"
    )
    original_find_all_cuda_paths = humming_cuda_utils.find_all_cuda_paths

    def find_all_cuda_paths_with_cuda_targets():
        envs = []
        for env in original_find_all_cuda_paths():
            patched_env = dict(env)
            include_dirs = list(patched_env.get("include_paths", []))
            for candidate in cuda_target_include_candidates:
                if candidate.is_dir():
                    include_dir_str = candidate.as_posix()
                    if include_dir_str not in include_dirs:
                        include_dirs.append(include_dir_str)
            patched_env["include_paths"] = include_dirs
            envs.append(patched_env)
        return envs

    humming_cuda_utils.find_all_cuda_paths = find_all_cuda_paths_with_cuda_targets
    original_get_include_dirs = NVRTCCompiler._get_include_dirs.__func__

    @classmethod
    def get_include_dirs_with_cuda_targets(cls):
        include_dirs = list(original_get_include_dirs(cls))
        for include_dir in cuda_target_include_candidates:
            if include_dir.is_dir():
                include_dir_str = include_dir.as_posix()
                if include_dir_str not in include_dirs:
                    include_dirs.append(include_dir_str)
        return include_dirs

    NVRTCCompiler._get_include_dirs = get_include_dirs_with_cuda_targets

    return {
        "ops": ops,
        "GemmType": GemmType,
        "HummingLayer": HummingLayer,
        "get_heuristics_config": get_heuristics_config,
        "random_fill_tensor": random_fill_tensor,
    }


def make_uint8_weight(rows: int, cols: int, seed: int):
    import torch

    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    weight = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, device="cuda", generator=gen)
    scale = torch.randint(114, 128, (rows, cols // 32), dtype=torch.uint8, device="cuda", generator=gen)
    return weight.contiguous(), scale.contiguous()


def make_flashinfer_case(fi: dict[str, Any], spec: WorkloadSpec, tp_size: int, ep_size: int, batch: int, seed: int):
    torch = fi["torch"]
    local_i = spec.local_intermediate(tp_size)
    local_e = spec.local_experts(ep_size)
    case_seed = stable_seed(seed, spec.name, tp_size, ep_size, batch)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(case_seed)
    x = torch.randn((batch, spec.hidden_size), dtype=torch.bfloat16, device="cuda", generator=gen).contiguous()

    w13, w13_scale = make_uint8_weight(local_e * 2 * local_i, spec.hidden_size, case_seed + 13)
    w2, w2_scale = make_uint8_weight(local_e * spec.hidden_size, local_i, case_seed + 29)
    w13 = w13.reshape(local_e, 2 * local_i, spec.hidden_size // 2)
    w13_scale = w13_scale.reshape(local_e, 2 * local_i, spec.hidden_size // 32)
    w2 = w2.reshape(local_e, spec.hidden_size, local_i // 2)
    w2_scale = w2_scale.reshape(local_e, spec.hidden_size, local_i // 32)

    w13_il, w13_scale_il, _ = fi["preprocess_humming"](w13, w13_scale)
    w2_il, w2_scale_il, _ = fi["preprocess_humming"](w2, w2_scale)
    # run_gemm_profile currently consumes only the interleaved weight pointer, but
    # keep scale tensors alive with the case to preserve the intended weight layout context.
    return {
        "x": x,
        "w13": w13_il,
        "w2": w2_il,
        "w13_scale": w13_scale_il,
        "w2_scale": w2_scale_il,
    }


def valid_flashinfer_tactics(runner, stage: int, stage_start: int, stage_count: int, gemm_n: int, gemm_k: int) -> list[int]:
    all_tactics = list(range(stage_start, stage_start + stage_count))
    try:
        shape_valid = set(int(x) for x in runner.get_valid_tactics_for_shape(stage, gemm_n, gemm_k))
        filtered = [t for t in all_tactics if t in shape_valid]
        if filtered:
            all_tactics = filtered
    except Exception:
        pass
    kept = []
    for tactic in all_tactics:
        try:
            if int(runner.get_tactic_occupancy(tactic)) > 0:
                kept.append(tactic)
        except Exception:
            kept.append(tactic)
    return kept if kept else all_tactics


def run_flashinfer_stage(
    fi: dict[str, Any],
    case: dict[str, Any],
    spec: WorkloadSpec,
    tp_size: int,
    ep_size: int,
    stage: int,
    warmup: int,
    repeat: int,
    tactic_warmup: int,
    tactic_repeat: int,
):
    runner = fi["runner"]
    activation_type = int(fi["ActivationType"].Swiglu)
    gemm1_count = int(runner.get_gemm1_tactic_count())
    gemm2_count = int(runner.get_gemm2_tactic_count())
    if stage == 1:
        stage_start = 0
        stage_count = gemm1_count
        gemm_n = int(case["w13"].shape[1])
        gemm_k = int(case["x"].shape[1])
    else:
        stage_start = gemm1_count
        stage_count = gemm2_count
        gemm_n = int(case["w2"].shape[1])
        gemm_k = int(case["w2"].shape[2]) * 2
    tactics = valid_flashinfer_tactics(runner, stage, stage_start, stage_count, gemm_n, gemm_k)
    if not tactics:
        raise RuntimeError(f"No FlashInfer tactics for stage={stage}, N={gemm_n}, K={gemm_k}")

    def profile_call(profile_id: int, do_preparation: bool = False):
        return runner.run_gemm_profile(
            case["x"],
            case["w13"],
            None,
            case["w2"],
            None,
            spec.top_k,
            tp_size,
            0,
            ep_size,
            0,
            1,
            0,
            False,
            False,
            stage,
            profile_id,
            do_preparation,
            False,
            activation_type,
        )

    profile_call(tactics[0], True)
    fi["torch"].cuda.synchronize()
    tactic_times: dict[int, float] = {}
    for tactic in tactics:
        try:
            times = cuda_event_bench(lambda t=tactic: profile_call(t, False), tactic_warmup, tactic_repeat)
            tactic_times[tactic] = float(statistics.median(times))
        except Exception as exc:
            tactic_times[tactic] = float("inf")
            print(f"[WARN] FlashInfer tactic failed stage={stage} tactic={tactic}: {exc}", flush=True)
    finite = {k: v for k, v in tactic_times.items() if math.isfinite(v)}
    if not finite:
        raise RuntimeError(f"All FlashInfer tactics failed for stage={stage}: {tactic_times}")
    best_tactic = min(finite, key=finite.get)
    profile_call(best_tactic, True)
    fi["torch"].cuda.synchronize()
    times = cuda_event_bench(lambda: profile_call(best_tactic, False), warmup, repeat)
    return {
        "times_ms": times,
        "profile_id": best_tactic,
        "candidate_profile_ids": tactics,
        "candidate_median_ms": tactic_times,
        "gemm_n": gemm_n,
        "gemm_k": gemm_k,
        "gemm1_tactic_count": gemm1_count,
        "gemm2_tactic_count": gemm2_count,
    }


def make_humming_layer(hm: dict[str, Any], shape_n: int, shape_k: int, num_experts: int, seed: int):
    import torch

    torch.manual_seed(seed)
    layer = hm["HummingLayer"](
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=num_experts,
        weight_config={"dtype": "float4e2m1", "group_size": 32, "scale_dtype": "float8e8m0"},
        input_config={"dtype": "float8e4m3"},
        torch_dtype=torch.bfloat16,
    ).to("cuda:0")
    for tensor in layer.parameters():
        hm["random_fill_tensor"](tensor)
    layer.transform()
    meta = layer.humming_metas[""]
    tuning_config = hm["get_heuristics_config"](meta=meta, gemm_type=hm["GemmType"].INDEXED)
    compute_config = {"use_f16_accum": False, "gemm_type": hm["GemmType"].INDEXED.value}
    return layer, compute_config, tuning_config


def run_humming_case(
    hm: dict[str, Any],
    spec: WorkloadSpec,
    tp_size: int,
    ep_size: int,
    batch: int,
    seed: int,
    warmup: int,
    repeat: int,
):
    import torch

    local_i = spec.local_intermediate(tp_size)
    local_e = spec.local_experts(ep_size)
    case_seed = stable_seed(seed, spec.name, tp_size, ep_size, batch)
    topk_ids = make_topk_ids(batch, local_e, spec.top_k, case_seed + 101)
    pairs = batch * spec.top_k
    stats = route_stats(topk_ids)

    gen = torch.Generator(device="cuda")
    gen.manual_seed(case_seed + 201)
    x_bf16 = torch.randn((batch, spec.hidden_size), dtype=torch.bfloat16, device="cuda", generator=gen).contiguous()
    x_fp8, x_scale = hm["ops"].quant_input(x_bf16, "float8e4m3", group_size=0)

    w13_layer, w13_compute, w13_tuning = make_humming_layer(
        hm, 2 * local_i, spec.hidden_size, local_e, case_seed + 301
    )
    w13_block = choose_indexed_block_size(w13_tuning, batch, spec.top_k)
    w13_sorted, w13_expert_ids, w13_padded = indexed_moe_tensors_from_topk_ids(topk_ids, local_e, w13_block)
    w13_out = torch.empty((pairs, 2 * local_i), dtype=torch.bfloat16, device="cuda")

    def run_w13():
        return w13_layer(
            inputs=x_fp8,
            input_scale=x_scale,
            outputs=w13_out,
            sorted_ids=w13_sorted,
            expert_ids=w13_expert_ids,
            num_tokens_padded=w13_padded,
            top_k=spec.top_k,
            compute_config=w13_compute,
            tuning_config=w13_tuning,
        )

    run_w13()
    torch.cuda.synchronize()
    w13_times = cuda_event_bench(run_w13, warmup, repeat)

    gen.manual_seed(case_seed + 401)
    inter_bf16 = torch.randn((pairs, local_i), dtype=torch.bfloat16, device="cuda", generator=gen).contiguous()
    inter_fp8, inter_scale = hm["ops"].quant_input(inter_bf16, "float8e4m3", group_size=0)
    w2_layer, w2_compute, w2_tuning = make_humming_layer(
        hm, spec.hidden_size, local_i, local_e, case_seed + 501
    )
    w2_block = choose_indexed_block_size(w2_tuning, pairs, 1)
    w2_sorted, w2_expert_ids, w2_padded = indexed_moe_tensors_from_topk_ids(topk_ids.reshape(-1, 1), local_e, w2_block)
    w2_out = torch.empty((pairs, spec.hidden_size), dtype=torch.bfloat16, device="cuda")

    def run_w2():
        return w2_layer(
            inputs=inter_fp8,
            input_scale=inter_scale,
            outputs=w2_out,
            sorted_ids=w2_sorted,
            expert_ids=w2_expert_ids,
            num_tokens_padded=w2_padded,
            top_k=1,
            compute_config=w2_compute,
            tuning_config=w2_tuning,
        )

    run_w2()
    torch.cuda.synchronize()
    w2_times = cuda_event_bench(run_w2, warmup, repeat)

    return {
        "w13_times_ms": w13_times,
        "w2_times_ms": w2_times,
        "w13_block": w13_block,
        "w2_block": w2_block,
        "w13_num_tokens_padded": int(w13_padded.item()),
        "w2_num_tokens_padded": int(w2_padded.item()),
        "route_stats": stats,
    }


def raw_row(
    backend: str,
    segment: str,
    spec: WorkloadSpec,
    tp_size: int,
    ep_size: int,
    batch: int,
    times: list[float],
    warmup: int,
    repeat: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    local_i = spec.local_intermediate(tp_size)
    local_e = spec.local_experts(ep_size)
    row = {
        "backend": backend,
        "segment": segment,
        "workload": spec.name,
        "tp_size": tp_size,
        "ep_size": ep_size,
        "tp_rank": 0,
        "ep_rank": 0,
        "batch_per_rank": batch,
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "local_intermediate_size": local_i,
        "num_experts": spec.num_experts,
        "local_num_experts": local_e,
        "top_k": spec.top_k,
        "median_time_ms": float(statistics.median(times)),
        "mean_time_ms": float(statistics.mean(times)),
        "std_time_ms": float(statistics.pstdev(times)) if len(times) > 1 else 0.0,
        "min_time_ms": float(min(times)),
        "p20_time_ms": quantile(times, 0.2),
        "p80_time_ms": quantile(times, 0.8),
        "max_time_ms": float(max(times)),
        "warmup": warmup,
        "repeat": repeat,
    }
    if extra:
        row.update(extra)
    return row


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int, int, int, str, str], dict[str, Any]] = {}
    for row in raw_rows:
        key = (
            row["workload"],
            int(row["tp_size"]),
            int(row["ep_size"]),
            int(row["batch_per_rank"]),
            row["backend"],
            row["segment"],
        )
        by_key[key] = row

    summaries = []
    observed_topologies = sorted({(int(r["tp_size"]), int(r["ep_size"])) for r in raw_rows})
    for workload in sorted({r["workload"] for r in raw_rows}):
        for tp, ep in observed_topologies:
            for batch in sorted({int(r["batch_per_rank"]) for r in raw_rows if r["workload"] == workload}):
                def get(backend: str, segment: str):
                    return by_key.get((workload, tp, ep, batch, backend, segment))

                fw13 = get(FLASHINFER_BACKEND, "w13_grouped_gemm")
                fw2 = get(FLASHINFER_BACKEND, "w2_grouped_gemm")
                hw13 = get(HUMMING_BACKEND, "w13_grouped_gemm")
                hw2 = get(HUMMING_BACKEND, "w2_grouped_gemm")
                if not (fw13 and fw2 and hw13 and hw2):
                    continue
                flashinfer_w13 = float(fw13["median_time_ms"])
                flashinfer_w2 = float(fw2["median_time_ms"])
                humming_w13 = float(hw13["median_time_ms"])
                humming_w2 = float(hw2["median_time_ms"])
                flashinfer_total = flashinfer_w13 + flashinfer_w2
                humming_total = humming_w13 + humming_w2
                speedup = humming_total / flashinfer_total if flashinfer_total else float("nan")
                summaries.append(
                    {
                        "workload": workload,
                        "tp_size": tp,
                        "ep_size": ep,
                        "batch_per_rank": batch,
                        "flashinfer_w13_grouped_gemm_ms": flashinfer_w13,
                        "flashinfer_w2_grouped_gemm_ms": flashinfer_w2,
                        "flashinfer_gemm_total_ms": flashinfer_total,
                        "humming_w13_grouped_gemm_ms": humming_w13,
                        "humming_w2_grouped_gemm_ms": humming_w2,
                        "humming_gemm_total_ms": humming_total,
                        "w13_grouped_gemm_ms": flashinfer_w13,
                        "w2_grouped_gemm_ms": flashinfer_w2,
                        "gemm_total_ms": flashinfer_total,
                        "speedup": speedup,
                        "percent": (speedup - 1.0) * 100.0,
                        "speedup_formula": "humming_gemm_total_ms / flashinfer_gemm_total_ms",
                        "percent_formula": "(humming_gemm_total_ms / flashinfer_gemm_total_ms - 1) * 100",
                    }
                )
    return summaries


def write_plot(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    workloads = sorted({str(row["workload"]) for row in summaries})
    topologies = sorted({(int(row["tp_size"]), int(row["ep_size"])) for row in summaries})
    if not workloads or not topologies:
        return
    fig, axes = plt.subplots(
        len(workloads),
        len(topologies),
        figsize=(5 * len(topologies), 4 * len(workloads)),
        sharex=True,
        squeeze=False,
    )
    for r, workload in enumerate(workloads):
        for c, (tp, ep) in enumerate(topologies):
            ax = axes[r][c]
            rows = [
                x for x in summaries
                if x["workload"] == workload and int(x["tp_size"]) == tp and int(x["ep_size"]) == ep
            ]
            rows.sort(key=lambda x: int(x["batch_per_rank"]))
            xs = [int(x["batch_per_rank"]) for x in rows]
            ys = [float(x["percent"]) for x in rows]
            ax.axhline(0.0, color="#555555", linewidth=0.8)
            ax.plot(xs, ys, marker="o", linewidth=1.6, color="#1f77b4")
            ax.set_xscale("log", base=2)
            ax.grid(True, which="both", alpha=0.25)
            ax.set_title(f"{workload} TP{tp}/EP{ep}")
            if c == 0:
                ax.set_ylabel("FlashInfer faster than Humming (%)")
            if r == 1:
                ax.set_xlabel("batch per rank")
    fig.suptitle("FlashInfer PR #3738 GEMM profiler vs Humming indexed GEMM")
    fig.tight_layout()
    fig.savefig(output_dir / "pr3738_vs_humming_gemm.png", dpi=180)
    plt.close(fig)


def collect_env(flashinfer_repo: Path, humming_repo: Path) -> dict[str, Any]:
    import torch

    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.STDOUT,
        ).splitlines()[0].strip()
    except Exception as exc:
        driver = repr(exc)
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [
            {
                "idx": i,
                "name": torch.cuda.get_device_name(i),
                "capability": torch.cuda.get_device_capability(i),
            }
            for i in range(torch.cuda.device_count())
        ],
        "nvidia_driver": driver,
        "flashinfer_commit": repo_commit(flashinfer_repo),
        "humming_commit": repo_commit(humming_repo),
    }


def run_matrix(args: argparse.Namespace) -> int:
    import torch

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log = output_dir / "run.log"
    run_log.write_text("")
    env = collect_env(args.flashinfer_repo, args.humming_repo)
    (output_dir / "environment.json").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n")
    log(f"benchmark_metadata={json.dumps(env, sort_keys=True)}", run_log)

    fi = configure_flashinfer_repo(args.flashinfer_repo, output_dir)
    hm = configure_humming_repo(args.humming_repo)

    raw_rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for workload_name in args.workloads:
        spec = WORKLOADS[workload_name]
        for tp_size, ep_size in args.topologies:
            for batch in args.batches:
                case_label = f"{workload_name} TP{tp_size}/EP{ep_size} batch={batch}"
                log(f"RUN {case_label}", run_log)
                try:
                    seed = stable_seed(args.seed, workload_name, tp_size, ep_size, batch)
                    flash_case = make_flashinfer_case(fi, spec, tp_size, ep_size, batch, args.seed)
                    fw13 = run_flashinfer_stage(
                        fi,
                        flash_case,
                        spec,
                        tp_size,
                        ep_size,
                        1,
                        args.warmup,
                        args.repeat,
                        args.tactic_warmup,
                        args.tactic_repeat,
                    )
                    fw2 = run_flashinfer_stage(
                        fi,
                        flash_case,
                        spec,
                        tp_size,
                        ep_size,
                        2,
                        args.warmup,
                        args.repeat,
                        args.tactic_warmup,
                        args.tactic_repeat,
                    )
                    raw_rows.append(
                        raw_row(
                            FLASHINFER_BACKEND,
                            "w13_grouped_gemm",
                            spec,
                            tp_size,
                            ep_size,
                            batch,
                            fw13["times_ms"],
                            args.warmup,
                            args.repeat,
                            {
                                "scope": "run_gemm_profile_gemm_idx_1",
                                "profile_id": fw13["profile_id"],
                                "candidate_profile_ids_json": json.dumps(fw13["candidate_profile_ids"]),
                                "candidate_median_ms_json": json.dumps(fw13["candidate_median_ms"], sort_keys=True),
                                "gemm_n": fw13["gemm_n"],
                                "gemm_k": fw13["gemm_k"],
                                "case_seed": seed,
                            },
                        )
                    )
                    raw_rows.append(
                        raw_row(
                            FLASHINFER_BACKEND,
                            "w2_grouped_gemm",
                            spec,
                            tp_size,
                            ep_size,
                            batch,
                            fw2["times_ms"],
                            args.warmup,
                            args.repeat,
                            {
                                "scope": "run_gemm_profile_gemm_idx_2",
                                "profile_id": fw2["profile_id"],
                                "candidate_profile_ids_json": json.dumps(fw2["candidate_profile_ids"]),
                                "candidate_median_ms_json": json.dumps(fw2["candidate_median_ms"], sort_keys=True),
                                "gemm_n": fw2["gemm_n"],
                                "gemm_k": fw2["gemm_k"],
                                "case_seed": seed,
                            },
                        )
                    )

                    humming = run_humming_case(hm, spec, tp_size, ep_size, batch, args.seed, args.warmup, args.repeat)
                    raw_rows.append(
                        raw_row(
                            HUMMING_BACKEND,
                            "w13_grouped_gemm",
                            spec,
                            tp_size,
                            ep_size,
                            batch,
                            humming["w13_times_ms"],
                            args.warmup,
                            args.repeat,
                            {
                                "scope": "HummingLayer_GemmType.INDEXED_w13",
                                "indexed_block_size": humming["w13_block"],
                                "num_tokens_padded": humming["w13_num_tokens_padded"],
                                "case_seed": seed,
                                **humming["route_stats"],
                            },
                        )
                    )
                    raw_rows.append(
                        raw_row(
                            HUMMING_BACKEND,
                            "w2_grouped_gemm",
                            spec,
                            tp_size,
                            ep_size,
                            batch,
                            humming["w2_times_ms"],
                            args.warmup,
                            args.repeat,
                            {
                                "scope": "HummingLayer_GemmType.INDEXED_w2",
                                "indexed_block_size": humming["w2_block"],
                                "num_tokens_padded": humming["w2_num_tokens_padded"],
                                "case_seed": seed,
                                **humming["route_stats"],
                            },
                        )
                    )
                    summaries = build_summary(raw_rows)
                    write_table(output_dir / "raw_rows.csv", raw_rows)
                    (output_dir / "raw_rows.json").write_text(json.dumps({"rows": raw_rows}, indent=2, sort_keys=True) + "\n")
                    write_table(output_dir / "summary_rows.csv", summaries)
                    (output_dir / "summary_rows.json").write_text(json.dumps({"rows": summaries}, indent=2, sort_keys=True) + "\n")
                    log(f"DONE {case_label}", run_log)
                    del flash_case
                    torch.cuda.empty_cache()
                except Exception as exc:
                    failure = {
                        "workload": workload_name,
                        "tp_size": tp_size,
                        "ep_size": ep_size,
                        "batch_per_rank": batch,
                        "error": repr(exc),
                    }
                    failed.append(failure)
                    (output_dir / "failed_cases.json").write_text(json.dumps({"failed": failed}, indent=2, sort_keys=True) + "\n")
                    log(f"FAILED {case_label}: {exc!r}", run_log)
                    if not args.keep_going:
                        raise
    summaries = build_summary(raw_rows)
    write_table(output_dir / "raw_rows.csv", raw_rows)
    (output_dir / "raw_rows.json").write_text(json.dumps({"rows": raw_rows}, indent=2, sort_keys=True) + "\n")
    write_table(output_dir / "summary_rows.csv", summaries)
    (output_dir / "summary_rows.json").write_text(json.dumps({"rows": summaries}, indent=2, sort_keys=True) + "\n")
    if summaries:
        write_plot(output_dir, summaries)
    if failed:
        log(f"completed_with_failures={json.dumps(failed, sort_keys=True)}", run_log)
        return 2
    log("completed_full_matrix=true", run_log)
    return 0


def run_ncu_case(args: argparse.Namespace) -> int:
    import torch

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log = output_dir / "run.log"
    log(f"NCU spot case workloads={args.workloads} topologies={args.topologies} batches={args.batches}", run_log)
    fi = configure_flashinfer_repo(args.flashinfer_repo, output_dir)
    hm = configure_humming_repo(args.humming_repo)
    for workload_name in args.workloads:
        spec = WORKLOADS[workload_name]
        tp_size, ep_size = args.topologies[0]
        batch = args.batches[0]
        flash_case = make_flashinfer_case(fi, spec, tp_size, ep_size, batch, args.seed)
        fw13 = run_flashinfer_stage(fi, flash_case, spec, tp_size, ep_size, 1, 2, 3, 1, 2)
        fw2 = run_flashinfer_stage(fi, flash_case, spec, tp_size, ep_size, 2, 2, 3, 1, 2)
        humming = run_humming_case(hm, spec, tp_size, ep_size, batch, args.seed, 2, 3)
        log(
            "NCU warmup done "
            + json.dumps(
                {
                    "workload": workload_name,
                    "flashinfer_w13_profile_id": fw13["profile_id"],
                    "flashinfer_w2_profile_id": fw2["profile_id"],
                    "humming_w13_ms": statistics.median(humming["w13_times_ms"]),
                    "humming_w2_ms": statistics.median(humming["w2_times_ms"]),
                },
                sort_keys=True,
            ),
            run_log,
        )
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        _ = run_flashinfer_stage(fi, flash_case, spec, tp_size, ep_size, 1, 0, 1, 0, 1)
        _ = run_flashinfer_stage(fi, flash_case, spec, tp_size, ep_size, 2, 0, 1, 0, 1)
        _ = run_humming_case(hm, spec, tp_size, ep_size, batch, args.seed, 0, 1)
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        log(f"NCU profiled workload={workload_name}", run_log)
    return 0


def parse_topologies(values: list[str]) -> list[tuple[int, int]]:
    out = []
    for value in values:
        if ":" in value:
            a, b = value.split(":", 1)
        elif "/" in value:
            a, b = value.split("/", 1)
        else:
            raise ValueError(f"Bad topology {value!r}, expected TP:EP")
        out.append((int(a), int(b)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["run", "ncu"], default="run")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flashinfer-repo", type=Path, required=True)
    parser.add_argument("--humming-repo", type=Path, required=True)
    parser.add_argument("--workloads", nargs="+", default=list(WORKLOADS))
    parser.add_argument("--topologies", nargs="+", default=[f"{tp}:{ep}" for tp, ep in DEFAULT_TOPOLOGIES])
    parser.add_argument("--batches", nargs="+", type=int, default=DEFAULT_BATCHES)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--tactic-warmup", type=int, default=3)
    parser.add_argument("--tactic-repeat", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()
    args.topologies = parse_topologies(args.topologies)
    args.workloads = list(dict.fromkeys(w for w in args.workloads if w in WORKLOADS))
    if args.warmup < 0 or args.repeat <= 0 or args.tactic_warmup < 0 or args.tactic_repeat <= 0:
        raise SystemExit("warmup/tactic-warmup must be >= 0 and repeat/tactic-repeat must be > 0")
    if any(batch <= 0 for batch in args.batches):
        raise SystemExit("all batch sizes must be positive")
    if any(tp <= 0 or ep <= 0 for tp, ep in args.topologies):
        raise SystemExit("all topology values must be positive")
    if not args.workloads:
        raise SystemExit("No valid workloads selected")
    if args.mode == "ncu":
        return run_ncu_case(args)
    return run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
