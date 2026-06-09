#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import runpy
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")


@dataclass(frozen=True)
class Case:
    case_tag: str
    num_tokens: int
    hidden_size: int = 4096
    intermediate_size: int = 1024
    num_experts: int = 8
    top_k: int = 1


CASES = [
    Case("pr3516_full_moe_t2048", 2048),
    Case("pr3516_full_moe_t8192", 8192),
    Case("pr3516_full_moe_t16384", 16384),
]


FP4_LUT = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
SCALE_BASE = 127


COMMON_PRECISION_CONTRACT = {
    "source_activation_dtype": "torch.bfloat16",
    "activation_compute_dtype": "fp8_e4m3",
    "weight_format": "mxfp4_e2m1_group32",
    "weight_scale_dtype": "ue8m0_style_uint8_or_float8e8m0",
    "output_dtype": "torch.bfloat16",
    "routing_weight_dtype": "torch.float32",
    "top_k": 1,
}


NON_EQUIVALENT_AXES = [
    "activation_scale_policy",
    "router_semantics",
    "weight_generation",
    "weight_layout",
    "intermediate_dtype_flow",
    "timed_scope",
    "cache_policy",
    "accumulation_visibility",
]


BACKEND_PRECISION = {
    "pr3516": {
        **COMMON_PRECISION_CONTRACT,
        "activation_quantization": "input is cast internally to torch.float8_e4m3fn without an explicit external scale",
        "weight_format_detail": "packed MXFP4 E2M1 weights, group size 32, uint8 scale exponent offset 127",
        "weight_generation": "random_fp4_codes_and_uint8_scales",
        "weight_scale_layout": "natural_[E, rows, K/32]",
        "router_source": "deterministic_token_mod_expert",
        "router_weight_semantics": "all_ones",
        "intermediate_dtype": "GEMM1 output is torch.float8_e4m3fn before GEMM2",
        "accumulation_dtype": "float32",
        "alignment_status": "aligned dtype family; activation scale policy differs from PR3349/Humming",
    },
    "pr3349_source_jit": {
        **COMMON_PRECISION_CONTRACT,
        "activation_quantization": "benchmark routine uses quantize_fp8 and passes dequant scales to CUTLASS mxfp4_fp8",
        "weight_format_detail": "BF16 random weights are quantized to MXFP4 E2M1 group size 32 and interleaved for SM90",
        "weight_generation": "benchmark_bf16_random_then_mxfp4_quantized",
        "weight_scale_layout": "sm90_interleaved_int32_view",
        "router_source": "benchmark_routine_router_logits_softmax_topk",
        "router_weight_semantics": "normalized_softmax_topk_weight",
        "intermediate_dtype": "CUTLASS mxfp4_fp8 path quantizes the hidden activation for the second GEMM",
        "accumulation_dtype": "cutlass_sm90_mxfp4_fp8_kernel_internal",
        "alignment_status": "aligned dtype family; timed scope, cold-L2 policy, routing/data generation are benchmark-routine owned",
    },
    "pr3349_direct_api": {
        **COMMON_PRECISION_CONTRACT,
        "activation_quantization": "local quantize_fp8_tensor produces torch.float8_e4m3fn plus global dequant scales",
        "weight_format_detail": "BF16 random weights are quantized to MXFP4 E2M1 group size 32 and interleaved for SM90",
        "weight_generation": "local_bf16_random_then_mxfp4_quantized",
        "weight_scale_layout": "sm90_interleaved_int32_view",
        "router_source": "deterministic_token_mod_expert",
        "router_weight_semantics": "all_ones",
        "intermediate_dtype": "CUTLASS mxfp4_fp8 path quantizes the hidden activation for the second GEMM",
        "accumulation_dtype": "cutlass_sm90_mxfp4_fp8_kernel_internal",
        "alignment_status": "aligned dtype family; direct fallback routing matches PR3516/Humming but is not the nod35 source-JIT result path",
    },
    "humming": {
        **COMMON_PRECISION_CONTRACT,
        "activation_quantization": "ops.quant_input(..., float8e4m3, group_size=0) for both input and activated intermediate",
        "weight_format_detail": "HummingLayer weight_config dtype=float4e2m1, group_size=32, scale_dtype=float8e8m0",
        "weight_generation": "humming_random_fill_tensor_then_transform",
        "weight_scale_layout": "humming_internal_transformed_layout",
        "router_source": "deterministic_token_mod_expert",
        "router_weight_semantics": "all_ones",
        "intermediate_dtype": "BF16 SwiGLU activation is quantized to torch.float8_e4m3fn before w2",
        "accumulation_dtype": "float32 when use_f16_accum=False",
        "alignment_status": "aligned dtype family; fuller scripted pipeline is not the same fused surface as FlashInfer",
    },
}


def precision_row_extra(backend: str) -> dict[str, str]:
    spec = BACKEND_PRECISION[backend]
    return {
        "source_activation_dtype": spec["source_activation_dtype"],
        "activation_compute_dtype": spec["activation_compute_dtype"],
        "activation_quantization": spec["activation_quantization"],
        "weight_format": spec["weight_format"],
        "weight_scale_dtype": spec["weight_scale_dtype"],
        "weight_generation": spec["weight_generation"],
        "weight_scale_layout": spec["weight_scale_layout"],
        "router_source": spec["router_source"],
        "router_weight_semantics": spec["router_weight_semantics"],
        "intermediate_dtype": spec["intermediate_dtype"],
        "output_dtype": spec["output_dtype"],
        "accumulation_dtype": spec["accumulation_dtype"],
        "routing_weight_dtype": spec["routing_weight_dtype"],
        "strict_precision_equivalent_to_other_backends": "false",
        "non_equivalent_axes": ",".join(NON_EQUIVALENT_AXES),
        "precision_alignment_status": spec["alignment_status"],
    }


def require_precision(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"precision contract violation: {message}")


def repo_commit(path: Path) -> str:
    commit_file = path / ".commit"
    if commit_file.exists():
        return commit_file.read_text().strip()
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def moe_tflops(case: Case, time_ms: float) -> float:
    pairs = case.num_tokens * case.top_k
    flops_per_pair = 2 * case.hidden_size * (2 * case.intermediate_size)
    flops_per_pair += 2 * case.intermediate_size * case.hidden_size
    return pairs * flops_per_pair / (time_ms * 1e-3) / 1e12


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    idx = (len(xs) - 1) * pct / 100.0
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return xs[lo]
    frac = idx - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def cuda_event_bench(fn, warmup: int, repeat: int) -> list[float]:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    torch.cuda.synchronize()
    return times


def make_result(
    *,
    backend: str,
    scope: str,
    case: Case,
    times_ms: list[float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    median_ms = statistics.median(times_ms)
    row: dict[str, Any] = {
        "backend": backend,
        "scope": scope,
        **asdict(case),
        "active_token_expert_pairs": case.num_tokens * case.top_k,
        "median_time_ms": median_ms,
        "mean_time_ms": statistics.mean(times_ms),
        "std_time_ms": statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min_time_ms": min(times_ms),
        "p20_time_ms": percentile(times_ms, 20),
        "p80_time_ms": percentile(times_ms, 80),
        "max_time_ms": max(times_ms),
        "tflops": moe_tflops(case, median_ms),
        "warmup": None,
        "repeat": len(times_ms),
    }
    if extra:
        row.update(extra)
    return row


def write_rows(output_dir: Path, name: str, metadata: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}.csv"
    json_path = output_dir / f"{name}.json"
    if rows:
        fields = list(rows[0])
        for row in rows[1:]:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    json_path.write_text(json.dumps({"metadata": metadata, "rows": rows}, indent=2, sort_keys=True) + "\n")
    print(f"[DONE] csv={csv_path}", flush=True)
    print(f"[DONE] json={json_path}", flush=True)


def patch_cutlass_for_pr3516_import() -> None:
    import pkgutil
    import types
    import cutlass.cute as cute

    if not hasattr(cute.nvgpu, "OperandMajorMode") and hasattr(cute.nvgpu, "tcgen05"):
        cute.nvgpu.OperandMajorMode = cute.nvgpu.tcgen05.OperandMajorMode
    if "cutlass.cute.experimental" not in sys.modules:
        experimental = types.ModuleType("cutlass.cute.experimental")
        experimental.__path__ = []
        sys.modules["cutlass.cute.experimental"] = experimental
        setattr(cute, "experimental", experimental)
    if not getattr(pkgutil.walk_packages, "_flashinfer_pr3516_filtered", False):
        original_walk_packages = pkgutil.walk_packages

        def filtered_walk_packages(path=None, prefix="", onerror=None):
            for info in original_walk_packages(path, prefix, onerror):
                if info.name.startswith("cutlass.cute.experimental"):
                    continue
                yield info

        filtered_walk_packages._flashinfer_pr3516_filtered = True
        pkgutil.walk_packages = filtered_walk_packages


def make_mxfp4_weight(n: int, k: int, device, seed: int):
    import torch

    gen = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (n, k), generator=gen, dtype=torch.uint8)
    flat = codes.reshape(-1)
    packed = (flat[0::2] | (flat[1::2] << 4)).reshape(n, k // 2).contiguous()
    scale = torch.randint(
        SCALE_BASE - 1, SCALE_BASE + 3, (n, k // 32), generator=gen, dtype=torch.uint8
    )
    lut = torch.tensor(FP4_LUT, dtype=torch.float32)
    values = lut[codes.long()]
    w_fp32 = values * (2.0 ** (scale.to(torch.int32) - SCALE_BASE).float()).repeat_interleave(32, dim=1)
    return packed.to(device), scale.to(device), w_fp32.to(device)


def run_pr3516(args: argparse.Namespace) -> None:
    import torch

    patch_cutlass_for_pr3516_import()
    sys.path.insert(0, str(args.pr3516_repo))
    from flashinfer.fused_moe import w4a8_mxfp4_moe
    from flashinfer.fused_moe.cute_dsl import w4a8_mxfp4_grouped_gemm_sm90 as pr3516_sm90

    rows: list[dict[str, Any]] = []
    metadata = base_metadata(args, "pr3516_w4a8_mxfp4_moe")
    metadata["pr3516_commit"] = repo_commit(args.pr3516_repo)
    metadata["backend_precision"] = BACKEND_PRECISION["pr3516"]
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count

    def hw_info_fallback(cluster_shape_mn):
        if tuple(cluster_shape_mn) == (1, 1):
            return sm_count, sm_count
        return sm_count, max(1, sm_count // (cluster_shape_mn[0] * cluster_shape_mn[1]))

    pr3516_sm90._w4a8_hw_info = hw_info_fallback

    for case in CASES:
        print(f"[RUN] pr3516 {case.case_tag}", flush=True)
        torch.manual_seed(args.seed)
        fc1_p, fc1_s, fc2_p, fc2_s = [], [], [], []
        for expert_id in range(case.num_experts):
            p1, s1, _ = make_mxfp4_weight(2 * case.intermediate_size, case.hidden_size, device, args.seed + expert_id)
            p2, s2, _ = make_mxfp4_weight(case.hidden_size, case.intermediate_size, device, args.seed + 1000 + expert_id)
            fc1_p.append(p1)
            fc1_s.append(s1)
            fc2_p.append(p2)
            fc2_s.append(s2)
        fc1 = torch.stack(fc1_p)
        fc1_scale = torch.stack(fc1_s)
        fc2 = torch.stack(fc2_p)
        fc2_scale = torch.stack(fc2_s)
        x = (torch.randn(case.num_tokens, case.hidden_size, device=device) / (case.hidden_size**0.5)).to(torch.bfloat16)
        tok = torch.arange(case.num_tokens, device=device)
        selected = (tok % case.num_experts).view(-1, 1).to(torch.int32)
        weights = torch.ones(case.num_tokens, 1, device=device, dtype=torch.float32)
        output = torch.empty(case.num_tokens, case.hidden_size, device=device, dtype=torch.bfloat16)
        require_precision(x.dtype == torch.bfloat16, "PR3516 source activation must be BF16")
        require_precision(selected.dtype == torch.int32, "PR3516 selected experts must be int32")
        require_precision(weights.dtype == torch.float32, "PR3516 routing weights must be FP32")
        require_precision(output.dtype == torch.bfloat16, "PR3516 output must be BF16")
        require_precision(fc1.dtype == torch.uint8 and fc2.dtype == torch.uint8, "PR3516 packed MXFP4 weights must be uint8")
        require_precision(
            fc1_scale.dtype == torch.uint8 and fc2_scale.dtype == torch.uint8,
            "PR3516 MXFP4 scales must be uint8 exponent scales",
        )

        def run():
            return w4a8_mxfp4_moe(
                x,
                selected,
                weights,
                fc1,
                fc2,
                torch.bfloat16,
                [fc1_scale, fc2_scale],
                output=output,
            )

        out = run()
        torch.cuda.synchronize()
        if not torch.isfinite(out).all():
            raise RuntimeError(f"non-finite output for {case.case_tag}")
        times = cuda_event_bench(run, args.warmup, args.repeat)
        row = make_result(
            backend="flashinfer_pr3516_cutedsl_w4a8_mxfp4_moe",
            scope="full_moe_pr3516_body_shape_cuda_events_no_graph",
            case=case,
            times_ms=times,
            extra={**precision_row_extra("pr3516"), "warmup": args.warmup, "repeat": args.repeat},
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        del fc1, fc1_scale, fc2, fc2_scale, x, selected, weights, output, out
        torch.cuda.empty_cache()

    write_rows(args.output_dir, "pr3516_w4a8_mxfp4_moe", metadata, rows)


def run_pr3349(args: argparse.Namespace) -> None:
    if args.pr3349_repo.exists():
        rows, metadata = run_pr3349_jit(args)
        write_rows(args.output_dir, "pr3349_mxfp4_fp8", metadata, rows)
        return
    metadata = base_metadata(args, "pr3349_cutlass_mxfp4_fp8")
    metadata["backend_precision"] = BACKEND_PRECISION["pr3349_direct_api"]
    metadata["note"] = (
        "Direct API benchmark of the FlashInfer SM90 mxfp4_fp8 / WFP4A8 path "
        "added by PR3349, using the installed flashinfer package in this pod."
    )
    rows = run_pr3349_direct_api(args)
    write_rows(args.output_dir, "pr3349_mxfp4_fp8", metadata, rows)


def run_pr3349_jit(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repo = args.pr3349_repo.resolve()
    output_dir = args.output_dir.resolve()
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

    import importlib.util

    installed_spec = importlib.util.find_spec("flashinfer")
    installed_data = None
    if installed_spec and installed_spec.origin:
        installed_data = Path(installed_spec.origin).parent / "data"

    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "benchmarks"))

    from flashinfer.jit import env

    def data_or_repo(name: str, required_header: str) -> Path:
        repo_path = repo / "3rdparty" / name
        if (repo_path / required_header).exists():
            return repo_path
        if installed_data is not None and (installed_data / name / required_header).exists():
            return installed_data / name
        return repo_path

    cutlass_root = data_or_repo("cutlass", "include/cute/tensor.hpp")
    cccl_root = data_or_repo("cccl", "cub/cub/cub.cuh")
    spdlog_root = data_or_repo("spdlog", "include/spdlog/spdlog.h")
    env.FLASHINFER_AOT_DIR = Path("/tmp/flashinfer-pr3349-no-aot")
    env.FLASHINFER_INCLUDE_DIR = repo / "include"
    env.FLASHINFER_CSRC_DIR = repo / "csrc"
    env.CUTLASS_INCLUDE_DIRS = [
        cutlass_root / "include",
        cutlass_root / "tools" / "util" / "include",
    ]
    env.SPDLOG_INCLUDE_DIR = spdlog_root / "include"
    env.CCCL_INCLUDE_DIRS = [
        cccl_root / "cub",
        cccl_root / "libcudacxx" / "include",
        cccl_root / "thrust",
    ]

    cases = []
    for case in CASES:
        cases.append(
            " ".join(
                [
                    "--routine cutlass_fused_moe",
                    f"--num_tokens {case.num_tokens}",
                    f"--hidden_size {case.hidden_size}",
                    f"--intermediate_size {case.intermediate_size}",
                    f"--num_experts {case.num_experts}",
                    f"--top_k {case.top_k}",
                    "--cutlass_variant mxfp4_fp8",
                    "--input_dtype bfloat16",
                    "--no_cuda_graph",
                    f"--num_iters {args.repeat}",
                    f"--dry_run_iters {args.warmup}",
                    "--generate_repro_command",
                    "-vv",
                    f"--case_tag {case.case_tag}_pr3349",
                ]
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    testlist = output_dir / "pr3349_mxfp4_fp8_testlist.txt"
    testlist.write_text("\n".join(cases) + "\n")
    raw_csv = output_dir / "pr3349_mxfp4_fp8_raw.csv"
    sys.argv = [
        "flashinfer_benchmark.py",
        "--testlist",
        str(testlist),
        "--output_path",
        str(raw_csv),
    ]
    runpy.run_path(str(repo / "benchmarks" / "flashinfer_benchmark.py"), run_name="__main__")
    rows = normalize_pr3349_rows(raw_csv, args)
    if not rows:
        raise RuntimeError(f"PR3349 benchmark produced no rows; see raw CSV at {raw_csv}")
    metadata = base_metadata(args, "pr3349_cutlass_mxfp4_fp8_source_jit")
    metadata.update(
        {
            "pr3349_commit": repo_commit(repo),
            "raw_csv": str(raw_csv),
            "flashinfer_aot_dir": str(env.FLASHINFER_AOT_DIR),
            "cutlass_root": str(cutlass_root),
            "cccl_root": str(cccl_root),
            "spdlog_root": str(spdlog_root),
            "backend_precision": BACKEND_PRECISION["pr3349_source_jit"],
        }
    )
    return rows, metadata


def normalize_pr3349_rows(raw_csv: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    with raw_csv.open() as f:
        raw_rows = list(csv.DictReader(f))
    rows = []
    case_by_tag = {f"{case.case_tag}_pr3349": case for case in CASES}
    for raw in raw_rows:
        case = case_by_tag[raw["case_tag"]]
        median_ms = float(raw.get("median_time") or raw.get("time"))
        row = {
            "backend": "flashinfer_pr3349_cutlass_fused_moe_mxfp4_fp8",
            "scope": "full_cutlass_fused_moe_source_jit_cuda_events_no_graph",
            **asdict(case),
            "active_token_expert_pairs": case.num_tokens * case.top_k,
            "median_time_ms": median_ms,
            "mean_time_ms": raw.get("mean_time", ""),
            "std_time_ms": float(raw.get("std_time") or raw.get("std") or 0.0),
            "min_time_ms": "",
            "p20_time_ms": "",
            "p80_time_ms": "",
            "max_time_ms": "",
            "tflops": moe_tflops(case, median_ms),
            "reported_tflops": float(raw.get("tflops", "nan") or "nan"),
            "reported_tb_per_sec": float(raw.get("tb_per_sec", "nan") or "nan"),
            "warmup": args.warmup,
            "repeat": args.repeat,
            "raw_backend": raw.get("backend", ""),
            **precision_row_extra("pr3349_source_jit"),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return rows


def quantize_e2m1(x):
    import torch

    x = x.clamp(-6, 6)
    sign_bit = x < 0
    abs_x = x.abs()
    log_x = torch.floor(torch.log2(abs_x))
    log_x = torch.nan_to_num(log_x, neginf=0.0).clamp(0, 2)
    exponent_value = torch.exp2(log_x)
    mantissa_scale = 2
    mantissa = torch.round(abs_x * mantissa_scale / exponent_value)
    carry = mantissa >= mantissa_scale
    raw = (
        sign_bit * 8
        + (log_x + carry) * mantissa_scale
        + mantissa
        - carry * mantissa_scale
    ).to(torch.uint8)
    return raw[..., ::2] + raw[..., 1::2] * 16


def quantize_mxfp4_no_global_batched(tensor):
    import torch

    quantized = []
    scales = []
    for expert_id in range(tensor.shape[0]):
        expert = tensor[expert_id].contiguous()
        tiled = expert.unflatten(-1, (-1, 32))
        amax = tiled.abs().amax(dim=-1)
        log2_scale = torch.floor(torch.log2(amax)) - 2
        log2_scale = torch.nan_to_num(log2_scale, neginf=-127.0).clamp(-127, 127)
        scaled = (tiled / torch.exp2(log2_scale)[..., None]).flatten(-2, -1)
        quantized.append(quantize_e2m1(scaled))
        scales.append((log2_scale + 127).to(torch.uint8))
    return torch.stack(quantized), torch.stack(scales)


def dequant_mxfp4_on_device(w_fp4, w_scale, dtype):
    import torch

    lut = torch.tensor(FP4_LUT, dtype=torch.float32, device=w_fp4.device)
    lo = w_fp4 & 0x0F
    hi = (w_fp4 >> 4) & 0x0F
    nib = torch.stack([lo, hi], dim=-1).reshape(*w_fp4.shape[:-1], -1)
    values = lut[nib.long()]
    scale = torch.exp2(w_scale.to(torch.float32) - 127.0)
    scale = scale.repeat_interleave(32, dim=-1)
    return (values * scale).to(dtype)


def quantize_fp8_tensor(x):
    import torch

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = torch.clamp(x.float().abs().max() / fp8_max, min=1e-6)
    return (x / scale.to(x.dtype)).to(torch.float8_e4m3fn), scale.float()


def compute_gated_inter_fp8_dequant_scale_mxfp4(
    x,
    w31_fp4,
    w31_scale,
    selected_experts,
    dtype,
):
    import torch
    import torch.nn.functional as F

    active_local_experts = torch.unique(selected_experts)
    max_abs = torch.zeros((), dtype=torch.float32, device=x.device)
    for local_expert_id in active_local_experts.tolist():
        mask = selected_experts == local_expert_id
        if not mask.any():
            continue
        batch_idx, _ = torch.where(mask)
        w31_expert = dequant_mxfp4_on_device(
            w31_fp4[local_expert_id : local_expert_id + 1],
            w31_scale[local_expert_id : local_expert_id + 1],
            dtype,
        )[0]
        w3_expert, w1_expert = torch.chunk(w31_expert, 2, dim=0)
        expert_inputs = x[batch_idx]
        gate = expert_inputs @ w1_expert.t()
        up = expert_inputs @ w3_expert.t()
        inter = F.silu(gate) * up
        max_abs = torch.maximum(max_abs, inter.float().abs().max())

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    return torch.clamp(max_abs / fp8_max, min=1e-6).float()


def run_pr3349_direct_api(args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch
    from flashinfer.fused_moe import (
        ActivationType,
        cutlass_fused_moe,
        interleave_moe_scales_for_sm90_mixed_gemm,
        interleave_moe_weights_for_sm90_mixed_gemm,
    )

    rows: list[dict[str, Any]] = []
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.set_grad_enabled(False)

    for case in CASES:
        print(f"[RUN] pr3349 {case.case_tag}", flush=True)
        torch.manual_seed(args.seed)
        x = torch.randn(case.num_tokens, case.hidden_size, dtype=dtype, device=device)
        w31_local = (
            torch.randn(
                case.num_experts,
                2 * case.intermediate_size,
                case.hidden_size,
                dtype=dtype,
                device=device,
            )
            / 10
        )
        w2_local = (
            torch.randn(
                case.num_experts,
                case.hidden_size,
                case.intermediate_size,
                dtype=dtype,
                device=device,
            )
            / 10
        )
        tok = torch.arange(case.num_tokens, device=device)
        selected = (tok % case.num_experts).view(-1, 1).to(torch.int32)
        routing_weights = torch.ones(case.num_tokens, 1, device=device, dtype=torch.float32)

        w31_mxfp4, w31_mxfp4_scale = quantize_mxfp4_no_global_batched(w31_local)
        w2_mxfp4, w2_mxfp4_scale = quantize_mxfp4_no_global_batched(w2_local)
        w31_mxfp4_il = interleave_moe_weights_for_sm90_mixed_gemm(
            w31_mxfp4.contiguous().view(torch.uint8), "int4"
        )
        w2_mxfp4_il = interleave_moe_weights_for_sm90_mixed_gemm(
            w2_mxfp4.contiguous().view(torch.uint8), "int4"
        )
        w31_mxfp4_scale_il = interleave_moe_scales_for_sm90_mixed_gemm(w31_mxfp4_scale)
        w2_mxfp4_scale_il = interleave_moe_scales_for_sm90_mixed_gemm(w2_mxfp4_scale)

        x_quant, fc1_dequant_scale = quantize_fp8_tensor(x)
        require_precision(x.dtype == torch.bfloat16, "PR3349 direct source activation must be BF16")
        require_precision(x_quant.dtype == torch.float8_e4m3fn, "PR3349 direct activation compute tensor must be FP8 E4M3")
        require_precision(selected.dtype == torch.int32, "PR3349 direct selected experts must be int32")
        require_precision(routing_weights.dtype == torch.float32, "PR3349 direct routing weights must be FP32")
        require_precision(
            w31_mxfp4.dtype == torch.uint8 and w2_mxfp4.dtype == torch.uint8,
            "PR3349 direct MXFP4 weights must be uint8-packed",
        )
        require_precision(
            w31_mxfp4_scale.dtype == torch.uint8 and w2_mxfp4_scale.dtype == torch.uint8,
            "PR3349 direct MXFP4 scales must be uint8 exponent scales",
        )
        x_dequant = (x_quant.to(dtype) * fc1_dequant_scale.to(dtype)).contiguous()
        fc1_global = fc1_dequant_scale.float().repeat(case.num_experts)
        fc2_dequant_scale = compute_gated_inter_fp8_dequant_scale_mxfp4(
            x_dequant, w31_mxfp4, w31_mxfp4_scale, selected, dtype
        )
        fc2_quant = (1.0 / fc2_dequant_scale).reshape(())
        fc2_global = fc2_dequant_scale.repeat(case.num_experts)
        quant_scales = [
            w31_mxfp4_scale_il.view(torch.int32),
            fc1_global,
            fc2_quant,
            w2_mxfp4_scale_il.view(torch.int32),
            fc2_global,
        ]
        require_precision(len(quant_scales) == 5, "PR3349 direct quant_scales must have five entries")
        require_precision(quant_scales[0].dtype == torch.int32, "PR3349 direct fc1 weight scales must be int32 view")
        require_precision(quant_scales[1].dtype == torch.float32, "PR3349 direct fc1 activation scale must be FP32")
        require_precision(quant_scales[2].dtype == torch.float32, "PR3349 direct fc2 quant scale must be FP32")
        require_precision(quant_scales[3].dtype == torch.int32, "PR3349 direct fc2 weight scales must be int32 view")
        require_precision(quant_scales[4].dtype == torch.float32, "PR3349 direct fc2 activation scale must be FP32")
        output = torch.empty_like(x)
        require_precision(output.dtype == torch.bfloat16, "PR3349 direct output must be BF16")

        def run():
            return cutlass_fused_moe(
                x_quant,
                selected,
                routing_weights,
                w31_mxfp4_il.contiguous().view(torch.long),
                w2_mxfp4_il.contiguous().view(torch.long),
                dtype,
                quant_scales=quant_scales,
                output=output,
                activation_type=ActivationType.Swiglu,
            )

        out = run()
        torch.cuda.synchronize()
        if isinstance(out, list):
            out_tensor = out[0]
        else:
            out_tensor = out
        if not torch.isfinite(out_tensor).all():
            raise RuntimeError(f"non-finite output for {case.case_tag}")
        times = cuda_event_bench(run, args.warmup, args.repeat)
        row = make_result(
            backend="flashinfer_pr3349_cutlass_fused_moe_mxfp4_fp8",
            scope="full_cutlass_fused_moe_direct_api_cuda_events_no_graph",
            case=case,
            times_ms=times,
            extra={**precision_row_extra("pr3349_direct_api"), "warmup": args.warmup, "repeat": args.repeat},
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        del x, w31_local, w2_local, w31_mxfp4, w2_mxfp4, w31_mxfp4_il, w2_mxfp4_il
        del w31_mxfp4_scale, w2_mxfp4_scale, w31_mxfp4_scale_il, w2_mxfp4_scale_il
        del x_quant, x_dequant, selected, routing_weights, output, out_tensor
        torch.cuda.empty_cache()
    return rows


def choose_indexed_block_size(tuning_config, shape_m: int, top_k: int) -> int:
    routed_shape_m = shape_m * top_k
    for min_shape_m, max_shape_m, config in tuning_config:
        if routed_shape_m > min_shape_m and routed_shape_m <= max_shape_m:
            return int(config["block_shape"][0])
    raise RuntimeError(f"No Humming indexed block config for routed_shape_m={routed_shape_m}")


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
        pad_size = padded_size - part_token_ids.size(0)
        part_token_ids = torch.nn.functional.pad(part_token_ids, pad=(0, pad_size), value=flat.nelement())
        part_token_ids_list.append(part_token_ids)
        expert_id_list += [expert_id] * num_blocks
    sorted_ids = torch.cat(part_token_ids_list).to(torch.int32)
    expert_ids = torch.tensor(expert_id_list, dtype=torch.int32, device=flat.device)
    num_tokens_padded = torch.tensor(sorted_ids.size(0), dtype=torch.int32, device=flat.device)
    return sorted_ids, expert_ids, num_tokens_padded


def make_humming_layer(shape_n: int, shape_k: int, num_experts: int):
    import torch
    from humming.config import GemmType
    from humming.layer import HummingLayer
    from humming.tune import get_heuristics_config
    from humming.utils.test import random_fill_tensor

    layer = HummingLayer(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=num_experts,
        weight_config={"dtype": "float4e2m1", "group_size": 32, "scale_dtype": "float8e8m0"},
        input_config={"dtype": "float8e4m3"},
        torch_dtype=torch.bfloat16,
    ).to("cuda:0")
    for tensor in layer.parameters():
        random_fill_tensor(tensor)
    layer.transform()
    meta = layer.humming_metas[""]
    tuning_config = get_heuristics_config(meta=meta, gemm_type=GemmType.INDEXED)
    compute_config = {"use_f16_accum": False, "gemm_type": GemmType.INDEXED.value}
    return layer, compute_config, tuning_config


def run_humming(args: argparse.Namespace) -> None:
    import torch

    sys.path.insert(0, str(args.humming_repo))
    from humming import ops

    rows: list[dict[str, Any]] = []
    metadata = base_metadata(args, "humming_main_fuller_pipeline")
    metadata["humming_commit"] = repo_commit(args.humming_repo)
    metadata["backend_precision"] = BACKEND_PRECISION["humming"]
    torch.set_grad_enabled(False)
    device = torch.device("cuda")

    for case in CASES:
        print(f"[RUN] humming {case.case_tag}", flush=True)
        torch.manual_seed(args.seed)
        routed_rows = case.num_tokens * case.top_k
        w13_layer, w13_compute, w13_tuning = make_humming_layer(
            2 * case.intermediate_size, case.hidden_size, case.num_experts
        )
        w2_layer, w2_compute, w2_tuning = make_humming_layer(
            case.hidden_size, case.intermediate_size, case.num_experts
        )
        require_precision(
            w13_compute.get("use_f16_accum") is False and w2_compute.get("use_f16_accum") is False,
            "Humming must use float32 accumulation by setting use_f16_accum=False",
        )
        tok = torch.arange(case.num_tokens, device=device)
        topk_ids = (tok % case.num_experts).view(-1, 1).to(torch.int32).contiguous()
        topk_weights = torch.ones(case.num_tokens, 1, dtype=torch.float32, device=device)
        w13_block = choose_indexed_block_size(w13_tuning, case.num_tokens, case.top_k)
        w13_sorted, w13_expert_ids, w13_padded = indexed_moe_tensors_from_topk_ids(
            topk_ids, case.num_experts, w13_block
        )
        w2_block = choose_indexed_block_size(w2_tuning, routed_rows, 1)
        w2_sorted, w2_expert_ids, w2_padded = indexed_moe_tensors_from_topk_ids(
            topk_ids.reshape(-1, 1), case.num_experts, w2_block
        )

        x = torch.randn(case.num_tokens, case.hidden_size, dtype=torch.bfloat16, device=device)
        x_fp8, x_scale = ops.quant_input(x, "float8e4m3", group_size=0)
        w13_out = torch.empty((routed_rows, 2 * case.intermediate_size), dtype=torch.bfloat16, device=device)
        sigmoid_buf = torch.empty((routed_rows, case.intermediate_size), dtype=torch.bfloat16, device=device)
        activated = torch.empty((routed_rows, case.intermediate_size), dtype=torch.bfloat16, device=device)
        inter_fp8 = torch.empty((routed_rows, case.intermediate_size), dtype=torch.float8_e4m3fn, device=device)
        w2_out = torch.empty((routed_rows, case.hidden_size), dtype=torch.bfloat16, device=device)
        combined = torch.empty((case.num_tokens, case.hidden_size), dtype=torch.bfloat16, device=device)
        require_precision(x.dtype == torch.bfloat16, "Humming source activation must be BF16")
        require_precision(x_fp8.dtype == torch.float8_e4m3fn, "Humming input compute tensor must be FP8 E4M3")
        require_precision(topk_ids.dtype == torch.int32, "Humming topk ids must be int32")
        require_precision(topk_weights.dtype == torch.float32, "Humming routing weights must be FP32")
        require_precision(inter_fp8.dtype == torch.float8_e4m3fn, "Humming intermediate compute tensor must be FP8 E4M3")
        require_precision(combined.dtype == torch.bfloat16, "Humming combined output must be BF16")

        def run_w13():
            return w13_layer(
                inputs=x_fp8,
                input_scale=x_scale,
                outputs=w13_out,
                sorted_ids=w13_sorted,
                expert_ids=w13_expert_ids,
                num_tokens_padded=w13_padded,
                top_k=case.top_k,
                compute_config=w13_compute,
                tuning_config=w13_tuning,
            )

        run_w13()
        up, gate = w13_out.chunk(2, dim=-1)
        torch.sigmoid(gate, out=sigmoid_buf)
        torch.mul(gate, sigmoid_buf, out=activated)
        activated.mul_(up)
        _, inter_scale = ops.quant_input(activated, "float8e4m3", group_size=0)
        torch.cuda.synchronize()

        def run_activation_quant():
            up_local, gate_local = w13_out.chunk(2, dim=-1)
            torch.sigmoid(gate_local, out=sigmoid_buf)
            torch.mul(gate_local, sigmoid_buf, out=activated)
            activated.mul_(up_local)
            ops.quant_input(
                activated,
                "float8e4m3",
                scales=inter_scale,
                outputs=inter_fp8,
                group_size=0,
            )
            return inter_fp8

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

        def run_combine():
            return ops.moe_fused_mul_sum(w2_out.view(case.num_tokens, case.top_k, case.hidden_size), topk_weights, outputs=combined)

        def run_full():
            run_w13()
            run_activation_quant()
            run_w2()
            run_combine()
            return combined

        out = run_full()
        torch.cuda.synchronize()
        if not torch.isfinite(out).all():
            raise RuntimeError(f"non-finite output for {case.case_tag}")
        times = cuda_event_bench(run_full, args.warmup, args.repeat)
        row = make_result(
            backend="inclusionai_humming_main_indexed_fuller_pipeline",
            scope="w13_torch_swiglu_fixed_fp8_quant_w2_triton_combine_cuda_events_no_graph",
            case=case,
            times_ms=times,
            extra={**precision_row_extra("humming"), "warmup": args.warmup, "repeat": args.repeat},
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        del w13_layer, w2_layer, x, x_fp8, x_scale, w13_out, sigmoid_buf, activated, inter_fp8, w2_out, combined, out
        torch.cuda.empty_cache()

    write_rows(args.output_dir, "humming_main_fuller_pipeline", metadata, rows)


def base_metadata(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    import torch

    return {
        "backend": backend,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "seed": args.seed,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "shape_standard": "PR3516 body shape: E=8 top_k=1 H=4096 I=1024 tokens=2048/8192/16384; PR3349 uses cutlass_variant=mxfp4_fp8",
        "precision_common": COMMON_PRECISION_CONTRACT,
        "strict_precision_equivalent": False,
        "non_equivalent_axes": NON_EQUIVALENT_AXES,
    }


def aggregate(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    for name in [
        "pr3516_w4a8_mxfp4_moe.csv",
        "pr3349_mxfp4_fp8.csv",
        "humming_main_fuller_pipeline.csv",
    ]:
        path = args.output_dir / name
        if not path.exists():
            continue
        with path.open() as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No backend CSVs found in {args.output_dir}")

    baseline: dict[tuple[int, int, int, int, int], float] = {}
    for row in rows:
        if row["backend"] == "flashinfer_pr3516_cutedsl_w4a8_mxfp4_moe":
            key = (
                int(row["num_tokens"]),
                int(row["hidden_size"]),
                int(row["intermediate_size"]),
                int(row["num_experts"]),
                int(row["top_k"]),
            )
            baseline[key] = float(row["median_time_ms"])
    for row in rows:
        key = (
            int(row["num_tokens"]),
            int(row["hidden_size"]),
            int(row["intermediate_size"]),
            int(row["num_experts"]),
            int(row["top_k"]),
        )
        base = baseline.get(key)
        if base:
            current = float(row["median_time_ms"])
            row["latency_vs_pr3516"] = current / base
            row["speedup_vs_pr3516"] = base / current
        else:
            row["latency_vs_pr3516"] = ""
            row["speedup_vs_pr3516"] = ""

    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    out_csv = args.output_dir / "summary_compare.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    out_json = args.output_dir / "summary_compare.json"
    out_json.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n")
    print(f"[DONE] csv={out_csv}", flush=True)
    print(f"[DONE] json={out_json}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["pr3516", "pr3349", "humming", "aggregate"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pr3516-repo", type=Path, default=Path("/tmp/flashinfer-pr3516-live"))
    parser.add_argument("--pr3349-repo", type=Path, default=Path("/tmp/flashinfer-wfp4afp8-ablation-work"))
    parser.add_argument("--humming-repo", type=Path, default=Path("/tmp/humming-main-live"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.backend == "pr3516":
        run_pr3516(args)
    elif args.backend == "pr3349":
        run_pr3349(args)
    elif args.backend == "humming":
        run_humming(args)
    elif args.backend == "aggregate":
        aggregate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
