"""Run the MXFP4 scale-layout acceptance plan on a Modal B200.

The runner clones both revisions at execution time. It does not upload a local
FlashInfer checkout, so every result records reproducible repository SHAs.

Prerequisites:

    python3 -m pip install modal
    python3 -m modal setup

Examples:

    # Validate the Modal image, GPU, revision ancestry, and feature import.
    modal run flashinfer/quant/modal_mxfp4_layout_validation.py --task smoke

    # Run the complete test file once on target and once on feature.
    modal run flashinfer/quant/modal_mxfp4_layout_validation.py --task tests

    # Run the public API matrix plus CUDA Graph and PDL checks.
    modal run flashinfer/quant/modal_mxfp4_layout_validation.py --task functional

    # Run five-round target/feature A/B and the feature layout sweep.
    modal run flashinfer/quant/modal_mxfp4_layout_validation.py --task performance

    # Run the full acceptance plan.
    modal run flashinfer/quant/modal_mxfp4_layout_validation.py --task all

Results and JIT caches are persisted in the Modal Volume printed by the local
entrypoint. Set MODAL_GPU, MODAL_VOLUME_NAME, or MODAL_BASE_IMAGE to override
the corresponding defaults before invoking ``modal run``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable, Sequence

import modal

APP_NAME = "flashinfer-mxfp4-layout-validation"
DEFAULT_FEATURE_REPO = "https://github.com/huangzhilin-hzl/flashinfer.git"
DEFAULT_FEATURE_REF = "molou/mxfp4-quantize-layout"
DEFAULT_TARGET_REPO = "https://github.com/flashinfer-ai/flashinfer.git"
DEFAULT_TARGET_REF = "main"
DEFAULT_BASE_IMAGE = "flashinfer/flashinfer-ci-cu130"
DEFAULT_VOLUME_NAME = "flashinfer-mxfp4-layout-validation"
SUPPORTED_TASKS = {"smoke", "tests", "functional", "performance", "all"}

REMOTE_PYTHON = "python"
WORK_ROOT = Path("/workspace/flashinfer-mxfp4-layout")
TARGET_TREE = WORK_ROOT / "target"
FEATURE_TREE = WORK_ROOT / "feature"
NEUTRAL_CWD = WORK_ROOT / "runner"
VOLUME_ROOT = Path("/cache")
RESULTS_ROOT = VOLUME_ROOT / "results"
JIT_ROOT = VOLUME_ROOT / "jit"

PERF_SHAPES = ((128, 4096), (2048, 8192), (8192, 16384))
PERF_DTYPES = ("bfloat16", "float16")
PERF_BACKENDS = ("cuda", "cute-dsl")
LAYOUTS = ("128x4", "8x4", "linear")


def _arg_value(flag: str) -> str | None:
    for index, argument in enumerate(sys.argv):
        if argument == flag and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        prefix = f"{flag}="
        if argument.startswith(prefix):
            return argument[len(prefix) :]
    return None


GPU_TYPE = _arg_value("--gpu") or os.environ.get("MODAL_GPU", "B200")
VOLUME_NAME = _arg_value("--volume-name") or os.environ.get(
    "MODAL_VOLUME_NAME", DEFAULT_VOLUME_NAME
)
BASE_IMAGE = os.environ.get("MODAL_BASE_IMAGE", DEFAULT_BASE_IMAGE)

app = modal.App(APP_NAME)
result_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(BASE_IMAGE, add_python=None)
    .env(
        {
            "FLASHINFER_CUDA_ARCH_LIST": "10.0a",
            "TORCH_CUDA_ARCH_LIST": "10.0a",
            "MAX_JOBS": os.environ.get("MAX_JOBS", "16"),
            "PIP_BREAK_SYSTEM_PACKAGES": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "MPLBACKEND": "Agg",
        }
    )
    .run_commands(
        "python -m pip install 'pytest>=8,<9' 'setuptools>=77,<82' 'wheel>=0.45'"
    )
)


class CommandFailure(RuntimeError):
    """Raised when a logged subprocess exits unsuccessfully."""


def _merged_env(overrides: dict[str, str] | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    if overrides:
        environment.update(overrides)
    return environment


def _run_logged(
    command: Sequence[str],
    log_path: Path,
    *,
    cwd: Path = NEUTRAL_CWD,
    env: dict[str, str] | None = None,
    check: bool = True,
    append: bool = False,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cwd.mkdir(parents=True, exist_ok=True)
    rendered = shlex.join([str(part) for part in command])
    print(f"$ {rendered}", flush=True)

    mode = "a" if append else "w"
    with log_path.open(mode, encoding="utf-8") as log_file:
        log_file.write(f"$ {rendered}\n")
        log_file.flush()
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=str(cwd),
            env=_merged_env(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()

    if check and return_code != 0:
        raise CommandFailure(f"Command exited with code {return_code}; see {log_path}")
    return return_code


def _capture(
    command: Sequence[str],
    *,
    cwd: Path = NEUTRAL_CWD,
    env: dict[str, str] | None = None,
) -> str:
    cwd.mkdir(parents=True, exist_ok=True)
    print(f"$ {shlex.join([str(part) for part in command])}", flush=True)
    completed = subprocess.run(
        [str(part) for part in command],
        cwd=str(cwd),
        env=_merged_env(env),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = completed.stdout.strip()
    if output:
        print(output, flush=True)
    return output


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _safe_run_id(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    if not sanitized:
        raise ValueError("run_name must contain at least one safe filename character")
    return sanitized


def _revision_env(label: str, sha: str) -> dict[str, str]:
    cache_suffix = f"{label}-{sha[:12]}"
    workspace_cache = JIT_ROOT / "flashinfer" / cache_suffix
    torch_cache = JIT_ROOT / "torch-extensions" / cache_suffix
    workspace_cache.mkdir(parents=True, exist_ok=True)
    torch_cache.mkdir(parents=True, exist_ok=True)
    return {
        "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        "FLASHINFER_WORKSPACE_BASE": str(workspace_cache),
        "TORCH_EXTENSIONS_DIR": str(torch_cache),
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": "",
    }


def _clone_at(repo: str, ref: str, destination: Path, log_path: Path) -> str:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    commands = (
        ["git", "init", str(destination)],
        ["git", "-C", str(destination), "remote", "add", "origin", repo],
        [
            "git",
            "-C",
            str(destination),
            "fetch",
            "--force",
            "--no-tags",
            "origin",
            ref,
        ],
        [
            "git",
            "-C",
            str(destination),
            "checkout",
            "--detach",
            "FETCH_HEAD",
        ],
    )
    for index, command in enumerate(commands):
        _run_logged(command, log_path, append=index > 0)
    return _capture(["git", "-C", str(destination), "rev-parse", "HEAD"])


def _update_submodules(tree: Path, log_path: Path) -> None:
    _run_logged(
        [
            "git",
            "-C",
            str(tree),
            "submodule",
            "update",
            "--init",
            "--recursive",
            "--jobs",
            "8",
        ],
        log_path,
    )


def _collect_environment(result_dir: Path) -> dict[str, Any]:
    nvidia_smi = _capture(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    probe = _capture(
        [
            REMOTE_PYTHON,
            "-c",
            (
                "import json, torch; "
                "cc=torch.cuda.get_device_capability(); "
                "cuda=tuple(int(x) for x in torch.version.cuda.split('.')[:2]); "
                "assert cc == (10, 0), f'expected SM100, got {cc}'; "
                "assert cuda >= (12, 8), f'expected CUDA >= 12.8, got {cuda}'; "
                "print(json.dumps({'torch': torch.__version__, "
                "'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(), "
                "'compute_capability': cc}))"
            ),
        ]
    )
    metadata = json.loads(probe.splitlines()[-1])
    metadata["nvidia_smi"] = nvidia_smi
    metadata["modal_gpu_request"] = GPU_TYPE
    metadata["base_image"] = BASE_IMAGE
    _write_json(result_dir / "environment.json", metadata)
    return metadata


def _prepare_repositories(
    result_dir: Path,
    *,
    feature_repo: str,
    feature_ref: str,
    target_repo: str,
    target_ref: str,
    allow_stale_feature: bool,
) -> dict[str, Any]:
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    NEUTRAL_CWD.mkdir(parents=True)
    logs = result_dir / "logs" / "prepare"

    target_sha = _clone_at(target_repo, target_ref, TARGET_TREE, logs / "target.log")
    feature_sha = _clone_at(
        feature_repo, feature_ref, FEATURE_TREE, logs / "feature.log"
    )

    target_fetch_log = logs / "feature_target_fetch.log"
    _run_logged(
        [
            "git",
            "-C",
            str(FEATURE_TREE),
            "remote",
            "add",
            "target",
            target_repo,
        ],
        target_fetch_log,
    )
    _run_logged(
        [
            "git",
            "-C",
            str(FEATURE_TREE),
            "fetch",
            "--force",
            "--no-tags",
            "target",
            target_ref,
        ],
        target_fetch_log,
        append=True,
    )
    fetched_target_sha = _capture(
        ["git", "-C", str(FEATURE_TREE), "rev-parse", "FETCH_HEAD"]
    )
    if fetched_target_sha != target_sha:
        raise RuntimeError(
            "Target ref changed while repositories were prepared: "
            f"{target_sha} != {fetched_target_sha}"
        )

    merge_base_sha = _capture(
        [
            "git",
            "-C",
            str(FEATURE_TREE),
            "merge-base",
            feature_sha,
            target_sha,
        ]
    )
    target_is_ancestor = merge_base_sha == target_sha

    refs = {
        "feature_repo": feature_repo,
        "feature_ref": feature_ref,
        "feature_sha": feature_sha,
        "feature_commit_date": _capture(
            [
                "git",
                "-C",
                str(FEATURE_TREE),
                "show",
                "-s",
                "--format=%cI",
                feature_sha,
            ]
        ),
        "target_repo": target_repo,
        "target_ref": target_ref,
        "target_sha": target_sha,
        "target_commit_date": _capture(
            [
                "git",
                "-C",
                str(TARGET_TREE),
                "show",
                "-s",
                "--format=%cI",
                target_sha,
            ]
        ),
        "merge_base_sha": merge_base_sha,
        "target_is_ancestor": target_is_ancestor,
    }
    _write_json(result_dir / "revisions.json", refs)

    if not target_is_ancestor and not allow_stale_feature:
        raise RuntimeError(
            "Feature does not contain the current target revision. "
            f"TARGET_SHA={target_sha}, FEATURE_SHA={feature_sha}, "
            f"MERGE_BASE_SHA={merge_base_sha}. Rebase the feature branch or "
            "use --allow-stale-feature for a diagnostic, non-mergeable run."
        )

    _update_submodules(TARGET_TREE, logs / "target_submodules.log")
    _update_submodules(FEATURE_TREE, logs / "feature_submodules.log")
    _collect_environment(result_dir)

    runner = TARGET_TREE / "benchmarks" / "flashinfer_benchmark.py"
    refs["fixed_benchmark_runner"] = str(runner)
    refs["fixed_benchmark_runner_sha256"] = hashlib.sha256(
        runner.read_bytes()
    ).hexdigest()
    _write_json(result_dir / "revisions.json", refs)
    return refs


def _install_revision(
    tree: Path,
    label: str,
    sha: str,
    result_dir: Path,
    *,
    log_suffix: str = "",
) -> dict[str, str]:
    suffix = f"_{log_suffix}" if log_suffix else ""
    log_path = result_dir / "logs" / "install" / f"{label}{suffix}.log"
    environment = _revision_env(label, sha)

    _run_logged(
        [
            REMOTE_PYTHON,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "flashinfer-python",
            "flashinfer",
        ],
        log_path,
        env=environment,
        check=False,
    )
    _run_logged(
        [
            REMOTE_PYTHON,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "-e",
            str(tree),
            "-v",
        ],
        log_path,
        env=environment,
        append=True,
    )

    verification = (
        "from pathlib import Path; "
        "import flashinfer, torch; "
        "from flashinfer.cute_dsl import is_cute_dsl_available; "
        f"expected=Path({str(tree)!r}).resolve(); "
        "loaded=Path(flashinfer.__file__).resolve(); "
        "cc=torch.cuda.get_device_capability(); "
        "assert expected in loaded.parents, (expected, loaded); "
        "assert cc == (10, 0), cc; "
        "assert is_cute_dsl_available(), 'CuTe-DSL is unavailable'; "
        "print(f'flashinfer={loaded}'); "
        "print(f'gpu={torch.cuda.get_device_name()} cc={cc} cuda={torch.version.cuda}'); "
        "print(f'cute_dsl={is_cute_dsl_available()}')"
    )
    _run_logged(
        [REMOTE_PYTHON, "-c", verification],
        log_path,
        env=environment,
        append=True,
    )
    return environment


def _parse_junit(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    cases: dict[str, dict[str, str]] = {}
    counts = defaultdict(int)
    for test_case in root.iter("testcase"):
        classname = test_case.attrib.get("classname", "")
        name = test_case.attrib.get("name", "")
        node_id = f"{classname}::{name}" if classname else name
        status = "passed"
        message = ""
        for candidate in ("failure", "error", "skipped"):
            child = test_case.find(candidate)
            if child is not None:
                status = candidate
                message = child.attrib.get("message", "") or (child.text or "")
                break
        counts[status] += 1
        cases[node_id] = {"status": status, "message": message[:4000]}
    summary = {
        "total": len(cases),
        "passed": counts["passed"],
        "failed": counts["failure"],
        "errors": counts["error"],
        "skipped": counts["skipped"],
    }
    return {"summary": summary, "cases": cases}


def _compare_pytest_results(
    baseline: dict[str, Any], feature: dict[str, Any]
) -> dict[str, Any]:
    baseline_cases = baseline["cases"]
    feature_cases = feature["cases"]
    missing = sorted(set(baseline_cases) - set(feature_cases))
    regressed = sorted(
        node_id
        for node_id, baseline_case in baseline_cases.items()
        if baseline_case["status"] == "passed"
        and node_id in feature_cases
        and feature_cases[node_id]["status"] != "passed"
    )
    new_skips = sorted(
        node_id
        for node_id, feature_case in feature_cases.items()
        if feature_case["status"] == "skipped"
        and (
            node_id not in baseline_cases
            or baseline_cases[node_id]["status"] != "skipped"
        )
    )
    added = sorted(set(feature_cases) - set(baseline_cases))
    return {
        "baseline": baseline["summary"],
        "feature": feature["summary"],
        "missing_baseline_cases": missing,
        "regressed_baseline_cases": regressed,
        "new_skips": new_skips,
        "feature_added_cases": added,
    }


def _run_full_pytest(result_dir: Path, refs: dict[str, Any]) -> dict[str, Any]:
    revisions = (
        ("baseline", TARGET_TREE, refs["target_sha"]),
        ("feature", FEATURE_TREE, refs["feature_sha"]),
    )
    parsed: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    for label, tree, sha in revisions:
        try:
            environment = _install_revision(tree, label, sha, result_dir)
            xml_path = result_dir / "pytest" / f"{label}_fp4.xml"
            log_path = result_dir / "logs" / "pytest" / f"{label}_fp4.log"
            xml_path.parent.mkdir(parents=True, exist_ok=True)
            return_code = _run_logged(
                [
                    REMOTE_PYTHON,
                    "-m",
                    "pytest",
                    "-vv",
                    "-rs",
                    "tests/utils/test_fp4_quantize.py",
                    "--junitxml",
                    str(xml_path),
                ],
                log_path,
                cwd=tree,
                env=environment,
                check=False,
            )
            if not xml_path.exists():
                errors.append(f"{label}: pytest did not produce {xml_path}")
                continue
            parsed[label] = _parse_junit(xml_path)
            parsed[label]["return_code"] = return_code
            _write_json(result_dir / "pytest" / f"{label}_summary.json", parsed[label])
            if return_code != 0:
                errors.append(f"{label}: pytest exited with code {return_code}")
        except Exception as exception:  # Continue so both revisions are attempted.
            errors.append(f"{label}: {type(exception).__name__}: {exception}")

    comparison: dict[str, Any] = {"errors": errors}
    if "baseline" in parsed and "feature" in parsed:
        comparison.update(
            _compare_pytest_results(parsed["baseline"], parsed["feature"])
        )
        if comparison["missing_baseline_cases"]:
            errors.append("feature is missing baseline test cases")
        if comparison["regressed_baseline_cases"]:
            errors.append("baseline test cases regressed on feature")
        if comparison["new_skips"]:
            errors.append("feature introduced new skipped test cases")
        if (
            parsed["feature"]["summary"]["total"]
            < parsed["baseline"]["summary"]["total"]
        ):
            errors.append("feature collected fewer tests than baseline")
    comparison["errors"] = errors
    _write_json(result_dir / "pytest" / "comparison.json", comparison)
    if errors:
        raise RuntimeError("; ".join(errors))
    return comparison


FUNCTIONAL_MATRIX_SCRIPT = r"""#!/usr/bin/env python3
import argparse
import json
import math
import traceback

import torch
import torch.nn.functional as F
from flashinfer import SfLayout, mxfp4_dequantize, mxfp4_quantize


CASES = ((9, 96), (129, 160), (2048, 8192))
DTYPES = (torch.bfloat16, torch.float16)
LAYOUTS = {
    "128x4": SfLayout.layout_128x4,
    "8x4": SfLayout.layout_8x4,
    "linear": SfLayout.layout_linear,
}


def expected_scale_shape(m, k, layout_name):
    groups = k // 32
    if layout_name == "linear":
        return (m, groups)
    row_tile = 8 if layout_name == "8x4" else 128
    return (math.ceil(m / row_tile) * row_tile, math.ceil(groups / 4) * 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    assert torch.cuda.get_device_capability() == (10, 0)
    failures = []
    with open(args.output, "w", encoding="utf-8") as output:
        for m, k in CASES:
            for dtype in DTYPES:
                for layout_name, layout in LAYOUTS.items():
                    row = {
                        "m": m,
                        "k": k,
                        "dtype": str(dtype),
                        "layout": layout_name,
                    }
                    try:
                        torch.manual_seed(42)
                        x = torch.randn((m, k), dtype=dtype, device="cuda")
                        x_cpu = x.cpu().float()
                        backend_outputs = {}
                        for backend in ("cuda", "cute-dsl"):
                            quantized, scales = mxfp4_quantize(
                                x,
                                backend=backend,
                                enable_pdl=False,
                                sfLayout=layout,
                            )
                            torch.cuda.synchronize()
                            assert tuple(quantized.shape) == (m, k // 2)
                            assert quantized.dtype == torch.uint8
                            assert tuple(scales.shape) == expected_scale_shape(
                                m, k, layout_name
                            )
                            assert scales.dtype == torch.uint8
                            assert torch.count_nonzero(quantized).item() > 0
                            assert torch.count_nonzero(scales).item() > 0

                            dequantized = mxfp4_dequantize(
                                quantized, scales, sfLayout=layout
                            )
                            assert tuple(dequantized.shape) == (m, k)
                            assert dequantized.dtype == torch.float32
                            assert torch.isfinite(dequantized).all()
                            torch.testing.assert_close(
                                dequantized,
                                x_cpu,
                                rtol=0.5,
                                atol=0.5,
                            )
                            backend_outputs[backend] = (
                                quantized,
                                scales,
                                dequantized,
                            )

                        quant_cuda, scale_cuda, dq_cuda = backend_outputs["cuda"]
                        quant_cute, scale_cute, dq_cute = backend_outputs["cute-dsl"]
                        quant_match = (
                            (quant_cuda == quant_cute).float().mean().item() * 100
                        )
                        scale_match = (
                            (scale_cuda == scale_cute).float().mean().item() * 100
                        )
                        cosine = F.cosine_similarity(
                            dq_cuda.reshape(-1), dq_cute.reshape(-1), dim=0
                        ).item()
                        assert quant_match > 95.0, quant_match
                        assert scale_match > 95.0, scale_match
                        assert cosine >= 0.9, cosine
                        row.update(
                            {
                                "status": "passed",
                                "quant_shape": list(quant_cuda.shape),
                                "scale_shape": list(scale_cuda.shape),
                                "quant_match_pct": quant_match,
                                "scale_match_pct": scale_match,
                                "dequant_cosine_similarity": cosine,
                            }
                        )
                        print(json.dumps(row, sort_keys=True), flush=True)
                    except Exception as exception:
                        row.update(
                            {
                                "status": "failed",
                                "error": f"{type(exception).__name__}: {exception}",
                            }
                        )
                        failures.append(row)
                        traceback.print_exc()
                    output.write(json.dumps(row, sort_keys=True) + "\n")
                    output.flush()
                    torch.cuda.empty_cache()

    if failures:
        raise SystemExit(f"{len(failures)} functional matrix cases failed")


if __name__ == "__main__":
    main()
"""


FEATURE_SWEEP_SCRIPT = r"""#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import torch


M_VALUES = (
    1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536,
    2048, 3072, 4096, 6144, 8192, 12288, 16384, 32768,
)
K_VALUES = (
    128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120,
    6144, 8192, 12288, 16384,
)


def load_benchmark(feature_tree):
    path = feature_tree / "benchmarks" / "bench_mxfp4_quantize_backend_comparison.py"
    spec = importlib.util.spec_from_file_location("mxfp4_feature_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-tree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    benchmark = load_benchmark(args.feature_tree)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "m", "k", "dtype", "layout", "status", "quant_match_pct",
        "scale_match_pct", "cuda_time_ms", "cute_dsl_time_ms", "speedup",
        "message",
    )
    failures = []
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for dtype_name, dtype in (
            ("bfloat16", torch.bfloat16),
            ("float16", torch.float16),
        ):
            for layout_name, layout in benchmark.LAYOUTS_BY_NAME.items():
                for m in M_VALUES:
                    for k in K_VALUES:
                        row = {
                            "m": m,
                            "k": k,
                            "dtype": dtype_name,
                            "layout": layout_name,
                        }
                        try:
                            success, message, quant_match, scale_match = (
                                benchmark.verify_mxfp4_correctness(
                                    m, k, dtype, layout
                                )
                            )
                            if not success:
                                raise AssertionError(message)
                            cuda_time = benchmark.bench_mxfp4_quantize(
                                m, k, dtype, layout, backend="cuda"
                            )
                            cute_time = benchmark.bench_mxfp4_quantize(
                                m, k, dtype, layout, backend="cute-dsl"
                            )
                            row.update(
                                {
                                    "status": "passed",
                                    "quant_match_pct": quant_match,
                                    "scale_match_pct": scale_match,
                                    "cuda_time_ms": cuda_time,
                                    "cute_dsl_time_ms": cute_time,
                                    "speedup": cuda_time / cute_time,
                                    "message": message,
                                }
                            )
                        except Exception as exception:
                            row.update(
                                {
                                    "status": "failed",
                                    "message": (
                                        f"{type(exception).__name__}: {exception}"
                                    ),
                                }
                            )
                            failures.append(row.copy())
                        writer.writerow(row)
                        output.flush()
                        print(json.dumps(row, sort_keys=True), flush=True)

    if failures:
        raise SystemExit(f"{len(failures)} feature sweep cases failed")


if __name__ == "__main__":
    main()
"""


def _run_functional_validation(
    result_dir: Path, refs: dict[str, Any]
) -> dict[str, Any]:
    environment = _install_revision(
        FEATURE_TREE,
        "feature",
        refs["feature_sha"],
        result_dir,
        log_suffix="functional",
    )
    helper_path = WORK_ROOT / "functional_matrix.py"
    helper_path.write_text(FUNCTIONAL_MATRIX_SCRIPT, encoding="utf-8")
    matrix_path = result_dir / "functional" / "api_matrix.jsonl"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    return_code = _run_logged(
        [REMOTE_PYTHON, str(helper_path), "--output", str(matrix_path)],
        result_dir / "logs" / "functional" / "api_matrix.log",
        env=environment,
        check=False,
    )
    errors = []
    if return_code != 0:
        errors.append(f"public API matrix exited with code {return_code}")

    runner = FEATURE_TREE / "benchmarks" / "flashinfer_benchmark.py"
    graph_results = []
    for layout in LAYOUTS:
        for enable_pdl in (False, True):
            mode = "graph_pdl" if enable_pdl else "graph"
            output_path = (
                result_dir / "functional" / f"{mode}_{layout.replace('x', 'by')}.csv"
            )
            output_path.unlink(missing_ok=True)
            command = [
                REMOTE_PYTHON,
                str(runner),
                "--routine",
                "mxfp4_quantize",
                "--m",
                "2048",
                "--k",
                "8192",
                "--input_dtype",
                "bfloat16",
                "--sf_layout",
                layout,
                "--backends",
                "cuda",
                "cute-dsl",
                "--refcheck",
                "-vv",
                "--dry_run_iters",
                "5",
                "--num_iters",
                "30",
                "--output_path",
                str(output_path),
            ]
            if enable_pdl:
                command.append("--enable_pdl")
            case_return_code = _run_logged(
                command,
                result_dir
                / "logs"
                / "functional"
                / f"{mode}_{layout.replace('x', 'by')}.log",
                env=environment,
                check=False,
            )
            graph_results.append(
                {
                    "layout": layout,
                    "enable_pdl": enable_pdl,
                    "return_code": case_return_code,
                    "output": str(output_path),
                }
            )
            if case_return_code != 0:
                errors.append(f"{mode} layout={layout} exited with {case_return_code}")

    summary = {
        "api_matrix": str(matrix_path),
        "api_matrix_return_code": return_code,
        "graph_cases": graph_results,
        "errors": errors,
    }
    _write_json(result_dir / "functional" / "summary.json", summary)
    if errors:
        raise RuntimeError("; ".join(errors))
    return summary


def _read_benchmark_csv(
    path: Path, *, label: str, round_index: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        for raw_row in csv.DictReader(csv_file):
            if raw_row.get("routine") != "mxfp4_quantize":
                continue
            rows.append(
                {
                    "label": label,
                    "round": round_index,
                    "m": int(raw_row["m"]),
                    "k": int(raw_row["k"]),
                    "dtype": raw_row["input_dtype"].removeprefix("torch."),
                    "backend": raw_row["backend"],
                    "layout": "128x4",
                    "median_time_ms": float(raw_row["median_time"]),
                    "std_time_ms": float(raw_row["std_time"]),
                }
            )
    if not rows:
        raise RuntimeError(f"No MXFP4 benchmark rows found in {path}")
    return rows


def _write_performance_summary(
    result_dir: Path,
    records: list[dict[str, Any]],
    perf_rounds: int,
) -> list[dict[str, Any]]:
    perf_dir = result_dir / "performance"
    perf_dir.mkdir(parents=True, exist_ok=True)
    raw_path = perf_dir / "ab_rounds.csv"
    fieldnames = (
        "label",
        "round",
        "m",
        "k",
        "dtype",
        "backend",
        "layout",
        "median_time_ms",
        "std_time_ms",
    )
    with raw_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    grouped: dict[tuple[int, int, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        key = (
            record["m"],
            record["k"],
            record["dtype"],
            record["backend"],
        )
        grouped[key][record["label"]].append(record["median_time_ms"])

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        labels = grouped[key]
        baseline = labels.get("baseline", [])
        feature = labels.get("feature", [])
        if len(baseline) != perf_rounds or len(feature) != perf_rounds:
            raise RuntimeError(
                f"Incomplete performance rounds for {key}: "
                f"baseline={len(baseline)}, feature={len(feature)}"
            )
        baseline_median = median(baseline)
        feature_median = median(feature)
        delta_pct = (feature_median - baseline_median) / baseline_median * 100
        row = {
            "m": key[0],
            "k": key[1],
            "dtype": key[2],
            "backend": key[3],
            "baseline_median_ms": baseline_median,
            "feature_median_ms": feature_median,
            "baseline_min_ms": min(baseline),
            "baseline_max_ms": max(baseline),
            "feature_min_ms": min(feature),
            "feature_max_ms": max(feature),
            "delta_pct": delta_pct,
            "speedup": baseline_median / feature_median,
            "non_overlapping_regression": min(feature) > max(baseline),
        }
        summary_rows.append(row)

    if not summary_rows:
        raise RuntimeError("Performance run produced no comparable rows")

    summary_path = perf_dir / "ab_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    _write_json(perf_dir / "ab_summary.json", summary_rows)

    markdown = [
        "# MXFP4 128x4 Performance A/B",
        "",
        "Positive delta means the feature revision is slower.",
        "",
        "| M | K | dtype | backend | baseline median (ms) | baseline min-max (ms) | feature median (ms) | feature min-max (ms) | delta | speedup | non-overlap regression |",
        "| ---: | ---: | --- | --- | ---: | --- | ---: | --- | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        markdown.append(
            "| {m} | {k} | {dtype} | {backend} | {baseline_median_ms:.6f} | "
            "{baseline_min_ms:.6f}-{baseline_max_ms:.6f} | "
            "{feature_median_ms:.6f} | "
            "{feature_min_ms:.6f}-{feature_max_ms:.6f} | "
            "{delta_pct:+.2f}% | {speedup:.4f}x | "
            "{non_overlapping_regression} |".format(**row)
        )
    (perf_dir / "ab_summary.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    return summary_rows


def _run_performance_ab(
    result_dir: Path,
    refs: dict[str, Any],
    *,
    perf_rounds: int,
    perf_warmup: int,
    perf_iters: int,
) -> list[dict[str, Any]]:
    if perf_rounds < 1 or perf_warmup < 1 or perf_iters < 1:
        raise ValueError("performance rounds, warmup, and iterations must be positive")

    runner = TARGET_TREE / "benchmarks" / "flashinfer_benchmark.py"
    records: list[dict[str, Any]] = []
    for round_index in range(1, perf_rounds + 1):
        order = (
            ("baseline", TARGET_TREE, refs["target_sha"]),
            ("feature", FEATURE_TREE, refs["feature_sha"]),
        )
        if round_index % 2 == 0:
            order = tuple(reversed(order))

        for label, tree, sha in order:
            environment = _install_revision(
                tree,
                label,
                sha,
                result_dir,
                log_suffix=f"perf_r{round_index}",
            )
            for m, k in PERF_SHAPES:
                for dtype in PERF_DTYPES:
                    stem = f"{label}_r{round_index}_{dtype}_m{m}_k{k}"
                    output_path = result_dir / "performance" / "rounds" / f"{stem}.csv"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.unlink(missing_ok=True)
                    _run_logged(
                        [
                            REMOTE_PYTHON,
                            str(runner),
                            "--routine",
                            "mxfp4_quantize",
                            "--m",
                            str(m),
                            "--k",
                            str(k),
                            "--input_dtype",
                            dtype,
                            "--sf_layout",
                            "128x4",
                            "--backends",
                            *PERF_BACKENDS,
                            "--refcheck",
                            "--no_cuda_graph",
                            "--use_cuda_events",
                            "--dry_run_iters",
                            str(perf_warmup),
                            "--num_iters",
                            str(perf_iters),
                            "--case_tag",
                            f"{label}-round-{round_index}",
                            "--output_path",
                            str(output_path),
                        ],
                        result_dir / "logs" / "performance" / f"{stem}.log",
                        env=environment,
                    )
                    records.extend(
                        _read_benchmark_csv(
                            output_path,
                            label=label,
                            round_index=round_index,
                        )
                    )

    return _write_performance_summary(result_dir, records, perf_rounds)


def _run_feature_sweep(result_dir: Path, refs: dict[str, Any]) -> dict[str, Any]:
    environment = _install_revision(
        FEATURE_TREE,
        "feature",
        refs["feature_sha"],
        result_dir,
        log_suffix="layout_sweep",
    )
    helper_path = WORK_ROOT / "feature_layout_sweep.py"
    helper_path.write_text(FEATURE_SWEEP_SCRIPT, encoding="utf-8")
    output_path = result_dir / "performance" / "feature_layout_sweep.csv"
    return_code = _run_logged(
        [
            REMOTE_PYTHON,
            str(helper_path),
            "--feature-tree",
            str(FEATURE_TREE),
            "--output",
            str(output_path),
        ],
        result_dir / "logs" / "performance" / "feature_layout_sweep.log",
        env=environment,
        check=False,
    )
    summary = {"return_code": return_code, "output": str(output_path)}
    _write_json(result_dir / "performance" / "feature_layout_sweep.json", summary)
    if return_code != 0:
        raise RuntimeError(f"feature layout sweep exited with code {return_code}")
    return summary


def _run_smoke(result_dir: Path, refs: dict[str, Any]) -> dict[str, str]:
    environment = _install_revision(
        FEATURE_TREE,
        "feature",
        refs["feature_sha"],
        result_dir,
        log_suffix="smoke",
    )
    log_path = result_dir / "logs" / "smoke.log"
    _run_logged(
        [
            REMOTE_PYTHON,
            "-c",
            (
                "import flashinfer, inspect; "
                "print(inspect.signature(flashinfer.mxfp4_quantize)); "
                "print(inspect.signature(flashinfer.mxfp4_dequantize)); "
                "assert 'sfLayout' in inspect.signature("
                "flashinfer.mxfp4_quantize).parameters; "
                "assert 'sfLayout' in inspect.signature("
                "flashinfer.mxfp4_dequantize).parameters"
            ),
        ],
        log_path,
        env=environment,
    )
    return {"log": str(log_path)}


def _write_run_summary(
    result_dir: Path,
    *,
    task: str,
    refs: dict[str, Any] | None,
    steps: list[dict[str, Any]],
    failures: list[str],
) -> None:
    status = "failed" if failures else "passed"
    summary = {
        "status": status,
        "task": task,
        "result_dir": str(result_dir),
        "revisions": refs,
        "steps": steps,
        "failures": failures,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(result_dir / "summary.json", summary)

    lines = [
        "# FlashInfer MXFP4 Layout Validation",
        "",
        f"- Status: `{status}`",
        f"- Task: `{task}`",
    ]
    if refs:
        lines.extend(
            [
                f"- Target: `{refs['target_sha']}`",
                f"- Feature: `{refs['feature_sha']}`",
                f"- Merge base: `{refs['merge_base_sha']}`",
                f"- Target is ancestor: `{refs['target_is_ancestor']}`",
            ]
        )
    lines.extend(
        [
            "",
            "| Step | Status | Duration (s) |",
            "| --- | --- | ---: |",
        ]
    )
    for step in steps:
        lines.append(
            f"| {step['name']} | {step['status']} | {step['duration_seconds']:.1f} |"
        )
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
    (result_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _execute_step(
    name: str,
    function: Callable[[], Any],
    result_dir: Path,
    steps: list[dict[str, Any]],
    failures: list[str],
) -> Any:
    started = time.monotonic()
    try:
        value = function()
    except Exception as exception:
        duration = time.monotonic() - started
        message = f"{name}: {type(exception).__name__}: {exception}"
        failures.append(message)
        steps.append({"name": name, "status": "failed", "duration_seconds": duration})
        error_path = result_dir / "errors" / f"{name}.txt"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        print(message, flush=True)
        return None
    duration = time.monotonic() - started
    steps.append({"name": name, "status": "passed", "duration_seconds": duration})
    return value


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=24 * 60 * 60,
    memory=64 * 1024,
    volumes={str(VOLUME_ROOT): result_volume},
)
def run_acceptance(
    task: str,
    run_id: str,
    feature_repo: str,
    feature_ref: str,
    target_repo: str,
    target_ref: str,
    allow_stale_feature: bool,
    perf_rounds: int,
    perf_warmup: int,
    perf_iters: int,
    skip_feature_sweep: bool,
) -> dict[str, Any]:
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"task must be one of {sorted(SUPPORTED_TASKS)}, got {task!r}")

    run_id = _safe_run_id(run_id)
    result_dir = RESULTS_ROOT / run_id
    if result_dir.exists():
        raise FileExistsError(
            f"Result directory already exists: {result_dir}. Use a new --run-name."
        )
    result_dir.mkdir(parents=True)
    steps: list[dict[str, Any]] = []
    failures: list[str] = []
    refs: dict[str, Any] | None = None

    _write_json(
        result_dir / "request.json",
        {
            "task": task,
            "run_id": run_id,
            "feature_repo": feature_repo,
            "feature_ref": feature_ref,
            "target_repo": target_repo,
            "target_ref": target_ref,
            "allow_stale_feature": allow_stale_feature,
            "perf_rounds": perf_rounds,
            "perf_warmup": perf_warmup,
            "perf_iters": perf_iters,
            "skip_feature_sweep": skip_feature_sweep,
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    try:
        refs = _execute_step(
            "prepare",
            lambda: _prepare_repositories(
                result_dir,
                feature_repo=feature_repo,
                feature_ref=feature_ref,
                target_repo=target_repo,
                target_ref=target_ref,
                allow_stale_feature=allow_stale_feature,
            ),
            result_dir,
            steps,
            failures,
        )
        if refs is not None:
            if task == "smoke":
                _execute_step(
                    "smoke",
                    lambda: _run_smoke(result_dir, refs),
                    result_dir,
                    steps,
                    failures,
                )
            if task in {"tests", "all"}:
                _execute_step(
                    "pytest",
                    lambda: _run_full_pytest(result_dir, refs),
                    result_dir,
                    steps,
                    failures,
                )
            if task in {"functional", "all"} and not failures:
                _execute_step(
                    "functional",
                    lambda: _run_functional_validation(result_dir, refs),
                    result_dir,
                    steps,
                    failures,
                )
            if task in {"performance", "all"} and not failures:
                _execute_step(
                    "performance_ab",
                    lambda: _run_performance_ab(
                        result_dir,
                        refs,
                        perf_rounds=perf_rounds,
                        perf_warmup=perf_warmup,
                        perf_iters=perf_iters,
                    ),
                    result_dir,
                    steps,
                    failures,
                )
                if not skip_feature_sweep:
                    _execute_step(
                        "feature_layout_sweep",
                        lambda: _run_feature_sweep(result_dir, refs),
                        result_dir,
                        steps,
                        failures,
                    )
    finally:
        revisions_path = result_dir / "revisions.json"
        if refs is None and revisions_path.exists():
            refs = json.loads(revisions_path.read_text(encoding="utf-8"))
        _write_run_summary(
            result_dir,
            task=task,
            refs=refs,
            steps=steps,
            failures=failures,
        )
        result_volume.commit()

    relative_result_path = result_dir.relative_to(VOLUME_ROOT).as_posix()
    if failures:
        raise RuntimeError(
            f"Acceptance failed; artifacts: {VOLUME_NAME}/{relative_result_path}; "
            + " | ".join(failures)
        )
    return {
        "status": "passed",
        "task": task,
        "volume": VOLUME_NAME,
        "result_path": relative_result_path,
        "target_sha": refs["target_sha"] if refs else None,
        "feature_sha": refs["feature_sha"] if refs else None,
    }


@app.local_entrypoint()
def main(
    task: str = "smoke",
    gpu: str = GPU_TYPE,
    volume_name: str = VOLUME_NAME,
    run_name: str = "",
    feature_repo: str = DEFAULT_FEATURE_REPO,
    feature_ref: str = DEFAULT_FEATURE_REF,
    target_repo: str = DEFAULT_TARGET_REPO,
    target_ref: str = DEFAULT_TARGET_REF,
    allow_stale_feature: bool = False,
    perf_rounds: int = 5,
    perf_warmup: int = 20,
    perf_iters: int = 100,
    skip_feature_sweep: bool = False,
) -> None:
    """Launch one SM100 acceptance task and print the artifact command."""
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"task must be one of {sorted(SUPPORTED_TASKS)}")
    if gpu != GPU_TYPE:
        raise ValueError(
            f"Decorator GPU is {GPU_TYPE!r}, but local argument is {gpu!r}. "
            "Pass --gpu during modal run so it is available while the app loads."
        )
    if volume_name != VOLUME_NAME:
        raise ValueError(
            f"Mounted Volume is {VOLUME_NAME!r}, but local argument is "
            f"{volume_name!r}. Pass --volume-name during modal run."
        )

    generated_name = "mxfp4-layout-" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_id = _safe_run_id(run_name or generated_name)
    remote_path = f"/results/{run_id}"
    local_parent = "modal-results"
    print(
        json.dumps(
            {
                "task": task,
                "gpu": GPU_TYPE,
                "volume": VOLUME_NAME,
                "run_id": run_id,
                "feature": f"{feature_repo}@{feature_ref}",
                "target": f"{target_repo}@{target_ref}",
            },
            indent=2,
        ),
        flush=True,
    )
    try:
        result = run_acceptance.remote(
            task=task,
            run_id=run_id,
            feature_repo=feature_repo,
            feature_ref=feature_ref,
            target_repo=target_repo,
            target_ref=target_ref,
            allow_stale_feature=allow_stale_feature,
            perf_rounds=perf_rounds,
            perf_warmup=perf_warmup,
            perf_iters=perf_iters,
            skip_feature_sweep=skip_feature_sweep,
        )
        print(json.dumps(result, indent=2), flush=True)
    finally:
        print("\nDownload artifacts with:", flush=True)
        print(f"mkdir -p {shlex.quote(local_parent)}", flush=True)
        print(
            f"modal volume get {shlex.quote(VOLUME_NAME)} "
            f"{shlex.quote(remote_path)} {shlex.quote(local_parent)}",
            flush=True,
        )
