"""Modal runner for DeepGEMM Mega MoE benchmarks.

Run from the lab-notebook repository:

    modal run deepgemm/modal_deepgemm_mega_moe.py --task smoke

Reproduce the PR #316 Mega MoE sweep:

    modal run deepgemm/modal_deepgemm_mega_moe.py \
      --task table --model both --gpu B200:8 --deepgemm-ref main

The wrapper clones DeepGEMM from GitHub during Modal image build, builds it,
and runs tests/test_mega_moe.py remotely. Results are written to a Modal
Volume under /cache/deepgemm/results/<run_id>/.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from statistics import mean

import modal


APP_NAME = "deepgemm-mega-moe-bench"
WORKDIR = "/workspace/DeepGEMM"
DEFAULT_DEEPGEMM_REPO = "https://github.com/deepseek-ai/DeepGEMM.git"
REMOTE_PYTHON = "python"

DEFAULT_BATCH_SIZES = "1,512,8192,32768"
MODEL_CONFIGS = {
    "flash": {
        "label": "DeepSeek-V4-Flash",
        "num_experts": 256,
        "num_topk": 6,
        "hidden": 4096,
        "intermediate_hidden": 2048,
    },
    "pro": {
        "label": "DeepSeek-V4-Pro",
        "num_experts": 384,
        "num_topk": 6,
        "hidden": 7168,
        "intermediate_hidden": 3072,
    },
}

PERF_RE = re.compile(
    r"EP:\s*(?P<rank>\d+)\s*/\s*(?P<num_ranks>\d+)\s*\|\s*"
    r"(?P<tflops>[-+\d.]+)\s+TFLOPS\s*\|\s*"
    r"overlap:\s*(?P<overlap_tflops>[-+\d.]+)\s+TFLOPS,\s*"
    r"HBM\s*(?P<hbm_gbs>[-+\d.]+)\s+GB/s,\s*"
    r"NVL\s*(?P<nvl_gbs>[-+\d.]+)\s+GB/s\s*\|\s*"
    r"(?P<time_us>[-+\d.]+)\s+us,.*?\|\s*"
    r"(?P<speedup>[-+\d.]+|nan|inf)x legacy"
)


def _arg_value(flag: str) -> str | None:
    for idx, arg in enumerate(sys.argv):
        if arg == flag and idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        prefix = f"{flag}="
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def _git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--short=12", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _clone_deepgemm_command(repo_url: str, ref: str) -> str:
    repo_arg = shlex.quote(repo_url)
    ref_arg = shlex.quote(ref)
    workdir_arg = shlex.quote(WORKDIR)
    return (
        "set -eux; "
        f"rm -rf {workdir_arg}; "
        f"git clone --recursive --depth=1 --branch {ref_arg} {repo_arg} {workdir_arg} || "
        "( "
        f"rm -rf {workdir_arg}; "
        f"git clone --recursive {repo_arg} {workdir_arg}; "
        f"cd {workdir_arg}; "
        f"git fetch --depth=1 origin {ref_arg} || git fetch origin {ref_arg} || true; "
        "git checkout --detach FETCH_HEAD || git checkout --detach "
        f"{ref_arg} || git checkout {ref_arg}; "
        "git submodule update --init --recursive "
        "); "
        f"cd {workdir_arg}; "
        "git rev-parse --short=12 HEAD"
    )


deepgemm_repo_url = (
    _arg_value("--deepgemm-repo")
    or os.environ.get("DEEPGEMM_REPO_URL")
    or DEFAULT_DEEPGEMM_REPO
)
deepgemm_git_ref = (
    _arg_value("--deepgemm-ref")
    or os.environ.get("DEEPGEMM_REF")
    or "main"
)
gpu_type = _arg_value("--gpu") or os.environ.get("MODAL_GPU", "B200:8")
base_image = os.environ.get("DEEPGEMM_BASE_IMAGE", "nvcr.io/nvidia/pytorch:26.04-py3")
cache_volume_name = os.environ.get("MODAL_CACHE_VOLUME_NAME", "deepgemm-cache")
install_baseline = os.environ.get("DEEPGEMM_INSTALL_BASELINE", "0") == "1"

cache_volume = modal.Volume.from_name(cache_volume_name, create_if_missing=True)
app = modal.App(APP_NAME)

image_env = {
    "CUDA_HOME": os.environ.get("CUDA_HOME", "/usr/local/cuda"),
    "DG_JIT_CACHE_DIR": "/cache/deepgemm/jit",
    "DG_JIT_NVCC_COMPILER": os.environ.get(
        "DG_JIT_NVCC_COMPILER", "/usr/local/cuda/bin/nvcc"
    ),
    "MAX_JOBS": os.environ.get("MAX_JOBS", "16"),
    "PIP_BREAK_SYSTEM_PACKAGES": "1",
    "PYTHONUNBUFFERED": "1",
    "DEEPGEMM_REPO_URL": deepgemm_repo_url,
    "DEEPGEMM_REF": deepgemm_git_ref,
    "MODAL_GPU_REQUEST": gpu_type,
}

baseline_commands = []
if install_baseline:
    baseline_commands = [
        "python -m pip install --upgrade tilelang",
        "python -m pip install --no-build-isolation git+https://github.com/deepseek-ai/DeepEP.git",
    ]

image = (
    modal.Image.from_registry(base_image, add_python=None)
    .apt_install("git", "build-essential", "cmake", "ninja-build", "python3-dev")
    .run_commands(_clone_deepgemm_command(deepgemm_repo_url, deepgemm_git_ref))
    .env(image_env)
    .workdir(WORKDIR)
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel packaging ninja",
        (
            "if [ ! -d third-party/cutlass/include ]; then "
            "rm -rf third-party/cutlass; "
            "git clone --depth=1 https://github.com/NVIDIA/cutlass.git third-party/cutlass; "
            "fi"
        ),
        (
            "if [ ! -d third-party/fmt/include ]; then "
            "rm -rf third-party/fmt; "
            "git clone --depth=1 https://github.com/fmtlib/fmt.git third-party/fmt; "
            "fi"
        ),
        "bash develop.sh",
        *baseline_commands,
    )
)


def _run(cmd: list[str], *, cwd: str = WORKDIR) -> None:
    print("$ " + shlex.join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _run_to_file(
    cmd: list[str],
    log_path: Path,
    *,
    cwd: str = WORKDIR,
    env: dict[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    shell_cmd = f"{shlex.join(cmd)} 2>&1 | tee {shlex.quote(str(log_path))}"
    print("$ " + shell_cmd, flush=True)
    subprocess.run(
        ["bash", "-o", "pipefail", "-lc", shell_cmd],
        cwd=cwd,
        env=env,
        check=True,
    )


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _select_models(model: str) -> list[str]:
    normalized = model.lower().replace("_", "-")
    if normalized == "both":
        return ["flash", "pro"]
    if normalized in ("flash", "deepseek-v4-flash", "v4-flash"):
        return ["flash"]
    if normalized in ("pro", "deepseek-v4-pro", "v4-pro"):
        return ["pro"]
    raise ValueError(f"Unknown model: {model}. Use flash, pro, or both.")


def _prepare_runtime(num_processes: int, require_baseline: bool) -> None:
    os.chdir(WORKDIR)
    pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{WORKDIR}:{pythonpath}" if pythonpath else WORKDIR
    Path("/cache/deepgemm/jit").mkdir(parents=True, exist_ok=True)

    _run(["nvidia-smi", "-L"])

    import torch

    print(f"torch={torch.__version__}, cuda={torch.version.cuda}", flush=True)
    print(f"cuda_device_count={torch.cuda.device_count()}", flush=True)
    if torch.cuda.device_count() < num_processes:
        raise RuntimeError(
            f"Requested {num_processes} processes but only "
            f"{torch.cuda.device_count()} CUDA devices are visible."
        )

    import torch.distributed._symmetric_memory  # noqa: F401
    import deep_gemm

    print(f"deep_gemm={deep_gemm.__version__}", flush=True)
    if require_baseline:
        import deep_ep  # noqa: F401
        import tilelang  # noqa: F401


def _read_deepep_commit_from_direct_url() -> str | None:
    for distribution_name in ("deep_ep", "deepep", "DeepEP"):
        try:
            dist = metadata.distribution(distribution_name)
        except metadata.PackageNotFoundError:
            continue
        direct_url = dist.read_text("direct_url.json")
        if not direct_url:
            continue
        try:
            vcs_info = json.loads(direct_url).get("vcs_info", {})
        except json.JSONDecodeError:
            continue
        commit_id = vcs_info.get("commit_id")
        if commit_id:
            return commit_id[:12]
        requested_revision = vcs_info.get("requested_revision")
        if requested_revision:
            return requested_revision[:12]
    return None


def _read_deepep_commit_from_package() -> str | None:
    try:
        import deep_ep
    except Exception:
        return None

    package_file = getattr(deep_ep, "__file__", None)
    if not package_file:
        return None

    path = Path(package_file).resolve()
    for parent in [path.parent, *path.parents]:
        commit = _git_commit(parent)
        if commit != "unknown":
            return commit
    return None


def _collect_run_metadata() -> dict[str, int | str]:
    import torch

    gpu_count = torch.cuda.device_count()
    gpu_models = sorted({torch.cuda.get_device_name(i) for i in range(gpu_count)})
    deepep_commit = (
        os.environ.get("DEEPEP_SOURCE_COMMIT")
        or _read_deepep_commit_from_direct_url()
        or _read_deepep_commit_from_package()
        or "not-installed"
    )
    return {
        "deepgemm_commit": _git_commit(Path(WORKDIR)),
        "deepep_commit": deepep_commit,
        "gpu_model": " + ".join(gpu_models) if gpu_models else "unknown",
        "gpu_count": gpu_count,
        "modal_gpu_request": os.environ.get("MODAL_GPU_REQUEST", "unknown"),
    }


def _enable_all_rank_printing() -> None:
    dist_py = Path(WORKDIR) / "deep_gemm" / "utils" / "dist.py"
    text = dist_py.read_text()
    old = "if not once_in_node or _local_rank == 0:\n        print(s, flush=True)"
    new = "if True:\n        print(s, flush=True)"
    if old in text:
        dist_py.write_text(text.replace(old, new))
        print("Patched deep_gemm.utils.dist.dist_print to print all ranks.", flush=True)


def _parse_perf_rows(log_path: Path) -> list[dict[str, float]]:
    rows = []
    for line in log_path.read_text(errors="replace").splitlines():
        for match in PERF_RE.finditer(line):
            row: dict[str, float] = {}
            for key, value in match.groupdict().items():
                row[key] = float(value)
            rows.append(row)
    return rows


def _summarize_case(
    *,
    model_key: str,
    batch_size: int,
    log_path: Path,
    expected_ranks: int,
    run_metadata: dict[str, int | str],
) -> dict[str, float | int | str]:
    rows = _parse_perf_rows(log_path)
    if not rows:
        raise RuntimeError(f"No performance lines found in {log_path}")

    summary = {
        "model": MODEL_CONFIGS[model_key]["label"],
        "batch_size": batch_size,
        "rank_rows": len(rows),
        "expected_ranks": expected_ranks,
        "time_us": mean(row["time_us"] for row in rows),
        "compute_tflops": mean(row["tflops"] for row in rows),
        "global_memory_gbs": mean(row["hbm_gbs"] for row in rows),
        "interconnect_gbs": mean(row["nvl_gbs"] for row in rows),
        "speedup_vs_legacy": mean(row["speedup"] for row in rows),
        "log_path": str(log_path),
        **run_metadata,
    }
    return summary


def _write_summaries(result_dir: Path, summaries: list[dict[str, float | int | str]]) -> None:
    json_path = result_dir / "summary.json"
    csv_path = result_dir / "summary.csv"
    json_path.write_text(json.dumps(summaries, indent=2) + "\n")

    fieldnames = [
        "model",
        "batch_size",
        "rank_rows",
        "expected_ranks",
        "time_us",
        "compute_tflops",
        "global_memory_gbs",
        "interconnect_gbs",
        "speedup_vs_legacy",
        "deepgemm_commit",
        "deepep_commit",
        "gpu_model",
        "gpu_count",
        "modal_gpu_request",
        "log_path",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {csv_path}", flush=True)


def _format_table(summaries: list[dict[str, float | int | str]], result_dir: Path) -> str:
    lines = [f"Result dir: {result_dir}", ""]
    by_model: dict[str, list[dict[str, float | int | str]]] = {}
    for row in summaries:
        by_model.setdefault(str(row["model"]), []).append(row)

    for model_name, rows in by_model.items():
        lines.append(model_name)
        lines.append(
            "| Batch Size | Time (us) | Compute (TFLOPS) | "
            "Global Memory (GB/s) | Interconnect (GB/s) | Speedup (vs legacy) | "
            "DeepGEMM Commit | DeepEP Commit | GPU | Cards |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---|---|---|---:|")
        for row in sorted(rows, key=lambda item: int(item["batch_size"])):
            lines.append(
                f"| {int(row['batch_size'])} "
                f"| {float(row['time_us']):.1f} "
                f"| {float(row['compute_tflops']):.0f} "
                f"| {float(row['global_memory_gbs']):.0f} "
                f"| {float(row['interconnect_gbs']):.0f} "
                f"| {float(row['speedup_vs_legacy']):.2f}x "
                f"| {row['deepgemm_commit']} "
                f"| {row['deepep_commit']} "
                f"| {row['gpu_model']} "
                f"| {int(row['gpu_count'])} |"
            )
        lines.append("")
    return "\n".join(lines)


@app.function(
    image=image,
    gpu=gpu_type,
    timeout=8 * 60 * 60,
    volumes={"/cache": cache_volume},
)
def run_benchmark(
    task: str = "table",
    model: str = "both",
    batch_size: int = 8192,
    batch_sizes: str = DEFAULT_BATCH_SIZES,
    num_processes: int = 8,
    num_correctness_tests: int = 0,
    activation_clamp: float = 10.0,
    fast_math: int = 1,
    masked_ratio: float = 0.0,
    all_rank_logs: bool = True,
    require_baseline: bool = False,
    print_configs: bool = False,
    ncu_profile_only: bool = False,
) -> str:
    _prepare_runtime(num_processes, require_baseline)
    if all_rank_logs:
        _enable_all_rank_printing()

    run_id = time.strftime("%Y%m%d-%H%M%S")
    result_dir = Path("/cache/deepgemm/results") / run_id
    result_dir.mkdir(parents=True, exist_ok=True)

    selected_models = _select_models(model)
    selected_batches = [batch_size] if task == "case" else _parse_int_list(batch_sizes)
    run_metadata = _collect_run_metadata()
    summaries: list[dict[str, float | int | str]] = []
    case_index = 0

    for model_key in selected_models:
        cfg = MODEL_CONFIGS[model_key]
        for bsz in selected_batches:
            case_name = f"{model_key}_bsz{bsz}"
            log_path = result_dir / f"{case_name}.log"
            env = os.environ.copy()
            env.update(
                {
                    "MASTER_ADDR": "localhost",
                    "MASTER_PORT": str(8361 + case_index),
                    "DG_PRINT_CONFIGS": "1" if print_configs else "0",
                    "PYTHONPATH": f"{WORKDIR}:{env.get('PYTHONPATH', '')}",
                }
            )

            cmd = [
                REMOTE_PYTHON,
                "tests/test_mega_moe.py",
                "--num-processes",
                str(num_processes),
                "--num-max-tokens-per-rank",
                str(bsz),
                "--num-tokens",
                str(bsz),
                "--num-experts",
                str(cfg["num_experts"]),
                "--num-topk",
                str(cfg["num_topk"]),
                "--hidden",
                str(cfg["hidden"]),
                "--intermediate-hidden",
                str(cfg["intermediate_hidden"]),
                "--activation-clamp",
                str(activation_clamp),
                "--fast-math",
                str(fast_math),
                "--masked-ratio",
                str(masked_ratio),
                "--num-correctness-tests",
                str(num_correctness_tests),
            ]
            if ncu_profile_only:
                cmd.append("--ncu-profile-only")

            _run_to_file(cmd, log_path, env=env)
            if not ncu_profile_only:
                summaries.append(
                    _summarize_case(
                        model_key=model_key,
                        batch_size=bsz,
                        log_path=log_path,
                        expected_ranks=num_processes,
                        run_metadata=run_metadata,
                    )
                )
            case_index += 1

    if ncu_profile_only:
        return f"NCU-profile-only run complete. Logs: {result_dir}"

    _write_summaries(result_dir, summaries)
    return _format_table(summaries, result_dir)


@app.local_entrypoint()
def main(
    task: str = "table",
    model: str = "both",
    batch_size: int = 8192,
    batch_sizes: str = DEFAULT_BATCH_SIZES,
    deepgemm_repo: str = deepgemm_repo_url,
    deepgemm_ref: str = deepgemm_git_ref,
    gpu: str = gpu_type,
    num_processes: int = 8,
    num_correctness_tests: int = 0,
    activation_clamp: float = 10.0,
    fast_math: int = 1,
    masked_ratio: float = 0.0,
    all_rank_logs: bool = True,
    require_baseline: bool = False,
    print_configs: bool = False,
    ncu_profile_only: bool = False,
) -> None:
    """Run a DeepGEMM Mega MoE benchmark on Modal."""
    if deepgemm_repo != deepgemm_repo_url or deepgemm_ref != deepgemm_git_ref:
        print(
            "DeepGEMM source is fixed while the app is imported. "
            f"Using repo={deepgemm_repo_url!r}, ref={deepgemm_git_ref!r}.",
            flush=True,
        )
    if gpu != gpu_type:
        print(
            "The Modal GPU is fixed while the app is imported. "
            f"Using decorator GPU {gpu_type!r}; pass --gpu before running if needed.",
            flush=True,
        )

    if task == "smoke":
        task = "case"
        model = "flash"
        batch_size = 1
        ncu_profile_only = True

    if task not in ("case", "table"):
        raise ValueError("task must be one of: smoke, case, table")

    result = run_benchmark.remote(
        task=task,
        model=model,
        batch_size=batch_size,
        batch_sizes=batch_sizes,
        num_processes=num_processes,
        num_correctness_tests=num_correctness_tests,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
        masked_ratio=masked_ratio,
        all_rank_logs=all_rank_logs,
        require_baseline=require_baseline,
        print_configs=print_configs,
        ncu_profile_only=ncu_profile_only,
    )
    print(result)
