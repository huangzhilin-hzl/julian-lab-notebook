"""Modal runner for MiniMax-AI/MSA benchmark commands.

Run from the lab-notebook repository:

    modal run msa-modal/modal_msa_benchmark.py --task smoke

README-aligned quick benchmark:

    modal run msa-modal/modal_msa_benchmark.py --preset smoke

README-aligned full presets:

    modal run msa-modal/modal_msa_benchmark.py --preset fp8
    modal run msa-modal/modal_msa_benchmark.py --preset bf16
    modal run msa-modal/modal_msa_benchmark.py --preset nvfp4
    modal run msa-modal/modal_msa_benchmark.py --preset readme

The wrapper clones MiniMax-AI/MSA during Modal image build, installs it in
editable mode, and runs benchmarks/bench_sparse_attention_ops.py remotely.
Results are written to a Modal Volume under /mnt/msa-cache/msa/results/<run_id>/.
"""

from __future__ import annotations

import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal


APP_NAME = "minimax-msa-benchmark"
WORKDIR = "/workspace/MSA"
REMOTE_PYTHON = "python"
DEFAULT_MSA_REPO = "https://github.com/MiniMax-AI/MSA.git"
README_BENCHMARK = "benchmarks/bench_sparse_attention_ops.py"


def _arg_value(flag: str) -> str | None:
    for idx, arg in enumerate(sys.argv):
        if arg == flag and idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        prefix = f"{flag}="
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def _clone_msa_command(repo_url: str, ref: str) -> str:
    repo_arg = shlex.quote(repo_url)
    ref_arg = shlex.quote(ref)
    workdir_arg = shlex.quote(WORKDIR)
    return (
        "set -eux; "
        f"rm -rf {workdir_arg}; "
        f"git clone --recursive --depth=1 --branch {ref_arg} "
        f"{repo_arg} {workdir_arg} || "
        "( "
        f"rm -rf {workdir_arg}; "
        f"git clone --recursive {repo_arg} {workdir_arg}; "
        f"cd {workdir_arg}; "
        f"git fetch --depth=1 origin {ref_arg} || git fetch origin {ref_arg} || true; "
        "git checkout --detach FETCH_HEAD || git checkout --detach "
        f"{ref_arg} || git checkout {ref_arg}; "
        "git submodule update --init --recursive --depth=1 "
        "); "
        f"cd {workdir_arg}; "
        "git submodule update --init --recursive --depth=1; "
        "git rev-parse --short=12 HEAD"
    )


msa_repo_url = (
    _arg_value("--msa-repo")
    or os.environ.get("MSA_REPO_URL")
    or DEFAULT_MSA_REPO
)
msa_git_ref = _arg_value("--msa-ref") or os.environ.get("MSA_REF") or "main"
modal_gpu_type = (
    _arg_value("--modal-gpu")
    or os.environ.get("MODAL_GPU")
    or "B200"
)
base_image = os.environ.get("MSA_BASE_IMAGE", "nvcr.io/nvidia/pytorch:26.04-py3")
cache_volume_name = os.environ.get("MODAL_CACHE_VOLUME_NAME", "msa-cache")
CACHE_ROOT = os.environ.get("MSA_CACHE_ROOT", "/mnt/msa-cache")
MSA_CACHE_DIR = f"{CACHE_ROOT}/msa"

cache_volume = modal.Volume.from_name(cache_volume_name, create_if_missing=True)
app = modal.App(APP_NAME)

image_env = {
    "CUDA_HOME": os.environ.get("CUDA_HOME", "/usr/local/cuda"),
    "MAX_JOBS": os.environ.get("MAX_JOBS", "16"),
    "PIP_BREAK_SYSTEM_PACKAGES": "1",
    "PYTHONUNBUFFERED": "1",
    "MSA_REPO_URL": msa_repo_url,
    "MSA_REF": msa_git_ref,
    "MODAL_GPU_REQUEST": modal_gpu_type,
}

runtime_cache_env = {
    "MINFER_FMHA_CACHE_DIR": f"{MSA_CACHE_DIR}/fmha_sm100",
    "XDG_CACHE_HOME": f"{MSA_CACHE_DIR}/xdg",
    "TORCHINDUCTOR_CACHE_DIR": f"{MSA_CACHE_DIR}/torchinductor",
}

image = (
    modal.Image.from_registry(base_image, add_python=None)
    .apt_install("git", "build-essential", "cmake", "ninja-build", "python3-dev")
    .run_commands(_clone_msa_command(msa_repo_url, msa_git_ref))
    .workdir(WORKDIR)
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel packaging ninja pytest",
        "python -m pip install -e . --no-build-isolation",
    )
    .env(image_env)
)


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    dtype: str
    sections: str
    output_mode: str
    seqs: str
    tp: str
    decode_k: str
    decode_b: str
    topk: int
    head_dim: int
    blk_kv: int
    dry_run_ms: int
    repeat_ms: int


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


def _check_output(cmd: list[str], *, cwd: str = WORKDIR) -> str:
    try:
        return subprocess.check_output(
            cmd,
            cwd=cwd,
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def _git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--short=12", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _coerce_tsv_value(key: str, value: str) -> int | float | str:
    if value == "":
        return value
    int_keys = {
        "q_len",
        "kv_len",
        "batch_size",
        "q_head",
        "kv_head",
        "head_dim",
    }
    float_keys = {
        "latency_ms",
        "std_ms",
        "tflops",
        "gbs",
        "mfu_mbu_pct",
        "predicted_split_speedup_pct",
    }
    if key in int_keys:
        return int(value)
    if key in float_keys:
        return float(value)
    return value


def _parse_tsv(path: Path) -> list[dict[str, int | float | str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = []
        for row in reader:
            rows.append({key: _coerce_tsv_value(key, value) for key, value in row.items()})
    return rows


def _collect_run_metadata(require_sm100: bool) -> dict[str, Any]:
    import torch

    gpu_count = torch.cuda.device_count()
    gpu_models = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    compute_caps = [torch.cuda.get_device_capability(i) for i in range(gpu_count)]

    if require_sm100 and not any(major == 10 for major, _ in compute_caps):
        raise RuntimeError(
            "MSA requires NVIDIA SM100. Visible CUDA capabilities: "
            + ", ".join(f"{major}.{minor}" for major, minor in compute_caps)
        )

    import fmha_sm100

    return {
        "msa_repo": os.environ.get("MSA_REPO_URL", DEFAULT_MSA_REPO),
        "msa_ref": os.environ.get("MSA_REF", "main"),
        "msa_commit": _git_commit(Path(WORKDIR)),
        "msa_package": str(Path(fmha_sm100.__file__).resolve()),
        "modal_gpu_request": os.environ.get("MODAL_GPU_REQUEST", "unknown"),
        "gpu_count": gpu_count,
        "gpu_models": gpu_models,
        "compute_capabilities": [f"{major}.{minor}" for major, minor in compute_caps],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nvcc": _check_output(["nvcc", "--version"]),
        "benchmark_script": README_BENCHMARK,
    }


def _prepare_runtime(require_sm100: bool) -> dict[str, Any]:
    os.chdir(WORKDIR)
    for key, value in runtime_cache_env.items():
        os.environ.setdefault(key, value)
    Path(os.environ["MINFER_FMHA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    _run(["nvidia-smi"])
    metadata = _collect_run_metadata(require_sm100=require_sm100)
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


def _spec(
    *,
    name: str,
    dtype: str,
    sections: str,
    output_mode: str,
    seqs: str,
    tp: str,
    decode_k: str,
    decode_b: str,
    topk: int,
    head_dim: int,
    blk_kv: int,
    dry_run_ms: int,
    repeat_ms: int,
) -> BenchmarkSpec:
    return BenchmarkSpec(
        name=name,
        dtype=dtype,
        sections=sections,
        output_mode=output_mode,
        seqs=seqs,
        tp=tp,
        decode_k=decode_k,
        decode_b=decode_b,
        topk=topk,
        head_dim=head_dim,
        blk_kv=blk_kv,
        dry_run_ms=dry_run_ms,
        repeat_ms=repeat_ms,
    )


def _full_dry_run_ms(value: int) -> int:
    return value if value > 0 else 200


def _full_repeat_ms(value: int) -> int:
    return value if value > 0 else 2000


def _smoke_dry_run_ms(value: int) -> int:
    return value if value > 0 else 50


def _smoke_repeat_ms(value: int) -> int:
    return value if value > 0 else 200


def _benchmark_specs(
    *,
    preset: str,
    dtype: str,
    sections: str,
    output_mode: str,
    seqs: str,
    tp: str,
    decode_k: str,
    decode_b: str,
    topk: int,
    head_dim: int,
    blk_kv: int,
    dry_run_ms: int,
    repeat_ms: int,
) -> list[BenchmarkSpec]:
    preset = preset.lower().replace("_", "-")

    if preset == "smoke":
        return [
            _spec(
                name="msa_smoke",
                dtype=dtype or "fp8",
                sections=sections or "prefill,decode,sparse_decode",
                output_mode=output_mode or "o",
                seqs=seqs or "8192,16384",
                tp=tp or "1,4",
                decode_k=decode_k or "8192,131072",
                decode_b=decode_b or "32",
                topk=topk,
                head_dim=head_dim,
                blk_kv=blk_kv,
                dry_run_ms=_smoke_dry_run_ms(dry_run_ms),
                repeat_ms=_smoke_repeat_ms(repeat_ms),
            )
        ]

    if preset == "fp8":
        return [
            _spec(
                name="msa_fp8",
                dtype="fp8",
                sections=sections or "all",
                output_mode=output_mode or "o",
                seqs=seqs,
                tp=tp,
                decode_k=decode_k,
                decode_b=decode_b,
                topk=topk,
                head_dim=head_dim,
                blk_kv=blk_kv,
                dry_run_ms=_full_dry_run_ms(dry_run_ms),
                repeat_ms=_full_repeat_ms(repeat_ms),
            )
        ]

    if preset == "bf16":
        return [
            _spec(
                name="msa_bf16",
                dtype="bf16",
                sections=sections or "all",
                output_mode=output_mode or "o",
                seqs=seqs,
                tp=tp,
                decode_k=decode_k,
                decode_b=decode_b,
                topk=topk,
                head_dim=head_dim,
                blk_kv=blk_kv,
                dry_run_ms=_full_dry_run_ms(dry_run_ms),
                repeat_ms=_full_repeat_ms(repeat_ms),
            )
        ]

    if preset == "nvfp4":
        return [
            _spec(
                name="msa_nvfp4",
                dtype="nvfp4",
                sections=sections or "sparse_prefill",
                output_mode=output_mode or "o",
                seqs=seqs,
                tp=tp,
                decode_k=decode_k,
                decode_b=decode_b,
                topk=topk,
                head_dim=head_dim,
                blk_kv=blk_kv,
                dry_run_ms=_full_dry_run_ms(dry_run_ms),
                repeat_ms=_full_repeat_ms(repeat_ms),
            )
        ]

    if preset in {"readme", "readme-all"}:
        return (
            _benchmark_specs(
                preset="smoke",
                dtype="",
                sections=sections,
                output_mode=output_mode,
                seqs=seqs,
                tp=tp,
                decode_k=decode_k,
                decode_b=decode_b,
                topk=topk,
                head_dim=head_dim,
                blk_kv=blk_kv,
                dry_run_ms=dry_run_ms,
                repeat_ms=repeat_ms,
            )
            + _benchmark_specs(
                preset="fp8",
                dtype="",
                sections=sections,
                output_mode=output_mode,
                seqs=seqs,
                tp=tp,
                decode_k=decode_k,
                decode_b=decode_b,
                topk=topk,
                head_dim=head_dim,
                blk_kv=blk_kv,
                dry_run_ms=dry_run_ms,
                repeat_ms=repeat_ms,
            )
            + _benchmark_specs(
                preset="bf16",
                dtype="",
                sections=sections,
                output_mode=output_mode,
                seqs=seqs,
                tp=tp,
                decode_k=decode_k,
                decode_b=decode_b,
                topk=topk,
                head_dim=head_dim,
                blk_kv=blk_kv,
                dry_run_ms=dry_run_ms,
                repeat_ms=repeat_ms,
            )
            + _benchmark_specs(
                preset="nvfp4",
                dtype="",
                sections="sparse_prefill",
                output_mode=output_mode,
                seqs=seqs,
                tp=tp,
                decode_k=decode_k,
                decode_b=decode_b,
                topk=topk,
                head_dim=head_dim,
                blk_kv=blk_kv,
                dry_run_ms=dry_run_ms,
                repeat_ms=repeat_ms,
            )
        )

    if preset == "custom":
        custom_dtype = dtype or "fp8"
        custom_sections = sections or ("sparse_prefill" if custom_dtype == "nvfp4" else "all")
        return [
            _spec(
                name=f"msa_custom_{custom_dtype}",
                dtype=custom_dtype,
                sections=custom_sections,
                output_mode=output_mode or "o",
                seqs=seqs,
                tp=tp,
                decode_k=decode_k,
                decode_b=decode_b,
                topk=topk,
                head_dim=head_dim,
                blk_kv=blk_kv,
                dry_run_ms=_full_dry_run_ms(dry_run_ms),
                repeat_ms=_full_repeat_ms(repeat_ms),
            )
        ]

    raise ValueError("preset must be one of: smoke, fp8, bf16, nvfp4, readme, custom")


def _benchmark_log_errors(log_path: Path) -> list[str]:
    if not log_path.exists():
        return [f"missing log file: {log_path}"]
    return [
        line
        for line in log_path.read_text(errors="replace").splitlines()
        if "ERROR:" in line
    ]


def _benchmark_cmd(spec: BenchmarkSpec, output_path: Path, gpu_id: int) -> list[str]:
    cmd = [
        REMOTE_PYTHON,
        README_BENCHMARK,
        "--gpu",
        str(gpu_id),
        "--dtype",
        spec.dtype,
        "--sections",
        spec.sections,
        "--output_mode",
        spec.output_mode,
        "--topk",
        str(spec.topk),
        "--head-dim",
        str(spec.head_dim),
        "--blk-kv",
        str(spec.blk_kv),
        "--dry-run-ms",
        str(spec.dry_run_ms),
        "--repeat-ms",
        str(spec.repeat_ms),
        "-o",
        str(output_path),
    ]
    if spec.seqs:
        cmd.extend(["--seqs", spec.seqs])
    if spec.tp:
        cmd.extend(["--tp", spec.tp])
    if spec.decode_k:
        cmd.extend(["--decode-k", spec.decode_k])
    if spec.decode_b:
        cmd.extend(["--decode-b", spec.decode_b])
    return cmd


def _write_artifacts(
    *,
    result_dir: Path,
    metadata: dict[str, Any],
    run_records: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
) -> None:
    summary = {
        "metadata": metadata,
        "runs": run_records,
        "results": result_rows,
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (result_dir / "results.json").write_text(json.dumps(result_rows, indent=2) + "\n")
    (result_dir / "summary.md").write_text(_format_markdown(summary, result_dir))

    print(f"Wrote {result_dir / 'summary.json'}", flush=True)
    print(f"Wrote {result_dir / 'results.json'}", flush=True)
    print(f"Wrote {result_dir / 'summary.md'}", flush=True)


def _format_markdown(summary: dict[str, Any], result_dir: Path) -> str:
    metadata = summary["metadata"]
    runs = summary["runs"]
    rows = summary["results"]

    lines = [
        "# MiniMax MSA Modal Benchmark",
        "",
        f"Result dir: `{result_dir}`",
        "",
        "## Provenance",
        "",
        f"- MSA repo: `{metadata['msa_repo']}`",
        f"- MSA ref: `{metadata['msa_ref']}`",
        f"- MSA commit: `{metadata['msa_commit']}`",
        f"- Modal GPU request: `{metadata['modal_gpu_request']}`",
        f"- GPU: `{', '.join(metadata['gpu_models'])}`",
        f"- Compute capability: `{', '.join(metadata['compute_capabilities'])}`",
        f"- Torch: `{metadata['torch']}` / CUDA `{metadata['torch_cuda']}`",
        "",
        "## Runs",
        "",
        "| Run | Rows | Dtype | Sections | TSV | Log |",
        "|---|---:|---|---|---|---|",
    ]
    for record in runs:
        lines.append(
            f"| `{record['name']}` | {record['row_count']} | `{record['dtype']}` "
            f"| `{record['sections']}` | `{record['tsv_path']}` | `{record['log_path']}` |"
        )

    lines.extend(["", "## Preview", ""])
    if not rows:
        lines.append("No benchmark rows were parsed.")
    else:
        preview_keys = [
            "run",
            "q_len",
            "kv_len",
            "dtype",
            "batch_size",
            "q_head",
            "kv_head",
            "latency_ms",
            "tflops",
            "gbs",
            "mfu_mbu_pct",
        ]
        lines.append("| " + " | ".join(preview_keys) + " |")
        lines.append("|" + "|".join("---" for _ in preview_keys) + "|")
        for row in rows[:30]:
            lines.append("| " + " | ".join(str(row.get(key, "")) for key in preview_keys) + " |")
        if len(rows) > 30:
            lines.append(f"\nShowing first 30 of {len(rows)} rows. See `results.json` for all rows.")

    return "\n".join(lines) + "\n"


@app.function(
    image=image,
    gpu=modal_gpu_type,
    timeout=8 * 60 * 60,
    volumes={CACHE_ROOT: cache_volume},
)
def run_benchmark(
    preset: str = "smoke",
    dtype: str = "",
    sections: str = "",
    output_mode: str = "o",
    seqs: str = "",
    tp: str = "",
    decode_k: str = "",
    decode_b: str = "",
    topk: int = 16,
    head_dim: int = 128,
    blk_kv: int = 128,
    dry_run_ms: int = 0,
    repeat_ms: int = 0,
    gpu_id: int = 0,
    run_smoke_test: bool = False,
    require_sm100: bool = True,
) -> str:
    metadata = _prepare_runtime(require_sm100=require_sm100)

    if run_smoke_test:
        _run([REMOTE_PYTHON, "tests/smoke/test_sparse_topk_forced.py"])

    specs = _benchmark_specs(
        preset=preset,
        dtype=dtype,
        sections=sections,
        output_mode=output_mode,
        seqs=seqs,
        tp=tp,
        decode_k=decode_k,
        decode_b=decode_b,
        topk=topk,
        head_dim=head_dim,
        blk_kv=blk_kv,
        dry_run_ms=dry_run_ms,
        repeat_ms=repeat_ms,
    )

    run_id = time.strftime("%Y%m%d-%H%M%S")
    result_dir = Path(MSA_CACHE_DIR) / "results" / run_id
    result_dir.mkdir(parents=True, exist_ok=True)

    run_records: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for spec in specs:
        tsv_path = result_dir / f"{spec.name}.tsv"
        log_path = result_dir / f"{spec.name}.log"
        cmd = _benchmark_cmd(spec, tsv_path, gpu_id)
        _run_to_file(cmd, log_path)

        rows = _parse_tsv(tsv_path)
        log_errors = _benchmark_log_errors(log_path)
        if not rows or log_errors:
            error_preview = "\n".join(log_errors[:20])
            raise RuntimeError(
                f"Benchmark {spec.name} did not complete cleanly. "
                f"Parsed rows: {len(rows)}. "
                f"TSV: {tsv_path}. Log: {log_path}."
                + (f"\nBenchmark errors:\n{error_preview}" if log_errors else "")
            )
        for row in rows:
            row.update(
                {
                    "run": spec.name,
                    "preset": preset,
                    "msa_commit": metadata["msa_commit"],
                    "modal_gpu_request": metadata["modal_gpu_request"],
                    "gpu_models": " + ".join(metadata["gpu_models"]),
                }
            )
        result_rows.extend(rows)
        run_records.append(
            {
                "name": spec.name,
                "dtype": spec.dtype,
                "sections": spec.sections,
                "command": cmd,
                "tsv_path": str(tsv_path),
                "log_path": str(log_path),
                "row_count": len(rows),
            }
        )

    _write_artifacts(
        result_dir=result_dir,
        metadata=metadata,
        run_records=run_records,
        result_rows=result_rows,
    )

    return (result_dir / "summary.md").read_text()


@app.function(
    image=image,
    gpu=modal_gpu_type,
    timeout=60 * 60,
    volumes={CACHE_ROOT: cache_volume},
)
def run_smoke(require_sm100: bool = True) -> str:
    metadata = _prepare_runtime(require_sm100=require_sm100)
    _run([REMOTE_PYTHON, "tests/smoke/test_sparse_topk_forced.py"])
    return json.dumps(metadata, indent=2)


@app.local_entrypoint()
def main(
    task: str = "benchmark",
    preset: str = "smoke",
    dtype: str = "",
    sections: str = "",
    output_mode: str = "o",
    seqs: str = "",
    tp: str = "",
    decode_k: str = "",
    decode_b: str = "",
    topk: int = 16,
    head_dim: int = 128,
    blk_kv: int = 128,
    dry_run_ms: int = 0,
    repeat_ms: int = 0,
    gpu_id: int = 0,
    run_smoke_test: bool = False,
    require_sm100: bool = True,
    msa_repo: str = msa_repo_url,
    msa_ref: str = msa_git_ref,
    modal_gpu: str = modal_gpu_type,
) -> None:
    """Run MiniMax-AI/MSA smoke tests or README-aligned benchmarks on Modal."""
    if msa_repo != msa_repo_url or msa_ref != msa_git_ref:
        print(
            "MSA source is fixed while the app is imported. "
            f"Using repo={msa_repo_url!r}, ref={msa_git_ref!r}.",
            flush=True,
        )
    if modal_gpu != modal_gpu_type:
        print(
            "The Modal GPU is fixed while the app is imported. "
            f"Using decorator GPU {modal_gpu_type!r}.",
            flush=True,
        )

    normalized_task = task.lower().replace("_", "-")
    if normalized_task == "smoke":
        print(run_smoke.remote(require_sm100=require_sm100))
        return
    if normalized_task not in {"benchmark", "bench"}:
        raise ValueError("task must be one of: smoke, benchmark")

    result = run_benchmark.remote(
        preset=preset,
        dtype=dtype,
        sections=sections,
        output_mode=output_mode,
        seqs=seqs,
        tp=tp,
        decode_k=decode_k,
        decode_b=decode_b,
        topk=topk,
        head_dim=head_dim,
        blk_kv=blk_kv,
        dry_run_ms=dry_run_ms,
        repeat_ms=repeat_ms,
        gpu_id=gpu_id,
        run_smoke_test=run_smoke_test,
        require_sm100=require_sm100,
    )
    print(result)
