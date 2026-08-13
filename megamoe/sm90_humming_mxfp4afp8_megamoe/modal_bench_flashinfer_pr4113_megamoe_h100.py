#!/usr/bin/env python3
"""Run the main-branch PR4113 MegaMoE benchmark on 8 Modal H100 GPUs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import modal


APP_NAME = "flashinfer-pr4113-megamoe-h100"
FLASHINFER_REPO = "https://github.com/flashinfer-ai/flashinfer.git"
FLASHINFER_COMMIT = "28483960d7a56dd6a77e735f2c874b8e4dbd9d44"
NOTEBOOK_COMMIT = "982ac7335ad5eab144be381e2cc9aa972cd2d218"
BENCH_SHA256 = "45f8a57615c5d71c2adf416612fe51f009c43ba2fc84bc268f103338962edc16"
BENCH_URL = (
    "https://raw.githubusercontent.com/huangzhilin-hzl/julian-lab-notebook/"
    f"{NOTEBOOK_COMMIT}/megamoe/sm90_humming_mxfp4afp8_megamoe/"
    "bench_flashinfer_pr4113_megamoe_sm90.py"
)
SOURCE_DIR = Path("/opt/flashinfer-pr4113")
BENCH_PATH = Path("/opt/bench_flashinfer_pr4113_megamoe_sm90.py")
RESULTS_ROOT = Path("/results")
GPU_TYPE = "H100!:8"

app = modal.App(APP_NAME)
jit_volume = modal.Volume.from_name(
    "flashinfer-pr4113-jit-cache", create_if_missing=True
)
result_volume = modal.Volume.from_name(
    "flashinfer-pr4113-megamoe-results", create_if_missing=True
)

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu130", add_python=None)
    .env(
        {
            "FLASHINFER_WORKSPACE_BASE": "/cache/flashinfer",
            "FLASHINFER_CUDA_ARCH_LIST": "9.0a",
            "TORCH_CUDA_ARCH_LIST": "9.0a",
            "MAX_JOBS": "16",
            "PIP_BREAK_SYSTEM_PACKAGES": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "PYTHONPATH": str(SOURCE_DIR),
        }
    )
    .run_commands(
        f"mkdir -p {SOURCE_DIR}",
        f"git init {SOURCE_DIR}",
        f"git -C {SOURCE_DIR} remote add origin {FLASHINFER_REPO}",
        f"git -C {SOURCE_DIR} fetch --depth=1 origin {FLASHINFER_COMMIT}",
        f"git -C {SOURCE_DIR} switch --detach FETCH_HEAD",
        f"git -C {SOURCE_DIR} submodule update --init --recursive",
        f"curl -fsSL {BENCH_URL} -o {BENCH_PATH}",
    )
)


def _prepare_source_tree() -> None:
    actual_bench_hash = hashlib.sha256(BENCH_PATH.read_bytes()).hexdigest()
    if actual_bench_hash != BENCH_SHA256:
        raise RuntimeError(
            f"bench SHA256 mismatch: expected {BENCH_SHA256}, got {actual_bench_hash}"
        )

    build_meta = SOURCE_DIR / "flashinfer" / "_build_meta.py"
    if not build_meta.exists():
        version_file = SOURCE_DIR / "version.txt"
        version = version_file.read_text().strip() if version_file.exists() else "unknown"
        build_meta.write_text(
            '"""Build metadata for the Modal source checkout."""\n'
            f'__version__ = "{version}+pr4113"\n'
            f'__git_version__ = "{FLASHINFER_COMMIT}"\n'
        )

    data_dir = SOURCE_DIR / "flashinfer" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "csrc": SOURCE_DIR / "csrc",
        "include": SOURCE_DIR / "include",
        "cutlass": SOURCE_DIR / "3rdparty" / "cutlass",
        "cccl": SOURCE_DIR / "3rdparty" / "cccl",
        "spdlog": SOURCE_DIR / "3rdparty" / "spdlog",
    }
    for name, source in sources.items():
        if not source.exists():
            raise RuntimeError(f"required FlashInfer source path is missing: {source}")
        target = data_dir / name
        if not target.exists() and not target.is_symlink():
            target.symlink_to(source, target_is_directory=True)


def _run_logged(
    command: list[str],
    log_path: Path,
    *,
    append: bool = False,
    quiet_rank_json: bool = False,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    rendered = shlex.join(command)
    print(f"$ {rendered}", flush=True)
    with log_path.open(mode, encoding="utf-8") as log_file:
        log_file.write(f"$ {rendered}\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=str(SOURCE_DIR),
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            if quiet_rank_json and (
                "FLASHINFER_4113_STAT_JSON" in line
                or "FLASHINFER_4113_OBSERVATION_JSON" in line
            ):
                continue
            print(line, end="", flush=True)
        return process.wait()


def _collect_environment(log_path: Path) -> None:
    commands = [
        (["nvidia-smi", "-L"], True),
        (["nvidia-smi", "topo", "-m"], False),
        (
            [
                sys.executable,
                "-c",
                (
                    "import json, torch; "
                    "print(json.dumps({"
                    "'python': __import__('sys').version, "
                    "'torch': torch.__version__, "
                    "'cuda': torch.version.cuda, "
                    "'gpu_count': torch.cuda.device_count(), "
                    "'gpus': [torch.cuda.get_device_name(i) "
                    "for i in range(torch.cuda.device_count())]}))"
                ),
            ],
            True,
        ),
        ([sys.executable, "-m", "flashinfer", "show-config"], True),
        (["git", "rev-parse", "HEAD"], True),
    ]
    for index, (command, required) in enumerate(commands):
        return_code = _run_logged(command, log_path, append=index > 0)
        if required and return_code != 0:
            raise RuntimeError(f"environment command failed: {shlex.join(command)}")


def _parse_benchmark_log(log_path: Path) -> dict[str, object]:
    prefixes = {
        "plan": "FLASHINFER_4113_PLAN_JSON ",
        "stats": "FLASHINFER_4113_STAT_JSON ",
        "observations": "FLASHINFER_4113_OBSERVATION_JSON ",
        "summaries": "FLASHINFER_4113_SUMMARY_JSON ",
        "failed_points": "FLASHINFER_4113_POINT_JSON ",
    }
    parsed: dict[str, object] = {
        "plan": None,
        "stats": [],
        "observations": [],
        "summaries": [],
        "failed_points": [],
    }
    contents = log_path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    for key, prefix in prefixes.items():
        search_from = 0
        while True:
            marker = contents.find(prefix, search_from)
            if marker < 0:
                break
            value, end = decoder.raw_decode(contents, marker + len(prefix))
            if key == "plan":
                parsed[key] = value
            else:
                assert isinstance(parsed[key], list)
                parsed[key].append(value)
            search_from = end
    return parsed


def _write_summary_csv(parsed: dict[str, object], output_path: Path) -> None:
    summaries = parsed["summaries"]
    assert isinstance(summaries, list)
    fieldnames = [
        "model",
        "m",
        "cap",
        "max_rank_median_us",
        "max_rank_min_us",
        "max_rank_max_us",
        "tokens_per_rank_s",
        "observations",
        "num_tests",
        "world_size",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    **{key: summary[key] for key in fieldnames if key != "tokens_per_rank_s"},
                    "tokens_per_rank_s": (
                        summary["m"] * 1_000_000 / summary["max_rank_median_us"]
                    ),
                }
            )


def _safe_run_id(task: str, run_name: str) -> str:
    value = run_name or f"pr4113-h100-{task}-{time.strftime('%Y%m%d-%H%M%S')}"
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError("run_name may contain only letters, digits, '.', '_', and '-'")
    return value


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=12 * 60 * 60,
    memory=128 * 1024,
    volumes={
        "/cache/flashinfer": jit_volume,
        str(RESULTS_ROOT): result_volume,
    },
)
def run_benchmark(task: str = "smoke", run_name: str = "") -> dict[str, object]:
    if task not in {"smoke", "full"}:
        raise ValueError("task must be 'smoke' or 'full'")

    _prepare_source_tree()
    run_id = _safe_run_id(task, run_name)
    result_dir = RESULTS_ROOT / run_id
    result_dir.mkdir(parents=True, exist_ok=False)
    environment_log = result_dir / "environment.log"
    benchmark_log = result_dir / f"{task}.log"
    parsed_path = result_dir / "parsed.json"
    summary_csv_path = result_dir / "summary.csv"
    metadata_path = result_dir / "metadata.json"

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        str(BENCH_PATH),
    ]
    if task == "smoke":
        command.extend(
            [
                "--model-config",
                "flash",
                "--batches",
                "8",
                "--observations",
                "1",
                "--warmups",
                "1",
                "--num-tests",
                "2",
                "--flush-l2-bytes",
                "100000000",
            ]
        )
    else:
        command.append("--continue-on-error")

    metadata: dict[str, object] = {
        "app": APP_NAME,
        "task": task,
        "run_id": run_id,
        "gpu_request": GPU_TYPE,
        "flashinfer_commit": FLASHINFER_COMMIT,
        "notebook_commit": NOTEBOOK_COMMIT,
        "bench_sha256": BENCH_SHA256,
        "command": command,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        _collect_environment(environment_log)
        return_code = _run_logged(
            command, benchmark_log, quiet_rank_json=task == "full"
        )
        parsed = _parse_benchmark_log(benchmark_log)
        parsed_path.write_text(json.dumps(parsed, indent=2, sort_keys=True) + "\n")
        _write_summary_csv(parsed, summary_csv_path)
        metadata.update(
            {
                "return_code": return_code,
                "summary_count": len(parsed["summaries"]),
                "observation_count": len(parsed["observations"]),
                "rank_stat_count": len(parsed["stats"]),
                "failed_point_count": len(parsed["failed_points"]),
                "finished_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        if return_code != 0 or parsed["failed_points"]:
            raise RuntimeError(
                f"benchmark failed: return_code={return_code}, "
                f"failed_points={len(parsed['failed_points'])}"
            )
        return metadata
    finally:
        if not metadata_path.exists():
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        jit_volume.commit()
        result_volume.commit()


@app.local_entrypoint()
def main(task: str = "smoke", run_name: str = "") -> None:
    result = run_benchmark.remote(task=task, run_name=run_name)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(
        "Download with: "
        "uvx modal volume get flashinfer-pr4113-megamoe-results "
        f"/{result['run_id']} modal-results/pr4113-h100/",
        flush=True,
    )
