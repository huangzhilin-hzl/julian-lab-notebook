"""Run Prime Flash MoE's default benchmark on one Modal B200.

Usage:

    uvx modal run prime-flash-moe/modal_benchmark.py

The remote function clones the pinned upstream revision, builds ``prime_moe``
for SM100, runs ``python benchmark/benchmark.py`` with its default arguments,
and returns the CSV, plots, log, and environment metadata to a timestamped
local directory under ``prime-flash-moe/results``.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal


APP_NAME = "prime-flash-moe-benchmark"
REPO_URL = "https://github.com/PrimeIntellect-ai/prime-flash-moe.git"
REPO_REF = "1820183d63eed79fd166fbf4b81cae2b27b326c2"
WORKDIR = "/workspace/prime-flash-moe"
BASE_IMAGE = os.environ.get(
    "PRIME_FLASH_MOE_BASE_IMAGE", "nvcr.io/nvidia/pytorch:26.07-py3"
)
GPU_TYPE = os.environ.get("MODAL_GPU", "B200")

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry(BASE_IMAGE, add_python=None)
    .apt_install("git", "build-essential", "ninja-build")
    .env(
        {
            "TORCH_CUDA_ARCH_LIST": "10.0a",
            "MAX_JOBS": "16",
            "MPLBACKEND": "Agg",
            "PIP_BREAK_SYSTEM_PACKAGES": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .run_commands(
        f"git clone {shlex.quote(REPO_URL)} {shlex.quote(WORKDIR)}",
        f"git -C {shlex.quote(WORKDIR)} checkout --detach {shlex.quote(REPO_REF)}",
        "python -m pip install matplotlib ninja",
        f"cd {shlex.quote(WORKDIR)} && python -m pip install -e . --no-build-isolation",
    )
    .workdir(WORKDIR)
)


def _capture(command: list[str]) -> str:
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


@app.function(image=image, gpu=GPU_TYPE, timeout=60 * 60)
def run_benchmark() -> dict[str, object]:
    import torch

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_dir = Path("/tmp") / f"prime-flash-moe-{run_id}"
    output_dir.mkdir(parents=True)
    log_path = output_dir / "benchmark.log"

    command = [
        "python",
        "benchmark/benchmark.py",
        "--outdir",
        str(output_dir),
    ]
    metadata = {
        "run_id_utc": run_id,
        "command": shlex.join(command),
        "repository": REPO_URL,
        "repository_commit": _capture(["git", "rev-parse", "HEAD"]),
        "modal_gpu_request": GPU_TYPE,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "compute_capability": ".".join(
            str(part) for part in torch.cuda.get_device_capability(0)
        ),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "python": _capture(["python", "--version"]),
        "nvcc": _capture(["nvcc", "--version"]),
        "nvidia_smi": _capture(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.memory",
                "--format=csv,noheader",
            ]
        ),
        "base_image": BASE_IMAGE,
    }
    (output_dir / "environment.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("$ " + shlex.join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=WORKDIR,
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

    artifacts = {
        path.name: path.read_bytes()
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    return {
        "run_id": run_id,
        "return_code": return_code,
        "metadata": metadata,
        "artifacts": artifacts,
    }


@app.local_entrypoint()
def main(output_root: str = "prime-flash-moe/results") -> None:
    result = run_benchmark.remote()
    run_dir = Path(output_root) / str(result["run_id"])
    run_dir.mkdir(parents=True, exist_ok=False)
    for filename, content in result["artifacts"].items():
        (run_dir / filename).write_bytes(content)

    manifest = {
        "run_id": result["run_id"],
        "return_code": result["return_code"],
        "files": sorted(result["artifacts"]),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"local_result_dir": str(run_dir), **manifest}, indent=2))

    if result["return_code"] != 0:
        raise SystemExit(int(result["return_code"]))
