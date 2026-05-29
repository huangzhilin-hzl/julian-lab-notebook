"""Modal runner for the TokenSpeed Qwen3.5 agentic benchmark.

Run from the lab-notebook repository:

    MODAL_GPU=B200:8 \
    MODAL_HF_SECRET_NAME=hf-secret \
    modal run tokenspeed-modal/modal_qwen3_5_agentic_perf.py

Use an 8x B200/GB200 environment and a TokenSpeed ref containing
test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic-b200-8gpu.yaml.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import modal


APP_NAME = "tokenspeed-qwen35-agentic-perf"
WORKDIR = "/workspace/tokenspeed"
REMOTE_PYTHON = "python"

MODEL = "nvidia/Qwen3.5-397B-A17B-NVFP4"
DEFAULT_TOKENSPEED_REPO = "https://github.com/lightseekorg/tokenspeed.git"
DEFAULT_TOKENSPEED_REF = "main"
DATASET_URL = (
    "https://huggingface.co/datasets/lightseekorg/agentic-dataset/"
    "resolve/main/agentic_dataset.json"
)


def _arg_value(flag: str) -> str | None:
    for idx, arg in enumerate(sys.argv):
        if arg == flag and idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        prefix = f"{flag}="
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def _clone_tokenspeed_command(repo_url: str, ref: str) -> str:
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


tokenspeed_repo_url = (
    _arg_value("--tokenspeed-repo")
    or os.environ.get("TOKENSPEED_REPO_URL")
    or DEFAULT_TOKENSPEED_REPO
)
tokenspeed_git_ref = (
    _arg_value("--tokenspeed-ref")
    or os.environ.get("TOKENSPEED_REF")
    or DEFAULT_TOKENSPEED_REF
)
gpu_type = _arg_value("--gpu") or os.environ.get("MODAL_GPU", "B200:8")
cache_volume_name = os.environ.get("MODAL_CACHE_VOLUME_NAME", "tokenspeed-cache")
cache_volume_version = int(os.environ.get("MODAL_CACHE_VOLUME_VERSION", "2"))
hf_secret_name = os.environ.get("MODAL_HF_SECRET_NAME", "")

function_secrets = [modal.Secret.from_name(hf_secret_name)] if hf_secret_name else []
cache_volume = modal.Volume.from_name(
    cache_volume_name,
    create_if_missing=True,
    version=cache_volume_version,
)
app = modal.App(APP_NAME)

image_env = {
    "PIP_BREAK_SYSTEM_PACKAGES": "1",
    "TOKENSPEED_KERNEL_BACKEND": "cuda",
    "FLASHINFER_CUDA_ARCH_LIST": os.environ.get("FLASHINFER_CUDA_ARCH_LIST", "10.0a"),
    "HF_HOME": "/cache/huggingface",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "MAX_JOBS": os.environ.get("MAX_JOBS", "16"),
    "PYTHONUNBUFFERED": "1",
    "TOKENSPEED_REPO_URL": tokenspeed_repo_url,
    "TOKENSPEED_REF": tokenspeed_git_ref,
    "MODAL_GPU_REQUEST": gpu_type,
}

image = (
    modal.Image.from_registry(
        "lightseekorg/tokenspeed-runner:latest",
        setup_dockerfile_commands=[
            "USER root",
            (
                "RUN set -eux; "
                "if ! command -v python >/dev/null 2>&1; then "
                'ln -sf "$(command -v python3)" /usr/local/bin/python; '
                "fi"
            ),
        ],
    )
    .apt_install("curl", "git", "libssl-dev", "libopenmpi-dev", "wget")
    .run_commands(_clone_tokenspeed_command(tokenspeed_repo_url, tokenspeed_git_ref))
    .env(image_env)
    .workdir(WORKDIR)
    .run_commands(
        (
            "python -m pip install --upgrade --ignore-installed "
            "pip setuptools wheel uv cmake ninja"
        ),
        (
            "python -m pip install tokenspeed-kernel/python/ "
            "--no-build-isolation -v"
        ),
        "python -m pip install tokenspeed-scheduler/",
        (
            "python -m pip install -e './python[cuda_sm100]' "
            "--extra-index-url https://download.pytorch.org/whl/cu130"
        ),
    )
)


def _run(cmd: list[str], *, cwd: str | None = None) -> None:
    print("$ " + shlex.join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _run_shell(command: str, *, cwd: str | None = None) -> None:
    print("$ " + command, flush=True)
    subprocess.run(["bash", "-o", "pipefail", "-lc", command], cwd=cwd, check=True)


def _run_to_file(cmd: list[str], log_path: Path, *, cwd: str | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    shell_cmd = f"{shlex.join(cmd)} 2>&1 | tee {shlex.quote(str(log_path))}"
    _run_shell(shell_cmd, cwd=cwd)


def _tail_file(path: Path, max_lines: int = 160) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _print_new_file_output(path: Path, offset: int) -> int:
    if not path.exists():
        return offset

    size = path.stat().st_size
    if size < offset:
        offset = 0
    if size == offset:
        return offset

    with path.open("rb") as file:
        file.seek(offset)
        chunk = file.read()

    if chunk:
        print(chunk.decode(errors="replace"), end="", flush=True)
    return size


def _wait_for_readiness(
    *, process: subprocess.Popen, port: int, timeout_s: int, server_log: Path
) -> None:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/readiness"
    log_offset = 0

    while time.time() < deadline:
        log_offset = _print_new_file_output(server_log, log_offset)

        if process.poll() is not None:
            _print_new_file_output(server_log, log_offset)
            raise RuntimeError(
                f"TokenSpeed server exited early with code {process.returncode}.\n"
                f"Last server log lines:\n{_tail_file(server_log)}"
            )

        try:
            with urlopen(url, timeout=5) as response:
                if response.status == 200:
                    _print_new_file_output(server_log, log_offset)
                    print(f"Server ready at {url}", flush=True)
                    return
        except (OSError, URLError):
            pass

        time.sleep(10)

    _print_new_file_output(server_log, log_offset)
    raise TimeoutError(
        f"TokenSpeed server did not become ready in {timeout_s}s.\n"
        f"Last server log lines:\n{_tail_file(server_log)}"
    )


def _json_list(value: str, flag_name: str) -> list[str]:
    if not value:
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"{flag_name} must be a JSON list of strings")
    return parsed


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=WORKDIR,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _prepare_evalscope() -> Path:
    evalscope_python = Path("/tmp/evalscope-perf/bin/python")
    _run(["python3", "-m", "uv", "venv", "--seed", "--clear", "/tmp/evalscope-perf"])
    _run(
        [
            "python3",
            "-m",
            "uv",
            "pip",
            "install",
            "--python",
            str(evalscope_python),
            "evalscope[perf]",
        ]
    )
    return Path("/tmp/evalscope-perf/bin/evalscope")


def _ensure_dataset() -> Path:
    dataset_path = Path("/cache/datasets/agentic_dataset.json")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset_path.exists():
        return dataset_path

    _run(["curl", "-fL", DATASET_URL, "-o", str(dataset_path)])
    return dataset_path


def _server_command(port: int, extra_serve_args_json: str) -> list[str]:
    cmd = [
        "ts",
        "serve",
        "--model",
        MODEL,
        "--attn-tp-size",
        "8",
        "--moe-tp-size",
        "8",
        "--gpu-memory-utilization",
        "0.6",
        "--max-num-seqs",
        "128",
        "--mamba-ssm-dtype",
        "float32",
        "--moe-backend",
        "flashinfer_trtllm",
        "--trust-remote-code",
        "--attention-backend",
        "trtllm",
        "--chunked-prefill-size",
        "2048",
        "--quantization",
        "nvfp4",
        "--kv-cache-dtype",
        "fp8",
        "--kvstore-ratio",
        "0.5",
        "--speculative-algorithm",
        "MTP",
        "--speculative-num-steps",
        "3",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    cmd.extend(_json_list(extra_serve_args_json, "extra_serve_args_json"))
    return cmd


def _evalscope_command(
    *,
    evalscope: Path,
    dataset_path: Path,
    outputs_dir: Path,
    port: int,
    number: str,
    parallel: str,
    dataset_offset: int | None,
    extra_perf_args_json: str,
) -> list[str]:
    cmd = [
        str(evalscope),
        "perf",
        "--model",
        MODEL,
        "--url",
        f"http://localhost:{port}/v1/chat/completions",
        "--api",
        "openai",
        "--dataset",
        "swe_smith",
        "--dataset-path",
        str(dataset_path),
        "--max-tokens",
        "500",
        "--multi-turn",
        "--number",
        *number.split(),
        "--parallel",
        *parallel.split(),
        "--outputs-dir",
        str(outputs_dir),
        "--extra-args",
        '{"ignore_eos": true}',
    ]
    if dataset_offset is not None:
        cmd.extend(["--dataset-offset", str(dataset_offset)])
    cmd.extend(_json_list(extra_perf_args_json, "extra_perf_args_json"))
    return cmd


@app.function(
    image=image,
    gpu=gpu_type,
    timeout=12 * 60 * 60,
    volumes={"/cache": cache_volume},
    secrets=function_secrets,
)
def qwen35_agentic_perf(
    *,
    port: int = 8000,
    ready_timeout_s: int = 1800,
    warmup_number: str = "2",
    warmup_parallel: str = "1",
    benchmark_number: str = "2",
    benchmark_parallel: str = "1",
    skip_warmup: bool = False,
    extra_serve_args_json: str = "",
    extra_perf_args_json: str = "",
) -> str:
    _run(["nvidia-smi"])

    import torch

    visible_gpus = torch.cuda.device_count()
    print(f"visible_cuda_devices={visible_gpus}", flush=True)
    if visible_gpus < 8:
        raise RuntimeError(
            "This reproduction target expects at least 8 visible B200/GB200 GPUs. "
            f"Only {visible_gpus} CUDA devices are visible."
        )

    run_id = time.strftime("%Y%m%d-%H%M%S")
    result_dir = Path("/cache/results/qwen3_5_agentic") / run_id
    result_dir.mkdir(parents=True, exist_ok=True)

    evalscope = _prepare_evalscope()
    dataset_path = _ensure_dataset()

    server_log = result_dir / "server.log"
    server_cmd = _server_command(port, extra_serve_args_json)
    run_config = {
        "model": MODEL,
        "tokenspeed_commit": _git_rev(),
        "gpu_request": gpu_type,
        "tokenspeed_repo": tokenspeed_repo_url,
        "tokenspeed_ref": tokenspeed_git_ref,
        "visible_cuda_devices": visible_gpus,
        "server_command": server_cmd,
        "warmup_number": warmup_number,
        "warmup_parallel": warmup_parallel,
        "benchmark_number": benchmark_number,
        "benchmark_parallel": benchmark_parallel,
        "dataset": str(dataset_path),
        "reference": {
            "latency_tps_per_user": 580,
            "throughput_tps_per_gpu": 4500,
            "source": (
                "test/ci/perf/"
                "qwen3.5-397b-a17b-nvfp4-evalscope-agentic-b200-8gpu.yaml"
            ),
        },
    }
    (result_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    print("$ " + shlex.join(server_cmd), flush=True)
    with server_log.open("w") as log_file:
        server = subprocess.Popen(
            server_cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=WORKDIR,
            start_new_session=True,
        )

    try:
        _wait_for_readiness(
            process=server,
            port=port,
            timeout_s=ready_timeout_s,
            server_log=server_log,
        )

        if not skip_warmup:
            warmup_dir = result_dir / "warmup"
            warmup_cmd = _evalscope_command(
                evalscope=evalscope,
                dataset_path=dataset_path,
                outputs_dir=warmup_dir,
                port=port,
                number=warmup_number,
                parallel=warmup_parallel,
                dataset_offset=68,
                extra_perf_args_json=extra_perf_args_json,
            )
            _run_to_file(warmup_cmd, result_dir / "warmup.log", cwd="/tmp")

        outputs_dir = result_dir / "outputs"
        perf_cmd = _evalscope_command(
            evalscope=evalscope,
            dataset_path=dataset_path,
            outputs_dir=outputs_dir,
            port=port,
            number=benchmark_number,
            parallel=benchmark_parallel,
            dataset_offset=None,
            extra_perf_args_json=extra_perf_args_json,
        )
        perf_cmd.append("--no-timestamp")
        _run_to_file(perf_cmd, result_dir / "benchmark.log", cwd="/tmp")

        default_evalscope_dir = outputs_dir / "Qwen3.5-397B-A17B-NVFP4"
        tp8_evalscope_dir = outputs_dir / "Qwen3.5-397B-A17B-NVFP4_attn_tp8"
        if default_evalscope_dir.exists() and not tp8_evalscope_dir.exists():
            default_evalscope_dir.rename(tp8_evalscope_dir)

        _run(
            [
                REMOTE_PYTHON,
                "test/agentic_benchmark/tokenspeed/collect_outputs.py",
                str(outputs_dir),
                "-o",
                str(result_dir / "sweep.csv"),
            ],
            cwd=WORKDIR,
        )
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=30)
        cache_volume.commit()

    print(f"Saved Qwen3.5 agentic perf artifacts to {result_dir}", flush=True)
    return str(result_dir)


@app.local_entrypoint()
def main(
    tokenspeed_repo: str = tokenspeed_repo_url,
    tokenspeed_ref: str = tokenspeed_git_ref,
    gpu: str = gpu_type,
    port: int = 8000,
    ready_timeout_s: int = 1800,
    warmup_number: str = "2",
    warmup_parallel: str = "1",
    benchmark_number: str = "2",
    benchmark_parallel: str = "1",
    skip_warmup: bool = False,
    extra_serve_args_json: str = "",
    extra_perf_args_json: str = "",
) -> None:
    if tokenspeed_repo != tokenspeed_repo_url or tokenspeed_ref != tokenspeed_git_ref:
        print(
            "TokenSpeed source is fixed while the app is imported. "
            f"Using repo={tokenspeed_repo_url!r}, ref={tokenspeed_git_ref!r}.",
            flush=True,
        )
    if gpu != gpu_type:
        print(
            "The Modal GPU is fixed while the app is imported. "
            f"Using decorator GPU {gpu_type!r}; pass --gpu before running if needed.",
            flush=True,
        )

    qwen35_agentic_perf.remote(
        port=port,
        ready_timeout_s=ready_timeout_s,
        warmup_number=warmup_number,
        warmup_parallel=warmup_parallel,
        benchmark_number=benchmark_number,
        benchmark_parallel=benchmark_parallel,
        skip_warmup=skip_warmup,
        extra_serve_args_json=extra_serve_args_json,
        extra_perf_args_json=extra_perf_args_json,
    )
