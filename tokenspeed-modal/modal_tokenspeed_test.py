"""Modal app for TokenSpeed smoke tests and serving benchmarks.

Run from the lab-notebook repository, for example:

    TOKENSPEED_SRC_DIR=/path/to/tokenspeed \
    MODAL_GPU=B200 \
    modal run tokenspeed-modal/modal_tokenspeed_test.py --task smoke

    modal run tokenspeed-modal/modal_tokenspeed_test.py --task kernel --kernel-mode cuda
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import modal


APP_NAME = "tokenspeed-test"
WORKDIR = "/workspace/tokenspeed"
TOKENSPEED_SRC_ENV = "TOKENSPEED_SRC_DIR"

tokenspeed_src_raw = os.environ.get(TOKENSPEED_SRC_ENV)
if not tokenspeed_src_raw:
    raise RuntimeError(f"Set {TOKENSPEED_SRC_ENV} to the local TokenSpeed checkout path.")

tokenspeed_src = Path(tokenspeed_src_raw).expanduser()
if not tokenspeed_src.exists():
    raise RuntimeError(
        f"TokenSpeed checkout does not exist: {tokenspeed_src}. "
        f"Set {TOKENSPEED_SRC_ENV} to the local checkout path."
    )

gpu_type = os.environ.get("MODAL_GPU", "B200")
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
    "FLASHINFER_CUDA_ARCH_LIST": os.environ.get(
        "FLASHINFER_CUDA_ARCH_LIST", "9.0a 10.0a"
    ),
    "HF_HOME": "/cache/huggingface",
    "MAX_JOBS": os.environ.get("MAX_JOBS", "16"),
}
if os.environ.get("TOKENSPEED_MLA_FMHA_BINARY_SO"):
    image_env["TOKENSPEED_MLA_FMHA_BINARY_SO"] = os.environ[
        "TOKENSPEED_MLA_FMHA_BINARY_SO"
    ]

image = (
    modal.Image.from_registry(
        "lightseekorg/tokenspeed-runner:latest",
        setup_dockerfile_commands=["USER root"],
    )
    .apt_install("libssl-dev", "libopenmpi-dev")
    .apt_install("wget")
    .add_local_dir(
        str(tokenspeed_src),
        remote_path=WORKDIR,
        copy=True,
        ignore=[
            ".git",
            ".git/**",
            ".mypy_cache",
            ".mypy_cache/**",
            ".pytest_cache",
            ".pytest_cache/**",
            ".ruff_cache",
            ".ruff_cache/**",
            ".venv",
            ".venv/**",
            "**/__pycache__/**",
            "**/*.pyc",
            "build",
            "build/**",
            "dist",
            "dist/**",
        ],
    )
    .env(image_env)
    .workdir(WORKDIR)
    .run_commands(
        "python -m pip install -e tokenspeed-kernel/python/ --no-build-isolation",
        "python -m pip install -e tokenspeed-scheduler/",
        "python -m pip install -e ./python --no-build-isolation",
    )
)


def _run(cmd: list[str], *, cwd: str | None = None) -> None:
    print("$ " + shlex.join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _run_to_file(cmd: list[str], log_path: Path, *, cwd: str | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    shell_cmd = f"{shlex.join(cmd)} 2>&1 | tee {shlex.quote(str(log_path))}"
    print("$ " + shell_cmd, flush=True)
    subprocess.run(["bash", "-o", "pipefail", "-lc", shell_cmd], cwd=cwd, check=True)


def _json_list(value: str, flag_name: str) -> list[str]:
    if not value:
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) for item in parsed
    ):
        raise ValueError(f"{flag_name} must be a JSON list of strings")
    return parsed


def _tail_file(path: Path, max_lines: int = 120) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _wait_for_server(
    *,
    process: subprocess.Popen,
    port: int,
    timeout_s: int,
    server_log: Path,
) -> None:
    import requests

    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/v1/models"

    while time.time() < deadline:
        if process.poll() is not None:
            tail = _tail_file(server_log)
            raise RuntimeError(
                f"TokenSpeed server exited early with code {process.returncode}.\n"
                f"Last server log lines:\n{tail}"
            )

        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(response.text, flush=True)
                return
        except requests.RequestException:
            pass

        time.sleep(5)

    tail = _tail_file(server_log)
    raise TimeoutError(
        f"TokenSpeed server did not become ready in {timeout_s}s.\n"
        f"Last server log lines:\n{tail}"
    )


@app.function(
    image=image,
    gpu=gpu_type,
    timeout=60 * 60,
    volumes={"/cache": cache_volume},
    secrets=function_secrets,
)
def smoke() -> None:
    _run(["nvidia-smi"])
    _run(["tokenspeed", "env"])
    _run(["tokenspeed", "serve", "--help"])


@app.function(
    image=image,
    gpu=gpu_type,
    timeout=3 * 60 * 60,
    volumes={"/cache": cache_volume},
    secrets=function_secrets,
)
def serve_and_bench(
    model: str = "openai/gpt-oss-20b",
    served_model_name: str = "gpt-oss-20b",
    tensor_parallel_size: int = 1,
    max_model_len: int = 131072,
    chunked_prefill_size: int = 8192,
    num_prompts: int = 32,
    input_len: int = 1024,
    output_len: int = 128,
    request_rate: str = "inf",
    ready_timeout_s: int = 900,
    port: int = 8000,
    reasoning_parser: str = "gpt-oss",
    tool_call_parser: str = "gpt-oss",
    extra_serve_args_json: str = "",
    extra_bench_args_json: str = "",
) -> str:
    if not served_model_name:
        served_model_name = model.rstrip("/").split("/")[-1]

    run_id = time.strftime("%Y%m%d-%H%M%S")
    result_dir = Path("/cache/results") / run_id
    result_dir.mkdir(parents=True, exist_ok=True)
    server_log = result_dir / "server.log"

    server_cmd = [
        "tokenspeed",
        "serve",
        model,
        "--served-model-name",
        served_model_name,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--max-model-len",
        str(max_model_len),
        "--chunked-prefill-size",
        str(chunked_prefill_size),
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]
    if reasoning_parser:
        server_cmd.extend(["--reasoning-parser", reasoning_parser])
    if tool_call_parser:
        server_cmd.extend(["--tool-call-parser", tool_call_parser])
    server_cmd.extend(_json_list(extra_serve_args_json, "extra_serve_args_json"))

    print("$ " + shlex.join(server_cmd), flush=True)
    with server_log.open("w") as log_file:
        server = subprocess.Popen(
            server_cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=WORKDIR,
        )

        try:
            _wait_for_server(
                process=server,
                port=port,
                timeout_s=ready_timeout_s,
                server_log=server_log,
            )

            bench_cmd = [
                "tokenspeed",
                "bench",
                "serve",
                "--backend",
                "openai",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--model",
                served_model_name,
                "--dataset-name",
                "random",
                "--num-prompts",
                str(num_prompts),
                "--input-len",
                str(input_len),
                "--output-len",
                str(output_len),
                "--request-rate",
                request_rate,
                "--disable-tqdm",
                "--save-result",
                "--result-dir",
                str(result_dir),
            ]
            bench_cmd.extend(
                _json_list(extra_bench_args_json, "extra_bench_args_json")
            )
            _run(bench_cmd, cwd=WORKDIR)
            print(f"Saved results to {result_dir}", flush=True)
            return str(result_dir)
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=30)
            cache_volume.commit()


@app.function(
    image=image,
    gpu=gpu_type,
    timeout=3 * 60 * 60,
    volumes={"/cache": cache_volume},
    secrets=function_secrets,
)
def kernel_tests(
    mode: str = "ci",
    target: str = "",
    keyword: str = "",
    extra_pytest_args_json: str = "",
) -> None:
    _run(["nvidia-smi"])

    extra_args = _json_list(extra_pytest_args_json, "extra_pytest_args_json")

    if target:
        cmd = ["python", "-m", "pytest", target, "-v"]
        if keyword:
            cmd.extend(["-k", keyword])
        cmd.extend(extra_args)
        _run(cmd, cwd=WORKDIR)
        return

    if mode == "numerics":
        commands = [
            ["python", "-m", "pytest", "tokenspeed-kernel/test/test_numerics.py", "-v"]
        ]
    elif mode == "cuda":
        commands = [
            [
                "python",
                "-m",
                "pytest",
                "tokenspeed-kernel/test/thirdparty/test_cuda.py",
                "-v",
            ]
        ]
    elif mode == "ops":
        commands = [
            ["python", "-m", "pytest", "tokenspeed-kernel/test/ops/", "-v"]
        ]
    elif mode == "ci":
        commands = [
            [
                "python",
                "-m",
                "pytest",
                "tokenspeed-kernel/test/test_numerics.py",
                "-v",
            ],
            [
                "python",
                "-m",
                "pytest",
                "tokenspeed-kernel/test/thirdparty/test_trtllm_comm.py",
                "-v",
            ],
            [
                "python",
                "-m",
                "pytest",
                "tokenspeed-kernel/test/thirdparty/test_cuda.py",
                "-v",
            ],
            [
                "python",
                "-m",
                "pytest",
                "tokenspeed-kernel/test/",
                "-v",
                "--ignore=tokenspeed-kernel/test/test_numerics.py",
                "--ignore=tokenspeed-kernel/test/thirdparty/test_trtllm_comm.py",
                "--ignore=tokenspeed-kernel/test/thirdparty/test_cuda.py",
            ],
        ]
    elif mode == "all":
        commands = [["python", "-m", "pytest", "tokenspeed-kernel/test/", "-v"]]
    else:
        raise ValueError("kernel mode must be one of: ci, numerics, cuda, ops, all")

    for cmd in commands:
        _run([*cmd, *extra_args], cwd=WORKDIR)


@app.function(
    image=image,
    gpu=gpu_type,
    timeout=12 * 60 * 60,
    volumes={"/cache": cache_volume},
    secrets=function_secrets,
)
def agentic_benchmark() -> str:
    _run(["nvidia-smi"])

    run_id = time.strftime("%Y%m%d-%H%M%S")
    result_dir = Path("/cache/results/agentic") / run_id
    result_dir.mkdir(parents=True, exist_ok=True)

    bench_dir = Path(WORKDIR) / "test/agentic_benchmark/tokenspeed"
    try:
        _run(["bash", "agentic_bench.sh"], cwd=str(bench_dir))
    finally:
        outputs_dir = bench_dir / "outputs"
        if outputs_dir.exists():
            shutil.copytree(outputs_dir, result_dir / "outputs", dirs_exist_ok=True)

        logs_dir = result_dir / "server_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        for log_path in Path("/tmp").glob("tokenspeed_server_*.log"):
            shutil.copy2(log_path, logs_dir / log_path.name)

        latest_outputs = sorted(outputs_dir.iterdir()) if outputs_dir.exists() else []
        if latest_outputs:
            latest = latest_outputs[-1]
            csv_path = result_dir / f"{latest.name}.csv"
            _run(
                [
                    "python3",
                    "collect_outputs.py",
                    f"outputs/{latest.name}",
                    "-o",
                    str(csv_path),
                ],
                cwd=str(bench_dir),
            )
            shutil.copy2(csv_path, result_dir / "sweep.csv")

        cache_volume.commit()

    print(f"Saved agentic benchmark artifacts to {result_dir}", flush=True)
    return str(result_dir)


@app.function(
    image=image,
    gpu=gpu_type,
    timeout=4 * 60 * 60,
    volumes={"/cache": cache_volume},
    secrets=function_secrets,
)
def mla_benchmark(mode: str = "both", prefill_backend: str = "cutedsl") -> str:
    if mode not in {"both", "prefill", "decode"}:
        raise ValueError("mla mode must be one of: both, prefill, decode")
    if prefill_backend not in {"cutedsl", "binary"}:
        raise ValueError("prefill_backend must be either 'cutedsl' or 'binary'")

    _run(["nvidia-smi"])
    os.environ["TOKENSPEED_MLA_PREFILL_BACKEND"] = prefill_backend

    run_id = time.strftime("%Y%m%d-%H%M%S")
    result_dir = Path("/cache/results/mla") / run_id
    result_dir.mkdir(parents=True, exist_ok=True)

    commands: list[tuple[str, list[str]]] = []

    if mode in {"both", "prefill"}:
        prefill_cases = [
            ("prefill_b1_q8k_k8k", "1,8192,128,192", "1,8192,128,192"),
            ("prefill_b1_q8k_k32k", "1,8192,128,192", "1,32768,128,192"),
            ("prefill_b1_q8k_k64k", "1,8192,128,192", "1,65536,128,192"),
            ("prefill_b4_q512_k80k", "4,512,128,192", "4,81920,128,192"),
            ("prefill_b4_q1024_k80k", "4,1024,128,192", "4,81920,128,192"),
        ]
        for name, q_shape, k_shape in prefill_cases:
            commands.append(
                (
                    name,
                    [
                        "python",
                        "./tokenspeed-mla/python/tokenspeed_mla/fmha.py",
                        "--is_causal",
                        "--bottom_right_align",
                        "--in_dtype",
                        "Float8E4M3FN",
                        "--out_dtype",
                        "Float8E4M3FN",
                        "--q_shape",
                        q_shape,
                        "--k_shape",
                        k_shape,
                        "--warmup_iterations",
                        "10",
                        "--iterations",
                        "10",
                        "--skip_ref_check",
                    ],
                )
            )

    if mode in {"both", "decode"}:
        decode_cases = [
            ("decode_b4_h16", 4, 16),
            ("decode_b8_h16", 8, 16),
            ("decode_b16_h16", 16, 16),
            ("decode_b4_h32", 4, 32),
            ("decode_b8_h32", 8, 32),
            ("decode_b16_h32", 16, 32),
        ]
        for name, batch_size, num_heads in decode_cases:
            commands.append(
                (
                    name,
                    [
                        "python",
                        "./tokenspeed-mla/python/tokenspeed_mla/mla_decode_fp8.py",
                        "--batch_size",
                        str(batch_size),
                        "--softmax_scale",
                        "0.07216882",
                        "--page_size",
                        "64",
                        "--seq_len_k",
                        "81920",
                        "--in_dtype",
                        "Float8E4M3FN",
                        "--out_dtype",
                        "Float8E4M3FN",
                        "--seq_len_q",
                        "4",
                        "--warmup_iterations",
                        "1",
                        "--iterations",
                        "10",
                        "--num_heads",
                        str(num_heads),
                        "--skip_ref_check",
                    ],
                )
            )

    try:
        for name, cmd in commands:
            _run_to_file(cmd, result_dir / f"{name}.log", cwd=WORKDIR)
    finally:
        cache_volume.commit()

    print(f"Saved MLA benchmark logs to {result_dir}", flush=True)
    return str(result_dir)


@app.local_entrypoint()
def main(
    task: str = "smoke",
    model: str = "openai/gpt-oss-20b",
    served_model_name: str = "gpt-oss-20b",
    tensor_parallel_size: int = 1,
    max_model_len: int = 131072,
    chunked_prefill_size: int = 8192,
    num_prompts: int = 32,
    input_len: int = 1024,
    output_len: int = 128,
    request_rate: str = "inf",
    ready_timeout_s: int = 900,
    port: int = 8000,
    reasoning_parser: str = "gpt-oss",
    tool_call_parser: str = "gpt-oss",
    extra_serve_args_json: str = "",
    extra_bench_args_json: str = "",
    kernel_mode: str = "ci",
    kernel_target: str = "",
    kernel_keyword: str = "",
    extra_pytest_args_json: str = "",
    mla_mode: str = "both",
    mla_prefill_backend: str = "cutedsl",
) -> None:
    if task == "smoke":
        smoke.remote()
        return

    if task == "kernel":
        kernel_tests.remote(
            mode=kernel_mode,
            target=kernel_target,
            keyword=kernel_keyword,
            extra_pytest_args_json=extra_pytest_args_json,
        )
        return

    if task == "agentic":
        agentic_benchmark.remote()
        return

    if task == "mla":
        mla_benchmark.remote(mode=mla_mode, prefill_backend=mla_prefill_backend)
        return

    if task == "bench":
        serve_and_bench.remote(
            model=model,
            served_model_name=served_model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            chunked_prefill_size=chunked_prefill_size,
            num_prompts=num_prompts,
            input_len=input_len,
            output_len=output_len,
            request_rate=request_rate,
            ready_timeout_s=ready_timeout_s,
            port=port,
            reasoning_parser=reasoning_parser,
            tool_call_parser=tool_call_parser,
            extra_serve_args_json=extra_serve_args_json,
            extra_bench_args_json=extra_bench_args_json,
        )
        return

    raise ValueError("task must be one of: smoke, kernel, agentic, mla, bench")
