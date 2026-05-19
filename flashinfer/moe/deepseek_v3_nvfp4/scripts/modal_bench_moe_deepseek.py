"""Modal runner for FlashInfer's DeepSeek-V3 NVFP4 MoE benchmark.

Run from the lab-notebook repository:

    FLASHINFER_SRC_DIR=/path/to/flashinfer \
    MODAL_GPU=B200 \
    modal run flashinfer/moe/deepseek_v3_nvfp4/scripts/modal_bench_moe_deepseek.py

Generation-phase run:

    FLASHINFER_SRC_DIR=/path/to/flashinfer \
    MODAL_GPU=B200 \
    modal run flashinfer/moe/deepseek_v3_nvfp4/scripts/modal_bench_moe_deepseek.py \
      --gen-phase --ep 8 --warmup 5 --iters 30

The wrapped benchmark requires an SM100-family GPU, so the default Modal GPU is
B200. Benchmark logs are written to the Modal Volume path printed at the end.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal


APP_NAME = "flashinfer-deepseek-moe-bench"
WORKDIR = "/workspace/flashinfer"
FLASHINFER_SRC_ENV = "FLASHINFER_SRC_DIR"
REMOTE_PYTHON = "python"
IS_MODAL_REMOTE = os.environ.get("MODAL_IS_REMOTE") == "1"


def _arg_value(flag: str) -> str | None:
    for idx, arg in enumerate(sys.argv):
        if arg == flag and idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        prefix = f"{flag}="
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


flashinfer_src_raw = os.environ.get(FLASHINFER_SRC_ENV)
if flashinfer_src_raw:
    flashinfer_src = Path(flashinfer_src_raw).expanduser()
elif IS_MODAL_REMOTE:
    flashinfer_src = Path(WORKDIR)
else:
    raise RuntimeError(f"Set {FLASHINFER_SRC_ENV} to the local FlashInfer checkout path.")

if not IS_MODAL_REMOTE and not flashinfer_src.exists():
    raise RuntimeError(
        f"The path configured by {FLASHINFER_SRC_ENV} does not exist."
    )

gpu_type = (
    _arg_value("--gpu")
    or os.environ.get("MODAL_GPU")
    or "B200"
)
cache_volume_name = os.environ.get(
    "MODAL_CACHE_VOLUME_NAME", "flashinfer-jit-cache"
)

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name(cache_volume_name, create_if_missing=True)

image = (
    modal.Image.from_registry(
        "flashinfer/flashinfer-ci-cu130",
        add_python=None,
    )
    .add_local_dir(
        str(flashinfer_src),
        remote_path=WORKDIR,
        copy=True,
        ignore=[
            ".git",
            ".git/**",
            ".cache",
            ".cache/**",
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
            "**/*.so",
            "*.egg-info",
            "*.egg-info/**",
            "benchmark_runs",
            "benchmark_runs/**",
            "build",
            "build/**",
            "dist",
            "dist/**",
        ],
    )
    .env(
        {
            "FLASHINFER_WORKSPACE_BASE": "/cache/flashinfer",
            "FLASHINFER_CUDA_ARCH_LIST": os.environ.get(
                "FLASHINFER_CUDA_ARCH_LIST", "10.0a"
            ),
            "MAX_JOBS": os.environ.get("MAX_JOBS", "16"),
            "PIP_BREAK_SYSTEM_PACKAGES": "1",
        }
    )
    .workdir(WORKDIR)
)


def _run(cmd: list[str], *, cwd: str = WORKDIR) -> None:
    print("$ " + shlex.join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _run_to_file(cmd: list[str], log_path: Path, *, cwd: str = WORKDIR) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    shell_cmd = f"{shlex.join(cmd)} 2>&1 | tee {shlex.quote(str(log_path))}"
    print("$ " + shell_cmd, flush=True)
    subprocess.run(
        ["bash", "-o", "pipefail", "-lc", shell_cmd],
        cwd=cwd,
        check=True,
    )


def _ensure_build_meta() -> None:
    build_meta_path = Path(WORKDIR) / "flashinfer" / "_build_meta.py"
    if build_meta_path.exists():
        return

    version_file = Path(WORKDIR) / "version.txt"
    version = (
        version_file.read_text().strip() if version_file.exists() else "0.0.0+modal"
    )
    build_meta_path.write_text(
        '"""Build metadata for flashinfer package."""\n'
        f'__version__ = "{version}"\n'
        '__git_version__ = "modal"\n'
    )
    print(f"Generated {build_meta_path} with version {version}", flush=True)


def _ensure_submodule(path: str, expected_child: str, url: str) -> None:
    full_path = Path(WORKDIR) / path
    if (full_path / expected_child).exists():
        return

    print(f"Downloading missing dependency {path}", flush=True)
    if full_path.exists():
        shutil.rmtree(full_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth=1", url, str(full_path)],
        cwd=WORKDIR,
        check=True,
    )


def _ensure_in_tree_package_data_layout() -> None:
    """Mirror package-data paths expected by JIT when importing from source."""
    data_dir = Path(WORKDIR) / "flashinfer" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for name, source in {
        "csrc": Path(WORKDIR) / "csrc",
        "include": Path(WORKDIR) / "include",
        "cutlass": Path(WORKDIR) / "3rdparty" / "cutlass",
        "spdlog": Path(WORKDIR) / "3rdparty" / "spdlog",
        "cccl": Path(WORKDIR) / "3rdparty" / "cccl",
    }.items():
        target = data_dir / name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(source, target_is_directory=True)


def _prepare_flashinfer_source() -> None:
    os.chdir(WORKDIR)
    pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = (
        f"{WORKDIR}:{pythonpath}" if pythonpath else WORKDIR
    )

    _ensure_build_meta()
    _ensure_submodule(
        "3rdparty/cutlass", "include", "https://github.com/NVIDIA/cutlass.git"
    )
    _ensure_submodule(
        "3rdparty/spdlog", "include", "https://github.com/gabime/spdlog.git"
    )
    _ensure_submodule("3rdparty/cccl", "cub", "https://github.com/NVIDIA/cccl.git")
    _ensure_in_tree_package_data_layout()


def _benchmark_cmd(
    *,
    num_tokens: str,
    warmup: int,
    iters: int,
    no_autotune: bool,
    quiet: bool,
    gen_phase: bool,
    ep: int,
    no_cuda_graph: bool,
    no_cupti: bool,
    functional_api: bool,
    routing_bias_scale: float,
) -> list[str]:
    cmd = [REMOTE_PYTHON, "benchmarks/bench_moe_deepseek.py"]
    if num_tokens:
        cmd.extend(["--num-tokens", num_tokens])
    cmd.extend(["--warmup", str(warmup), "--iters", str(iters), "--ep", str(ep)])
    if no_autotune:
        cmd.append("--no-autotune")
    if quiet:
        cmd.append("--quiet")
    if gen_phase:
        cmd.append("--gen-phase")
    if no_cuda_graph:
        cmd.append("--no-cuda-graph")
    if no_cupti:
        cmd.append("--no-cupti")
    if functional_api:
        cmd.append("--functional-api")
    cmd.extend(["--routing-bias-scale", str(routing_bias_scale)])
    return cmd


@app.function(
    image=image,
    gpu=gpu_type,
    timeout=6 * 60 * 60,
    memory=64 * 1024,
    volumes={"/cache/flashinfer": cache_volume},
)
def smoke() -> None:
    _prepare_flashinfer_source()
    _run(["nvidia-smi"])
    _run([REMOTE_PYTHON, "-m", "flashinfer", "show-config"])
    cache_volume.commit()


@app.function(
    image=image,
    gpu=gpu_type,
    timeout=12 * 60 * 60,
    memory=64 * 1024,
    volumes={"/cache/flashinfer": cache_volume},
)
def deepseek_moe_benchmark(
    num_tokens: str = "",
    warmup: int = 10,
    iters: int = 100,
    no_autotune: bool = False,
    quiet: bool = False,
    gen_phase: bool = False,
    ep: int = 1,
    no_cuda_graph: bool = False,
    no_cupti: bool = False,
    functional_api: bool = False,
    routing_bias_scale: float = 0.01,
    label: str = "deepseek_v3_nvfp4",
) -> str:
    _prepare_flashinfer_source()
    _run(["nvidia-smi"])

    run_id = time.strftime("%Y%m%d-%H%M%S")
    result_dir = Path("/cache/flashinfer/results/moe/deepseek_v3_nvfp4") / run_id
    result_dir.mkdir(parents=True, exist_ok=True)

    cmd = _benchmark_cmd(
        num_tokens=num_tokens,
        warmup=warmup,
        iters=iters,
        no_autotune=no_autotune,
        quiet=quiet,
        gen_phase=gen_phase,
        ep=ep,
        no_cuda_graph=no_cuda_graph,
        no_cupti=no_cupti,
        functional_api=functional_api,
        routing_bias_scale=routing_bias_scale,
    )
    log_path = result_dir / f"{label}.log"
    try:
        _run_to_file(cmd, log_path)
    finally:
        cache_volume.commit()

    print(f"Saved benchmark log to {log_path}", flush=True)
    return str(result_dir)


@app.local_entrypoint()
def main(
    task: str = "bench",
    gpu: str = gpu_type,
    num_tokens: str = "",
    warmup: int = 10,
    iters: int = 100,
    no_autotune: bool = False,
    quiet: bool = False,
    gen_phase: bool = False,
    ep: int = 1,
    no_cuda_graph: bool = False,
    no_cupti: bool = False,
    functional_api: bool = False,
    routing_bias_scale: float = 0.01,
    label: str = "deepseek_v3_nvfp4",
) -> None:
    if gpu != gpu_type:
        print(
            "GPU selection is fixed when Modal imports this file; pass MODAL_GPU "
            "or --gpu on the modal run command before the app is loaded.",
            flush=True,
        )

    if task == "smoke":
        smoke.remote()
        return

    if task != "bench":
        raise ValueError("task must be either 'smoke' or 'bench'")

    result_dir = deepseek_moe_benchmark.remote(
        num_tokens=num_tokens,
        warmup=warmup,
        iters=iters,
        no_autotune=no_autotune,
        quiet=quiet,
        gen_phase=gen_phase,
        ep=ep,
        no_cuda_graph=no_cuda_graph,
        no_cupti=no_cupti,
        functional_api=functional_api,
        routing_bias_scale=routing_bias_scale,
        label=label,
    )
    print(f"Modal result directory: {result_dir}", flush=True)
