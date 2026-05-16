#!/usr/bin/env python3
import argparse
import csv
import importlib.metadata
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


PORT = int(os.environ.get("SGLANG_PORT", "30000"))
HOST = os.environ.get("SGLANG_HOST", "localhost")
BASE_URL = f"http://{HOST}:{PORT}"
DATASET = os.environ.get("DATASET_PATH", "ShareGPT_V3_unfiltered_cleaned_split")
FP8_MODEL = os.environ.get("FP8_MODEL_PATH", "sgl-project/DeepSeek-V4-Flash-FP8")
FP4_MODEL = os.environ.get("FP4_MODEL_PATH", "deepseek-ai/DeepSeek-V4-Flash")
HUMMING_REPO = os.environ.get("HUMMING_REPO", "<humming-source>")
EXPECTED_HUMMING_COMMIT = "f43305a55cd5e9eb49bad2a2554c31be6df244d7"
SGLANG_WORKTREE = os.environ.get("SGLANG_WORKTREE", "<sglang-worktree>")
SERVER_LOG_DIR = os.environ.get("SERVER_LOG_DIR", "<server-log-dir>")
BIND_HOST = os.environ.get("SGLANG_BIND_HOST", "localhost")

TTFT_LENS = [16384, 32768, 65536, 131072]
TPOT_LENS = [1024, 32768, 65536, 131072]
TPOT_BATCHES = [1, 4, 8]

METRIC_PATTERNS = {
    "successful_requests": r"Successful requests:\s+([0-9.]+)",
    "benchmark_duration_s": r"Benchmark duration \(s\):\s+([0-9.]+)",
    "total_input_tokens": r"Total input tokens:\s+([0-9.]+)",
    "total_generated_tokens": r"Total generated tokens:\s+([0-9.]+)",
    "request_throughput_req_s": r"Request throughput \(req/s\):\s+([0-9.]+)",
    "input_throughput_tok_s": r"Input token throughput \(tok/s\):\s+([0-9.]+)",
    "output_throughput_tok_s": r"Output token throughput \(tok/s\):\s+([0-9.]+)",
    "total_throughput_tok_s": r"Total Token throughput \(tok/s\):\s+([0-9.]+)",
    "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([0-9.]+)",
    "median_ttft_ms": r"Median TTFT \(ms\):\s+([0-9.]+)",
    "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([0-9.]+)",
    "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([0-9.]+)",
    "median_tpot_ms": r"Median TPOT \(ms\):\s+([0-9.]+)",
    "p99_tpot_ms": r"P99 TPOT \(ms\):\s+([0-9.]+)",
    "mean_itl_ms": r"Mean ITL \(ms\):\s+([0-9.]+)",
    "median_itl_ms": r"Median ITL \(ms\):\s+([0-9.]+)",
    "p99_itl_ms": r"P99 ITL \(ms\):\s+([0-9.]+)",
}


@dataclass
class Variant:
    name: str
    model_path: str
    served_model_name: str
    env: dict[str, str] = field(default_factory=dict)
    extra_args: list[str] = field(default_factory=list)


VARIANTS = {
    "marlin_mxfp4a16": Variant(
        name="marlin_mxfp4a16",
        model_path=FP4_MODEL,
        served_model_name="dsv4-flash-marlin-mxfp4a16",
        env={"SGLANG_DSV4_FP4_EXPERTS": "1"},
        extra_args=["--moe-runner-backend", "marlin"],
    ),
    "humming_mxfp4a16": Variant(
        name="humming_mxfp4a16",
        model_path=FP4_MODEL,
        served_model_name="dsv4-flash-humming-mxfp4a16",
        env={
            "SGLANG_DSV4_FP4_EXPERTS": "1",
            "HUMMING_COMPILER": "nvcc",
            "SGLANG_HUMMING_MOE_GEMM_TYPE": "indexed",
        },
        extra_args=["--moe-runner-backend", "humming"],
    ),
    "flashinfer_mxfp4a16": Variant(
        name="flashinfer_mxfp4a16",
        model_path=FP4_MODEL,
        served_model_name="dsv4-flash-flashinfer-mxfp4a16",
        env={"SGLANG_DSV4_FP4_EXPERTS": "1"},
        extra_args=["--moe-runner-backend", "flashinfer_mxfp4"],
    ),
}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def check_file(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(path)


def wait_http(path: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    url = BASE_URL + path
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
        time.sleep(5)
    raise TimeoutError(f"timed out waiting for {url}: {last_error}")


def read_url(path: str) -> str:
    with urllib.request.urlopen(BASE_URL + path, timeout=10) as response:
        return response.read().decode("utf-8", errors="replace")


def check_output(cmd: list[str], cwd: str | None = None) -> str:
    return subprocess.check_output(cmd, cwd=cwd, text=True).strip()


def collect_runtime_info() -> dict:
    info: dict[str, object] = {
        "python": sys.version,
        "humming_repo": HUMMING_REPO,
        "expected_humming_commit": EXPECTED_HUMMING_COMMIT,
    }
    for package in [
        "sglang",
        "sgl-kernel",
        "humming-kernels",
        "flashinfer-python",
        "torch",
        "triton",
    ]:
        try:
            info[f"{package}_version"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            info[f"{package}_version"] = None
    spec = importlib.util.find_spec("humming")
    info["humming_origin"] = spec.origin if spec else None
    if Path(HUMMING_REPO).exists():
        info["humming_commit"] = check_output(["git", "rev-parse", "HEAD"], cwd=HUMMING_REPO)
        info["humming_status_short"] = check_output(["git", "status", "--short"], cwd=HUMMING_REPO)
    info["sglang_commit"] = check_output(["git", "rev-parse", "HEAD"], cwd=SGLANG_WORKTREE)
    info["sglang_branch"] = check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=SGLANG_WORKTREE)
    try:
        import flashinfer

        info["flashinfer_origin"] = flashinfer.__file__
    except Exception as exc:
        info["flashinfer_origin_error"] = repr(exc)
    try:
        from flashinfer.fused_moe.core import (
            interleave_moe_scales_for_sm90_mixed_gemm,
            interleave_moe_weights_for_sm90_mixed_gemm,
        )

        info["flashinfer_sm90_mixed_input_helpers"] = {
            "weights": repr(interleave_moe_weights_for_sm90_mixed_gemm),
            "scales": repr(interleave_moe_scales_for_sm90_mixed_gemm),
        }
    except Exception as exc:
        info["flashinfer_sm90_mixed_input_helpers_error"] = repr(exc)
    return info


def scan_server_log(variant_dir: Path) -> dict:
    log_path = variant_dir / "server.log"
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    moe_config_hits = sorted(
        set(re.findall(r"Using MoE kernel config from (\S+\.json)", text))
    )
    down_fallbacks = re.findall(r"Using MoE kernel config with down_moe=False", text)
    default_fallbacks = re.findall(r"Using default MoE kernel config", text)
    return {
        "moe_config_hits": moe_config_hits,
        "moe_config_hit_count": len(moe_config_hits),
        "down_fallback_count": len(down_fallbacks),
        "default_fallback_count": len(default_fallbacks),
    }


def launch_server(variant: Variant, out_dir: Path) -> subprocess.Popen:
    Path(SERVER_LOG_DIR).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
            "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
        }
    )
    env.update(variant.env)
    if variant.name.startswith("humming"):
        env.setdefault("HUMMING_CACHE_DIR", str(out_dir / variant.name / "humming_cache"))
        env.setdefault("HUMMING_TMP_DIR", str(out_dir / variant.name / "humming_tmp"))
        Path(env["HUMMING_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        Path(env["HUMMING_TMP_DIR"]).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        variant.model_path,
        "--served-model-name",
        variant.served_model_name,
        "--host",
        BIND_HOST,
        "--port",
        str(PORT),
        "--trust-remote-code",
        "--enable-cache-report",
        "--disable-radix-cache",
        "--log-level",
        "info",
        "--enable-metrics",
        "--page-size",
        "64",
        "--cuda-graph-max-bs",
        "64",
        "--max-running-requests",
        "64",
        "--mem-fraction-static",
        "0.80",
        "--tp-size",
        "8",
        "--enable-nsa-prefill-context-parallel",
        "--nsa-prefill-cp-mode",
        "round-robin-split",
        "--tool-call-parser",
        "deepseekv4",
        "--reasoning-parser",
        "deepseek-v4",
        "--speculative-algo",
        "EAGLE",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
    ]
    cmd.extend(variant.extra_args)

    variant_dir = out_dir / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "launch_command.json").write_text(
        json.dumps(
            {
                "cmd": cmd,
                "env": {
                    key: env[key]
                    for key in sorted(env)
                    if key.startswith("SGLANG_")
                    or key.startswith("HUMMING")
                    or key in {"CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"}
                },
            },
            indent=2,
        )
        + "\n"
    )
    log_file = open(variant_dir / "server.log", "w", buffering=1)
    log(f"launch {variant.name}: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return process


def stop_server(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(60):
            if process.poll() is not None:
                break
            time.sleep(1)
    if process.poll() is None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=30)


def parse_metrics(text: str) -> dict[str, float]:
    metrics = {}
    for key, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            value = float(match.group(1))
            if value.is_integer():
                value = int(value)
            metrics[key] = value
    return metrics


def run_bench(
    variant: Variant,
    out_dir: Path,
    bench_type: str,
    input_len: int,
    output_len: int,
    batch_size: int,
    round_id: int,
    timeout_s: int,
) -> dict:
    run_name = f"{bench_type}_isl{input_len}_osl{output_len}_bs{batch_size}_r{round_id}"
    raw_dir = out_dir / variant.name / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    detail_json = raw_dir / f"{run_name}.json"

    cmd = [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--host",
        "localhost",
        "--port",
        str(PORT),
        "--dataset-name",
        "random",
        "--dataset-path",
        DATASET,
        "--num-prompts",
        str(batch_size if bench_type == "tpot" else 10),
        "--max-concurrency",
        str(batch_size),
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--random-range-ratio",
        "1.0",
        "--model",
        variant.model_path,
        "--served-model-name",
        variant.served_model_name,
        "--tokenizer",
        variant.model_path,
        "--output-file",
        str(detail_json),
        "--output-details",
        "--disable-tqdm",
    ]

    log(f"bench {variant.name} {run_name}")
    start = time.time()
    completed = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )
    elapsed = time.time() - start
    (raw_dir / f"{run_name}.stdout.txt").write_text(completed.stdout)
    (raw_dir / f"{run_name}.stderr.txt").write_text(completed.stderr)

    record = {
        "variant": variant.name,
        "bench_type": bench_type,
        "input_len": input_len,
        "output_len": output_len,
        "batch_size": batch_size,
        "round": round_id,
        "returncode": completed.returncode,
        "elapsed_s": round(elapsed, 3),
        "detail_json": str(detail_json),
        "cmd": cmd,
        "metrics": parse_metrics(completed.stdout + "\n" + completed.stderr),
    }
    status = "ok" if completed.returncode == 0 else f"rc={completed.returncode}"
    log(f"done {variant.name} {run_name} {status} elapsed={elapsed:.1f}s")
    return record


def best_records(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for rec in records:
        if rec["returncode"] != 0:
            continue
        key = (
            rec["variant"],
            rec["bench_type"],
            rec["input_len"],
            rec["output_len"],
            rec["batch_size"],
        )
        groups.setdefault(key, []).append(rec)

    best = []
    for key, group in sorted(groups.items()):
        metric = "mean_ttft_ms" if key[1] == "ttft" else "mean_tpot_ms"
        valid = [rec for rec in group if metric in rec["metrics"]]
        if not valid:
            continue
        best.append(min(valid, key=lambda rec: rec["metrics"][metric]))
    return best


def write_summary(out_dir: Path, records: list[dict]) -> None:
    best = best_records(records)
    all_keys = sorted({key for rec in records for key in rec["metrics"]})
    fieldnames = [
        "variant",
        "bench_type",
        "input_len",
        "output_len",
        "batch_size",
        "round",
        "returncode",
        "elapsed_s",
    ] + all_keys

    with open(out_dir / "all_runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = {key: rec.get(key) for key in fieldnames}
            row.update(rec["metrics"])
            writer.writerow(row)

    with open(out_dir / "best_runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in best:
            row = {key: rec.get(key) for key in fieldnames}
            row.update(rec["metrics"])
            writer.writerow(row)

    lines = ["# DeepSeek-V4 Flash CP8 TP8 No-DeepEP Compare", ""]
    lines.append("Best run per scenario, taking the minimum mean TTFT/TPOT across 3 rounds.")
    lines.append("")
    for bench_type in ["ttft", "tpot"]:
        lines.append(f"## {bench_type.upper()}")
        lines.append("")
        rows = [rec for rec in best if rec["bench_type"] == bench_type]
        metric = "mean_ttft_ms" if bench_type == "ttft" else "mean_tpot_ms"
        lines.append("| variant | input_len | output_len | batch_size | round | metric_ms | output_tput_tok_s |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for rec in sorted(rows, key=lambda r: (r["variant"], r["input_len"], r["batch_size"])):
            metrics = rec["metrics"]
            lines.append(
                "| {variant} | {input_len} | {output_len} | {batch_size} | {round} | {metric:.2f} | {out_tput} |".format(
                    variant=rec["variant"],
                    input_len=rec["input_len"],
                    output_len=rec["output_len"],
                    batch_size=rec["batch_size"],
                    round=rec["round"],
                    metric=float(metrics.get(metric, float("nan"))),
                    out_tput=(
                        f"{float(metrics['output_throughput_tok_s']):.2f}"
                        if "output_throughput_tok_s" in metrics
                        else ""
                    ),
                )
            )
        lines.append("")

    failures = [rec for rec in records if rec["returncode"] != 0]
    if failures:
        lines.append("## Failures")
        lines.append("")
        lines.append("| variant | bench_type | input_len | output_len | batch_size | round | returncode |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for rec in failures:
            lines.append(
                f"| {rec['variant']} | {rec['bench_type']} | {rec['input_len']} | "
                f"{rec['output_len']} | {rec['batch_size']} | {rec['round']} | {rec['returncode']} |"
            )
        lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    (out_dir / "all_runs.json").write_text(json.dumps(records, indent=2) + "\n")
    (out_dir / "best_runs.json").write_text(json.dumps(best, indent=2) + "\n")


def run_variant(variant: Variant, out_dir: Path, rounds: int, timeout_s: int) -> list[dict]:
    check_file(variant.model_path + "/config.json")
    process = None
    records = []
    variant_dir = out_dir / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    try:
        process = launch_server(variant, out_dir)
        wait_http("/health", timeout_s=1800)
        (variant_dir / "models.json").write_text(read_url("/v1/models") + "\n")
        log(f"server ready {variant.name}")

        for input_len in TTFT_LENS:
            for round_id in range(1, rounds + 1):
                records.append(
                    run_bench(
                        variant,
                        out_dir,
                        "ttft",
                        input_len=input_len,
                        output_len=256,
                        batch_size=1,
                        round_id=round_id,
                        timeout_s=timeout_s,
                    )
                )

        for input_len in TPOT_LENS:
            for batch_size in TPOT_BATCHES:
                for round_id in range(1, rounds + 1):
                    records.append(
                        run_bench(
                            variant,
                            out_dir,
                            "tpot",
                            input_len=input_len,
                            output_len=1024,
                            batch_size=batch_size,
                            round_id=round_id,
                            timeout_s=timeout_s,
                        )
                    )
    finally:
        stop_server(process)
        (variant_dir / "server_log_check.json").write_text(
            json.dumps(scan_server_log(variant_dir), indent=2) + "\n"
        )
        log(f"server stopped {variant.name}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("BENCH_OUT_DIR", f"./runs/{time.strftime('%Y%m%d_%H%M%S')}"),
    )
    parser.add_argument(
        "--variants",
        default="marlin_mxfp4a16,humming_mxfp4a16,flashinfer_mxfp4a16",
        help="Comma-separated variant names.",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout-s", type=int, default=3600)
    args = parser.parse_args()

    check_file(DATASET)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = [name.strip() for name in args.variants.split(",") if name.strip()]
    meta = {
        "started_at": now(),
        "dataset": DATASET,
        "port": PORT,
        "ttft_lens": TTFT_LENS,
        "tpot_lens": TPOT_LENS,
        "tpot_batches": TPOT_BATCHES,
        "rounds": args.rounds,
        "variants": selected,
        "note": "Flash CP8 TP8 reference style, no DeepEP launch args or env. Server launched with --disable-radix-cache. MXFP4/W4A16 backend comparison.",
    }
    meta["runtime"] = collect_runtime_info()
    if "humming_mxfp4a16" in selected and meta["runtime"].get("humming_commit") != EXPECTED_HUMMING_COMMIT:
        raise RuntimeError(f"unexpected humming commit: {meta['runtime'].get('humming_commit')}")
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    records = []
    try:
        for name in selected:
            if name not in VARIANTS:
                raise KeyError(f"unknown variant: {name}")
            records.extend(run_variant(VARIANTS[name], out_dir, args.rounds, args.timeout_s))
            write_summary(out_dir, records)
    finally:
        write_summary(out_dir, records)
        log(f"results: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
