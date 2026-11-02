"""Validate one sglang bench_serving JSONL result and extract the headline metrics.

Keys come from sglang python/sglang/benchmark/serving.py (result dict written with
--output-file; one JSON object appended per run).
"""

import json
import math
import os
import sys

REPORT_KEYS = [
    "output_throughput", "input_throughput", "total_throughput", "request_throughput",
    "median_ttft_ms", "p99_ttft_ms",
    "median_tpot_ms", "p99_tpot_ms",
    "median_itl_ms", "p99_itl_ms",
    "median_e2e_latency_ms", "p99_e2e_latency_ms",
]
CONTEXT_KEYS = [
    "backend", "dataset_name", "random_input_len", "random_output_len", "random_range_ratio",
    "max_concurrency", "request_rate", "completed", "duration", "accept_length",
]


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def expect_int(result: dict, key: str, env: str) -> None:
    raw = os.environ.get(env, "")
    if not raw:
        return
    if result.get(key) != int(raw):
        fail(f"{key}={result.get(key)!r}, expected {raw} ({env})")


def main(path: str, metrics_out: str) -> None:
    if not os.path.isfile(path):
        fail(f"{path} not found; the benchmark did not run")
    lines = [ln for ln in open(path).read().splitlines() if ln.strip()]
    if len(lines) != 1:
        fail(f"expected exactly 1 result line in {path}, found {len(lines)}")
    try:
        result = json.loads(lines[0])
    except json.JSONDecodeError as e:
        fail(f"result line is not JSON: {e}")

    for key in REPORT_KEYS:
        v = result.get(key)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            fail(f"{key}={v!r} is not a finite number")
    if result["output_throughput"] <= 0:
        fail("output_throughput <= 0")

    expect_int(result, "completed", "EXPECTED_NUM_PROMPTS")
    expect_int(result, "random_input_len", "EXPECTED_INPUT_LEN")
    expect_int(result, "random_output_len", "EXPECTED_OUTPUT_LEN")
    expect_int(result, "max_concurrency", "EXPECTED_MAX_CONCURRENCY")

    expected_info = os.environ.get("EXPECTED_SERVER_INFO_JSON", "").strip()
    if expected_info:
        info = result.get("server_info") or {}
        for k, want in json.loads(expected_info).items():
            if info.get(k) != want:
                fail(f"server_info[{k!r}]={info.get(k)!r}, expected {want!r}")

    metrics = {k: result.get(k) for k in CONTEXT_KEYS + REPORT_KEYS}
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print("PASS")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
