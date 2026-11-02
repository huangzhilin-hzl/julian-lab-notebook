"""Validate sglang bench_serving JSONL results (one line per concurrency) and extract metrics.

Result keys come from sglang python/sglang/benchmark/serving.py (commit 4ae54e9, result dict at
L1855-1908; one JSON object appended per run with --output-file, L1961).
Derived per-point metrics follow InferenceX infx/results/fixed_sequence.py L192-206
(SemiAnalysisAI/InferenceX@de8884c): tput_per_gpu, output/input_tput_per_gpu, {mean,median,p90,p99}_intvty = 1000/tpot_ms.
Reward is pass/fail only; no performance thresholds.
"""

import json
import math
import os
import sys

REPORT_KEYS = [
    "output_throughput", "input_throughput", "total_throughput", "request_throughput",
    "mean_ttft_ms", "median_ttft_ms", "p90_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "median_tpot_ms", "p90_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "median_itl_ms", "p90_itl_ms", "p99_itl_ms",
    "mean_e2e_latency_ms", "median_e2e_latency_ms", "p90_e2e_latency_ms", "p99_e2e_latency_ms",
]
TPOT_KEYS = ["mean_tpot_ms", "median_tpot_ms", "p90_tpot_ms", "p99_tpot_ms"]
CONTEXT_KEYS = [
    "backend", "dataset_name", "random_input_len", "random_output_len", "random_range_ratio",
    "max_concurrency", "request_rate", "completed", "duration",
    "total_input_tokens", "total_output_tokens", "accept_length",
]


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def env_int_list(name: str) -> list[int]:
    raw = os.environ.get(name, "").split()
    if not raw:
        fail(f"{name} is empty")
    try:
        return [int(x) for x in raw]
    except ValueError:
        fail(f"{name}={os.environ[name]!r} is not a list of ints")


def env_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else None


def finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check_point(i: int, r: dict, conc: int, per_conc: int, isl, osl, ratio, info) -> None:
    where = f"line {i + 1} (concurrency {conc})"
    if r.get("max_concurrency") != conc:
        fail(f"{where}: max_concurrency={r.get('max_concurrency')!r}, expected {conc}")
    if r.get("completed") != conc * per_conc:
        fail(f"{where}: completed={r.get('completed')!r}, expected {conc * per_conc} "
             f"(= {conc} x {per_conc})")
    if isl is not None and r.get("random_input_len") != isl:
        fail(f"{where}: random_input_len={r.get('random_input_len')!r}, expected {isl}")
    if osl is not None and r.get("random_output_len") != osl:
        fail(f"{where}: random_output_len={r.get('random_output_len')!r}, expected {osl}")
    if ratio is not None and r.get("random_range_ratio") != ratio:
        fail(f"{where}: random_range_ratio={r.get('random_range_ratio')!r}, expected {ratio}")
    for key in REPORT_KEYS:
        if not finite(r.get(key)):
            fail(f"{where}: {key}={r.get(key)!r} is not a finite number")
    if r["output_throughput"] <= 0:
        fail(f"{where}: output_throughput <= 0")
    if info:
        got = r.get("server_info") or {}
        for k, want in info.items():
            if got.get(k) != want:
                fail(f"{where}: server_info[{k!r}]={got.get(k)!r}, expected {want!r}")


def point_metrics(r: dict, num_gpus: int) -> dict:
    p = {k: r.get(k) for k in CONTEXT_KEYS + REPORT_KEYS}
    p["num_gpus"] = num_gpus
    p["tput_per_gpu"] = r["total_throughput"] / num_gpus
    p["output_tput_per_gpu"] = r["output_throughput"] / num_gpus
    p["input_tput_per_gpu"] = (r["total_throughput"] - r["output_throughput"]) / num_gpus
    for k in TPOT_KEYS:  # tok/s/user; InferenceX skips tpot <= 0
        p[k.replace("_ms", "").replace("tpot", "intvty")] = 1000.0 / r[k] if r[k] > 0 else None
    return p


def main(path: str, metrics_out: str) -> None:
    concs = env_int_list("EXPECTED_CONC_LIST")
    per_conc = env_int("EXPECTED_PROMPTS_PER_CONC")
    num_gpus = env_int("NUM_GPUS")
    if per_conc is None or per_conc <= 0:
        fail("EXPECTED_PROMPTS_PER_CONC must be a positive int")
    if num_gpus is None or num_gpus <= 0:
        fail("NUM_GPUS must be a positive int")
    isl = env_int("EXPECTED_INPUT_LEN")
    osl = env_int("EXPECTED_OUTPUT_LEN")
    raw_ratio = os.environ.get("EXPECTED_RANGE_RATIO", "").strip()
    ratio = float(raw_ratio) if raw_ratio else None
    raw_info = os.environ.get("EXPECTED_SERVER_INFO_JSON", "").strip()
    info = json.loads(raw_info) if raw_info else None

    if not os.path.isfile(path):
        fail(f"{path} not found; the benchmark did not run")
    lines = [ln for ln in open(path).read().splitlines() if ln.strip()]
    if len(lines) != len(concs):
        fail(f"expected {len(concs)} result lines (one per concurrency {concs}) in {path}, "
             f"found {len(lines)}")
    results = []
    for i, ln in enumerate(lines):
        try:
            results.append(json.loads(ln))
        except json.JSONDecodeError as e:
            fail(f"line {i + 1} is not JSON: {e}")
    got = [r.get("max_concurrency") for r in results]
    if got != concs:
        fail(f"max_concurrency per line = {got}, expected {concs} in this order")
    for i, (r, c) in enumerate(zip(results, concs)):
        check_point(i, r, c, per_conc, isl, osl, ratio, info)

    metrics = {"points": [point_metrics(r, num_gpus) for r in results]}
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print("PASS")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
