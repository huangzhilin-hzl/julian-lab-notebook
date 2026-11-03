#!/bin/bash
# Oracle, step "bench": one bench_serving run per concurrency in BENCH_CONC_LIST (ascending), all
# against the server started in step "serve". Each run appends one JSON line to bench.jsonl.
# The step healthcheck in task.toml already confirmed /health. The server is left running;
# Harbor stops the environment after the last step.
# All knobs come from [environment.env] in task.toml; nothing is hardcoded here.
set -euo pipefail

OUT=/app/results
mkdir -p "$OUT"
: "${SERVER_PORT:?}" "${MODEL_PATH:?}" "${BENCH_CONC_LIST:?}" \
  "${BENCH_PROMPTS_PER_CONC:?}" "${BENCH_WARMUPS_PER_CONC:?}"

# bench_serving appends to --output-file; start clean so a retried step cannot leave extra lines.
rm -f "$OUT/bench.jsonl" "$OUT/bench.log"

for c in $BENCH_CONC_LIST; do
  echo "=== concurrency $c: num_prompts $((c * BENCH_PROMPTS_PER_CONC)), warmups $((c * BENCH_WARMUPS_PER_CONC))" \
    | tee -a "$OUT/bench.log"
  # shellcheck disable=SC2086  # BENCH_EXTRA_ARGS is intentionally word-split
  python3 -m "$BENCH_MODULE" --backend sglang --host 127.0.0.1 --port "$SERVER_PORT" \
    --model "$MODEL_PATH" --dataset-name "$BENCH_DATASET" \
    --random-input-len "$BENCH_INPUT_LEN" --random-output-len "$BENCH_OUTPUT_LEN" \
    --random-range-ratio "$BENCH_RANGE_RATIO" \
    --num-prompts "$((c * BENCH_PROMPTS_PER_CONC))" --max-concurrency "$c" \
    --warmup-requests "$((c * BENCH_WARMUPS_PER_CONC))" \
    --request-rate "$BENCH_REQUEST_RATE" --seed "$BENCH_SEED" \
    --output-file "$OUT/bench.jsonl" $BENCH_EXTRA_ARGS \
    2>&1 | tee -a "$OUT/bench.log"
done
