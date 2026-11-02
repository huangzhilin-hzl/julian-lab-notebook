#!/bin/bash
# Oracle, step "bench": run one bench_serving pass against the server started in step "serve".
# The step healthcheck in task.toml already confirmed /health. The server is left running;
# Harbor stops the environment after the last step.
# All knobs come from [environment.env] in task.toml; nothing is hardcoded here.
set -euo pipefail

OUT=/app/results
mkdir -p "$OUT"
: "${SERVER_PORT:?}" "${MODEL_PATH:?}"

# shellcheck disable=SC2086  # BENCH_EXTRA_ARGS is intentionally word-split
python3 -m "$BENCH_MODULE" --backend sglang --host 127.0.0.1 --port "$SERVER_PORT" \
  --model "$MODEL_PATH" --dataset-name "$BENCH_DATASET" \
  --random-input-len "$BENCH_INPUT_LEN" --random-output-len "$BENCH_OUTPUT_LEN" \
  --random-range-ratio "$BENCH_RANGE_RATIO" --num-prompts "$BENCH_NUM_PROMPTS" \
  --max-concurrency "$BENCH_MAX_CONCURRENCY" --request-rate "$BENCH_REQUEST_RATE" \
  --seed "$BENCH_SEED" --output-file "$OUT/bench.jsonl" $BENCH_EXTRA_ARGS \
  2>&1 | tee "$OUT/bench.log"
