#!/bin/bash
# Oracle: start the server from $SGLANG_LAUNCH_CMD, wait for /health, run one bench_serving pass.
# All knobs come from [environment.env] in task.toml; nothing is hardcoded here.
set -euo pipefail

OUT=/app/results
mkdir -p "$OUT"
: "${SGLANG_LAUNCH_CMD:?}" "${SERVER_PORT:?}" "${MODEL_PATH:?}"

{ echo "date: $(date -Is)"; echo "launch: $SGLANG_LAUNCH_CMD"; nvidia-smi -L || true; } > "$OUT/run_meta.txt"

bash -c "exec $SGLANG_LAUNCH_CMD" > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true' EXIT

python3 - <<PY
import sys, time, urllib.request, os
deadline = time.time() + int(os.environ.get("SERVER_READY_TIMEOUT_SEC", "2400"))
url = "http://127.0.0.1:${SERVER_PORT}/health"
while time.time() < deadline:
    try:
        os.kill(${SERVER_PID}, 0)
    except OSError:
        sys.exit("server process exited before becoming healthy; see /app/results/server.log")
    try:
        if urllib.request.urlopen(url, timeout=5).status == 200:
            print("server healthy"); sys.exit(0)
    except Exception:
        pass
    time.sleep(10)
sys.exit("server not healthy before SERVER_READY_TIMEOUT_SEC")
PY

# shellcheck disable=SC2086  # BENCH_EXTRA_ARGS is intentionally word-split
python3 -m "$BENCH_MODULE" --backend sglang --host 127.0.0.1 --port "$SERVER_PORT" \
  --model "$MODEL_PATH" --dataset-name "$BENCH_DATASET" \
  --random-input-len "$BENCH_INPUT_LEN" --random-output-len "$BENCH_OUTPUT_LEN" \
  --random-range-ratio "$BENCH_RANGE_RATIO" --num-prompts "$BENCH_NUM_PROMPTS" \
  --max-concurrency "$BENCH_MAX_CONCURRENCY" --request-rate "$BENCH_REQUEST_RATE" \
  --seed "$BENCH_SEED" --output-file "$OUT/bench.jsonl" $BENCH_EXTRA_ARGS \
  2>&1 | tee "$OUT/bench.log"
