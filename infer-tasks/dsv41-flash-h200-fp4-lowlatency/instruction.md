Benchmark DeepSeek-V4.1-Flash serving with the sglang cookbook's verified **H200 · flash · FP4 ·
Low-Latency · single node** recipe (8× H200, image `lmsysorg/sglang:dev-dsv41`).

1. Start the server with exactly this command (also in `$SGLANG_LAUNCH_CMD`):

   ```bash
   sglang serve \
     --trust-remote-code \
     --model-path deepseek-ai/DeepSeek-V4.1-Flash \
     --tp 8 \
     --ep-size 8 \
     --mem-fraction-static 0.8 \
     --attention-backend dsv4 \
     --moe-runner-backend flashinfer_mxfp4 \
     --cuda-graph-max-bs-decode 64 \
     --reasoning-parser auto \
     --tool-call-parser auto \
     --host 0.0.0.0 \
     --port 30000
   ```

   Wait until `http://127.0.0.1:30000/health` returns 200. Keep the full server log at
   `/app/results/server.log`.

2. Run one benchmark pass with a fixed load and no other traffic. The load is set by the
   `BENCH_*` variables (TODO: the cookbook gives no load for this recipe):

   ```bash
   python3 -m $BENCH_MODULE --backend sglang --host 127.0.0.1 --port $SERVER_PORT \
     --model $MODEL_PATH --dataset-name $BENCH_DATASET \
     --random-input-len $BENCH_INPUT_LEN --random-output-len $BENCH_OUTPUT_LEN \
     --random-range-ratio $BENCH_RANGE_RATIO --num-prompts $BENCH_NUM_PROMPTS \
     --max-concurrency $BENCH_MAX_CONCURRENCY --request-rate $BENCH_REQUEST_RATE \
     --seed $BENCH_SEED --output-file /app/results/bench.jsonl $BENCH_EXTRA_ARGS
   ```

3. Do not change server flags or load parameters, and do not edit `/app/results/bench.jsonl`.
   Stop the server when done.
