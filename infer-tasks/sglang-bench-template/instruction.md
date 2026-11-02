<!-- TODO(instruction): restate the source in 1–3 sentences and paste the literal launch command. -->
Benchmark sglang serving for the configuration in this task.

1. Start the sglang server with exactly this command (also in `$SGLANG_LAUNCH_CMD`):

   ```bash
   TODO_LAUNCH_CMD
   ```

   Wait until `http://127.0.0.1:$SERVER_PORT/health` returns 200. Keep the full server log at
   `/app/results/server.log`.

2. Run one benchmark pass against it with a fixed load and no other traffic:

   ```bash
   python3 -m $BENCH_MODULE --backend sglang --host 127.0.0.1 --port $SERVER_PORT \
     --model $MODEL_PATH --dataset-name $BENCH_DATASET \
     --random-input-len $BENCH_INPUT_LEN --random-output-len $BENCH_OUTPUT_LEN \
     --random-range-ratio $BENCH_RANGE_RATIO --num-prompts $BENCH_NUM_PROMPTS \
     --max-concurrency $BENCH_MAX_CONCURRENCY --request-rate $BENCH_REQUEST_RATE \
     --seed $BENCH_SEED --output-file /app/results/bench.jsonl $BENCH_EXTRA_ARGS
   ```

3. Do not change server flags, load parameters, or edit `/app/results/bench.jsonl` by hand.
   Stop the server when done.
