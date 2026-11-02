The sglang server from the previous step is running on `127.0.0.1:$SERVER_PORT`. Do not restart
it or change its flags.

Run one benchmark pass against it with a fixed load and no other traffic:

```bash
python3 -m $BENCH_MODULE --backend sglang --host 127.0.0.1 --port $SERVER_PORT \
  --model $MODEL_PATH --dataset-name $BENCH_DATASET \
  --random-input-len $BENCH_INPUT_LEN --random-output-len $BENCH_OUTPUT_LEN \
  --random-range-ratio $BENCH_RANGE_RATIO --num-prompts $BENCH_NUM_PROMPTS \
  --max-concurrency $BENCH_MAX_CONCURRENCY --request-rate $BENCH_REQUEST_RATE \
  --seed $BENCH_SEED --output-file /app/results/bench.jsonl $BENCH_EXTRA_ARGS
```

Do not change load parameters or edit `/app/results/bench.jsonl` by hand.
