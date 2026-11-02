The DeepSeek-V4.1-Flash sglang server from the previous step is running on `127.0.0.1:30000`.
Do not restart it or change its flags.

Benchmark it with the InferenceX fixed-seq-len load (8k/1k, `random`, range ratio 0.8, request rate
inf) and no other traffic: one run per concurrency in `$BENCH_CONC_LIST` (`1 2 4 8 16`),
in the listed order, all appending to the same result file. Start from an empty
`/app/results/bench.jsonl`.

```bash
for c in $BENCH_CONC_LIST; do
  python3 -m $BENCH_MODULE --backend sglang --host 127.0.0.1 --port $SERVER_PORT \
    --model $MODEL_PATH --dataset-name $BENCH_DATASET \
    --random-input-len $BENCH_INPUT_LEN --random-output-len $BENCH_OUTPUT_LEN \
    --random-range-ratio $BENCH_RANGE_RATIO \
    --num-prompts $((c * BENCH_PROMPTS_PER_CONC)) --max-concurrency $c \
    --warmup-requests $((c * BENCH_WARMUPS_PER_CONC)) \
    --request-rate $BENCH_REQUEST_RATE --seed $BENCH_SEED \
    --output-file /app/results/bench.jsonl $BENCH_EXTRA_ARGS
done
```

Do not change load parameters or edit `/app/results/bench.jsonl` by hand.
