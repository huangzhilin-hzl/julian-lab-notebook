# molou/dsv41-flash-h200-fp4-lowlatency

DeepSeek-V4.1 Flash FP4 serving benchmark on one node with 8× H200, TP8/EP8,
CUDA graphs and DSPARK (block size 5).

Image: `lmsysorg/sglang:dev@sha256:77ff63ee69fd7ebef20df3a6d036ee4b8a4ffe0734efb0028c3b7643386e3ccf`.
sglang build: `38d9e4a3fe4a493b0ae468477a274e83defda81a`.
Image differs from the cookbook's `dev-dsv41` recommendation.

Exact launch command:

```bash
sglang serve --trust-remote-code --model-path deepseek-ai/DeepSeek-V4.1-Flash --tp 8 --ep-size 8 --mem-fraction-static 0.8 --attention-backend dsv4 --moe-runner-backend flashinfer_mxfp4 --cuda-graph-max-bs-decode 64 --speculative-algorithm DSPARK --speculative-dspark-block-size 5 --reasoning-parser auto --tool-call-parser auto --host 0.0.0.0 --port 30000
```

Load: `sglang.bench_serving`, `random-ids`, ISL/OSL 8192/1024, range ratio 0.8,
concurrency 1, 2, 4, 8, 16 in ascending order against one warm server; 10× concurrency
prompts and 2× concurrency warmups per point, request rate `inf`, seed 42, ignore EOS,
`--flush-cache` after warmup. No quantization or Humming environment overrides.

From this task directory, run:

```bash
harbor lint .
harbor run -p . -a oracle -e modal
```

Configure Modal credentials and install `harbor[modal]` 0.25.0. For the paired job
with a persistent weights cache, use [the experiment](../../infer-experiments/dsv41-humming-ab/README.md).

The shared serve verifier checks `/health` and `/server_info`; the separate CPU-only
bench verifier checks five JSONL points for completeness and matching load.
Reward is 0/1 with no performance threshold. Outputs in `/app/results` include
`server.log`, `server.pid`, `run_meta.txt`, `bench.log` and `bench.jsonl`.
Verifier `metrics.json` records throughput (total and per GPU), TTFT, TPOT, ITL,
E2E latency (mean/p50/p90/p99), interactivity (1000/TPOT), and available acceptance length.

See [SOURCES.md](SOURCES.md) for pinned citations and client differences from
InferenceX, and [Harbor notes](../HARBOR_NOTES.md) for provider constraints.
