# DeepSeek-V4.1 Flash Humming A/B

Compare serving throughput and latency with humming off and humming on under a fixed load.

| Humming | Task path (from repo root) | MoE runner |
| --- | --- | --- |
| Off | `infer-tasks/dsv41-flash-h200-fp4-lowlatency` | `flashinfer_mxfp4` |
| On | `infer-tasks/dsv41-flash-h200-fp4-lowlatency-humming` | `humming` |

The single intended runtime difference is the `--moe-runner-backend` value in
`SGLANG_LAUNCH_CMD`. Task identity, metadata and README/SOURCES may also differ.
Both tasks use DSPARK (`--speculative-algorithm DSPARK --speculative-dspark-block-size 5`),
TP8/EP8 and CUDA graphs. No `--quantization humming` or `SGLANG_HUMMING_*` overrides.

Both use `lmsysorg/sglang:dev@sha256:77ff63ee69fd7ebef20df3a6d036ee4b8a4ffe0734efb0028c3b7643386e3ccf`
(index digest), sglang build `38d9e4a3fe4a493b0ae468477a274e83defda81a`
(`nightly-dev-20261102-38d9e4a3`). See each task's SOURCES.md for AMD64 digest,
PR #46018 provenance and cookbook citations.

Hardware: one Modal node with H200 x8 for each task. Load: `random-ids`, input/output
8192/1024, range ratio 0.8, concurrency 1, 2, 4, 8, 16 in ascending order;
`num_prompts = 10 * concurrency`, `warmup_requests = 2 * concurrency`, seed 42,
request rate inf, ignore EOS, flush radix cache after warmup. Load and verifier
settings are identical across tasks.

## Run

From the repo root:

```bash
cd infer-experiments/dsv41-humming-ab
harbor run -c job.yaml -y
```

The job's task paths and `jobs_dir: runs` are relative to this experiment working
directory. Install `harbor[modal]` 0.25.0, configure Modal credentials, and create
`hf-cache` once with `modal volume create hf-cache` before running. The job uses
the oracle, a single concurrent trial, timeout multiplier 3.0 and sandbox lifetime
14400 seconds.

Run humming off then humming on with no competing traffic. For repeats, use a fresh
`--job-name` and alternate task order in a local job copy. Record trial IDs, order,
timestamps, GPU placement, task revision, image digest and launch commands; serial
Modal trials can use different hosts.

## Measurements

Preserve raw `/app/results/{run_meta.txt,server.log,bench.log,bench.jsonl}`, serve
`server_info.json`, and bench `metrics.json`, rewards and errors under `runs/`.
Compare successful points at matching concurrency, retaining each repeat:

- Output tokens/s and total/output/input throughput per GPU (divide by 8).
- TTFT, TPOT and ITL p50/p99 in ms.
- DSPARK `accept_length` when emitted, alongside latency.

Report raw values and `100 * (on - off) / off`. Flag absolute changes greater than
5% for review; higher throughput and lower latency are favorable. Zero baseline or
missing metrics have no relative comparison. Flags do not affect the completeness/load
reward and do not establish statistical significance.

Image differs from the cookbook's `dev-dsv41` recommendation.

Exact launch commands (off, then on):

```bash
sglang serve --trust-remote-code --model-path deepseek-ai/DeepSeek-V4.1-Flash --tp 8 --ep-size 8 --mem-fraction-static 0.8 --attention-backend dsv4 --moe-runner-backend flashinfer_mxfp4 --cuda-graph-max-bs-decode 64 --speculative-algorithm DSPARK --speculative-dspark-block-size 5 --reasoning-parser auto --tool-call-parser auto --host 0.0.0.0 --port 30000
sglang serve --trust-remote-code --model-path deepseek-ai/DeepSeek-V4.1-Flash --tp 8 --ep-size 8 --mem-fraction-static 0.8 --attention-backend dsv4 --moe-runner-backend humming --cuda-graph-max-bs-decode 64 --speculative-algorithm DSPARK --speculative-dspark-block-size 5 --reasoning-parser auto --tool-call-parser auto --host 0.0.0.0 --port 30000
```
