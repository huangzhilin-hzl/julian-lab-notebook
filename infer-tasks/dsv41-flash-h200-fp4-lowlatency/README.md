# molou/dsv41-flash-h200-fp4-lowlatency

sglang cookbook DeepSeek-V4.1 Flash, H200 · FP4 · Low-Latency · single node, as a multi-step Harbor
task. Built from `../sglang-bench-template` (see its README for the step layout and caveats).
Sources and line refs: `SOURCES.md`.

- `steps/serve`: the cookbook launch command (`SGLANG_LAUNCH_CMD`), shared verifier on `/health` and
  `/server_info`, `min_reward = 1.0`.
- `steps/bench`: one `sglang.bench_serving` run per concurrency `1 2 4 8 16` against the same warm
  server, graded in a separate CPU-only container (5 result lines, `metrics.json` with one point each).

## Load

The cookbook gives no load for this cell. The task applies InferenceX's fixed-seq-len rules
(`SemiAnalysisAI/InferenceX@de8884c`) to the **cookbook** server config: ISL/OSL 8192/1024,
synthetic token ids (sglang `random-ids`), range ratio 0.8, `num_prompts = 10 × conc`,
`2 × conc` warmups, request rate `inf`, ignore EOS.
Concurrency list, 8k/1k only, and the sglang client are zhilin's choices (2026-11-02).

Not comparable to any InferenceX number:

- InferenceX has **no fixed-load result for dsv41flash**; it benchmarks this model only as agentic
  coding (trace replay), with different server flags: EP 1, `marlin` MoE runner, MTP/DSpark
  speculative decoding, `mem-fraction-static 0.70`. The cookbook cell uses EP 8, `flashinfer_mxfp4`,
  0.8, no speculative decoding.
- Warmup differs: InferenceX warms up with the full first prompt and full output length; sglang
  `bench_serving` reuses the first prompt but caps warmup output at 32 tokens.
- Client differs: sglang `bench_serving --backend sglang` (`/generate`) instead of InferenceX's own
  client over OpenAI completions. Both build prompts from synthetic token ids (sglang `random-ids`,
  InferenceX `random`), but InferenceX re-encodes each prompt until it is exactly the target length;
  sglang decodes the ids to text once, so the server-side token count can drift from the target.
- All points share one warm server (run ascending, `--flush-cache` after each warmup).

## Run

```bash
harbor lint .
harbor run -p . -a oracle -e <env>   # needs 8x H200; see ../HARBOR_NOTES.md for k8s
```

Open points: the server surviving from `serve` into `bench` is unverified until the first real run;
the bench timeout (3600 s) is a reasoned budget, not measured; the cookbook's `docker run` flags
`--shm-size 32g --ipc=host` have no `task.toml` field (SOURCES.md).
