# DeepSeek-V4.1 Flash Humming A/B

Hypothesis: selecting Humming for the MoE runner changes serving throughput and
latency under the fixed InferenceX load. No performance result is available yet.

| Arm | Task path (from repo root) | MoE runner |
| --- | --- | --- |
| Off | `infer-tasks/dsv41-flash-h200-fp4-lowlatency` | `flashinfer_mxfp4` |
| On | `infer-tasks/dsv41-flash-h200-fp4-lowlatency-humming` | `humming` |

The single intended runtime difference is the `--moe-runner-backend` value in
`SGLANG_LAUNCH_CMD`. Task identity, metadata and README/SOURCES may also differ.
Both arms use DSPARK (`--speculative-algorithm DSPARK --speculative-dspark-block-size 5`),
TP8/EP8 and CUDA graphs. No `--quantization humming` or `SGLANG_HUMMING_*` overrides.

Both use `lmsysorg/sglang:dev@sha256:77ff63ee69fd7ebef20df3a6d036ee4b8a4ffe0734efb0028c3b7643386e3ccf`
(index digest), sglang build `38d9e4a3fe4a493b0ae468477a274e83defda81a`
(`nightly-dev-20261102-38d9e4a3`). See each task's SOURCES.md for AMD64 digest,
PR #46018 provenance and cookbook citations.

Hardware: one Modal node with H200 x8 for each arm. Load: `random-ids`, input/output
8192/1024, range ratio 0.8, concurrency 1, 2, 4, 8, 16 in ascending order;
`num_prompts = 10 * concurrency`, `warmup_requests = 2 * concurrency`, seed 42,
request rate inf, ignore EOS, flush radix cache after warmup. Load and verifier
settings are identical across arms.

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
14400 seconds. No GPU run was performed when adding this experiment.

Run arms back to back on the same Modal GPU type (H200:8), with no competing
benchmark traffic. The listed order is off then on; record actual trial order and
hardware placement since serial concurrency does not guarantee identical hosts.
For repeats, use a fresh job name (`--job-name`) and alternate the arm order in a
local job copy to assess order/cache effects. Do not change either task's load.

## Record and compare

Keep Harbor outputs under `runs/` (ignored by git). Record the job/trial IDs, actual
order and timestamps, GPU details, task revision, image digest and exact launch
commands. Preserve both trials' `/app/results` artifacts: `run_meta.txt`,
`server.log`, `bench.log` and raw `bench.jsonl`; also retain the serve verifier's
`server_info.json` and each bench
verifier's `metrics.json`, reward and errors. Harbor's artifact/log directories
hold these files; never replace raw measurements with a hand-edited summary.

Compare only successful points at matching concurrency. Produce a comparison table
with both raw values and `100 * (on - off) / off` for:

- Output tokens/s (`output_throughput`).
- Per-GPU throughput: total, output and input throughput divided by 8
  (`tput_per_gpu`, `output_tput_per_gpu`, `input_tput_per_gpu`).
- TTFT, TPOT and ITL p50/p99 in ms (`median_*_ms` and `p99_*_ms`).
- DSPARK `accept_length`, if bench_serving reports it (otherwise record unavailable).

Flag any absolute relative change **greater than 5%** at a matching point, in either
direction; higher throughput and lower latency are favorable. An off value of zero
or a missing metric is unavailable for relative comparison. Flags request review,
not a reward failure or a statistical significance claim. Verifiers grade result
completeness and load only. Keep repeat-level results rather than selecting best runs.

## Known risks

- zhilin chose latest dev instead of cookbook `dev-dsv41`. V4.1 on this pinned dev
  image is unverified; the original cookbook cell's verified marker does not apply.
- Humming + TP8/EP8 + CUDA graph + DSPARK together is unverified. H200:8 NCCL under
  Modal gVisor is also unverified. Startup or runtime failure is a result to record.
- DSPARK acceptance affects TPOT: compare acceptance length alongside latency.
  The existing verifier preserves `accept_length` when present without requiring it.
- Small prompt counts make tail percentiles noisy. Sequential runs can differ in
  placement, compilation, HF cache state and scheduling. Record those conditions.
- Client field compatibility with the pinned image remains unverified. Earlier
  client citations and warmup/tokenization differences from InferenceX are in SOURCES.md.
