# sglang-bench-template

Minimal Harbor task: launch an sglang server with model X and flags Y, run one
`sglang.bench_serving` pass at a fixed load, keep the result JSON, and record
throughput / TTFT / TPOT / ITL (p50, p99). The verifier checks that the benchmark ran and
its JSON is complete. It applies no performance thresholds.

## Fill in (all knobs live in `task.toml`)

`rg -n TODO task.toml instruction.md` lists every placeholder:

| Placeholder | Where |
| --- | --- |
| image | `[environment].docker_image` |
| GPU count / type | `[environment].gpus`, `gpu_types` |
| model | `[environment.env].MODEL_PATH` |
| launch command (flags Y) | `[environment.env].SGLANG_LAUNCH_CMD` (verbatim from the source) and `instruction.md` |
| load | `[environment.env].BENCH_*` **and** the matching `[verifier.env].EXPECTED_*` |

For an A/B comparison (for example, a flag on vs. off), copy the task twice. The copies differ
only in `SGLANG_LAUNCH_CMD` and `task.name`, so a reviewer can diff the two `task.toml` files.

## Layout

```
task.toml            config + every knob (env vars)
instruction.md       what the agent must do (literal launch command)
solution/solve.sh    oracle: start server, wait /health, run bench_serving -> /app/results/
tests/Dockerfile     CPU-only verifier image (python:3.12-slim, stdlib only)
tests/test.sh        reward.txt = 1 iff check_result.py passes
tests/check_result.py  parse bench.jsonl, require finite metrics, check load, write metrics.json
```

Outputs: `/app/results/{bench.jsonl,server.log,bench.log,run_meta.txt}` (artifact) and
`/logs/verifier/{reward.txt,metrics.json}`.

## Run

```bash
harbor lint .
harbor run -p . -a oracle -e docker     # GPU host with docker + nvidia runtime
```
