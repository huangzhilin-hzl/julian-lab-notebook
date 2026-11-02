# sglang-bench-template

Minimal multi-step Harbor task: launch an sglang server with model X and flags Y, run one
`sglang.bench_serving` pass per concurrency in `BENCH_CONC_LIST` at a fixed load, keep the result JSON
lines, and record throughput (total and per GPU) / TTFT / TPOT / ITL / E2E (mean, p50, p90, p99) and
interactivity (1000/TPOT). No performance thresholds: numbers are recorded, not judged.

## Steps

Both steps run in one container, in order (`[[steps]]` in `task.toml`).

| Step | Agent / oracle does | Verifier | Gate |
| --- | --- | --- | --- |
| `serve` | start the server detached (`setsid nohup`), wait for `/health` | **shared** (agent container): `/health` = 200 and `/server_info` returns JSON | `min_reward = 1.0` |
| `bench` | step healthcheck re-checks `/health`, then one `bench_serving` run per `BENCH_CONC_LIST` entry | **separate** CPU-only container: `check_result.py` on the `/app/results` artifact | — |

Trial reward uses `multi_step_reward_strategy = "final"`: bench's 0/1, or serve's 0 if the server
never came up. If the bench healthcheck fails, bench has no verifier result and the trial has no reward.

## Caveats

1. Harbor documents that the filesystem persists between steps, not processes. The container
   is not restarted between steps (`src/harbor/trial/multi_step.py`), so the detached server
   should still be running in `bench`. This is unverified until the first real GPU run; the bench
   healthcheck fails fast if it is not.
2. The serve verifier must share the agent container, because the server listens only there.
   The task ignores lint rule `shared-verifier` for that reason (`[metadata.harbor.lint]`).

## Fill in (all knobs live in `task.toml`)

`rg -n TODO task.toml steps/` lists every placeholder:

| Placeholder | Where |
| --- | --- |
| image | `[environment].docker_image` |
| GPU count / type | `[environment].gpus`, `gpu_types` |
| model | `[environment.env].MODEL_PATH` |
| launch command (flags Y) | `[environment.env].SGLANG_LAUNCH_CMD` (verbatim from the source) and `steps/serve/instruction.md` |
| load | `[environment.env].BENCH_*` (incl. `BENCH_CONC_LIST`) **and** the matching `EXPECTED_*` / `NUM_GPUS` in the bench step's `[steps.verifier.env]` |
| timeouts | `[steps.agent].timeout_sec` per step (serve covers weight load + CUDA graph capture) |

For an A/B comparison (for example, a flag on vs. off), copy the task twice. The copies differ
only in `SGLANG_LAUNCH_CMD` and `task.name`, so a reviewer can diff the two `task.toml` files.

## Layout

```
task.toml                        config, every knob (env vars), [[steps]]
steps/serve/instruction.md       start the server, leave it running
steps/serve/solution/solve.sh    oracle: detached server, wait /health, write server.pid
steps/serve/tests/test.sh        reward.txt = 1 iff check_server.py passes
steps/serve/tests/check_server.py  /health + /server_info (falls back to /get_server_info)
steps/bench/instruction.md       literal bench_serving loop
steps/bench/solution/solve.sh    oracle: one bench_serving run per concurrency -> /app/results/bench.jsonl
steps/bench/tests/Dockerfile     CPU-only verifier image (python:3.12-slim, stdlib only)
steps/bench/tests/test.sh        reward.txt = 1 iff check_result.py passes
steps/bench/tests/check_result.py  one line per concurrency, finite metrics, load matches, metrics.json
```

Outputs: `/app/results/{server.log,server.pid,run_meta.txt,bench.jsonl,bench.log}` (artifact, collected
after each step), serve `/logs/verifier/{reward.txt,server_info.json}`, bench
`/logs/verifier/{reward.txt,metrics.json}`.

## Run

```bash
harbor lint .
harbor run -p . -a oracle -e docker     # GPU host with docker + nvidia runtime
```

`steps/bench/solution/solve.sh` and `steps/bench/tests/check_result.py` are byte-identical across
tasks; only `task.toml` and the instructions differ.
