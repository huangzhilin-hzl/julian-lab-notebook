# sglang-bench-template

Two-step Harbor 0.25.0 template: launch sglang, then run a fixed-load concurrency sweep.
Fill the placeholders in `task.toml` before running:

| Setting | Template value |
| --- | --- |
| Image / sglang build | `TODO_SGLANG_IMAGE`; pin a digest and record its build commit |
| Launch command | `SGLANG_LAUNCH_CMD = "TODO_LAUNCH_CMD"` |
| Model | `TODO_MODEL_PATH` |
| Hardware | 1× `TODO_GPU_TYPE`, 16 CPUs, 65536 MiB RAM, 204800 MiB storage |
| Load | `random-ids`, `TODO_ISL`/`TODO_OSL`, ratio 1.0, concurrency 1 |
| Sampling | 10× concurrency prompts, 2× concurrency warmups, seed 42, rate `inf`, ignore EOS |
| Cache | `--flush-cache` after warmup |

Set matching `EXPECTED_*` and `NUM_GPUS` values in the bench verifier environment.
Copy the exact launch command into `steps/serve/instruction.md`. Set timeouts for the
model and load; template agent deadlines are 3600 s for serve and 1800 s for bench.

From this directory, run on a configured GPU host:

```bash
harbor lint .
harbor run -p . -a oracle -e docker
```

For Modal credentials and job configuration, see [Harbor notes](../HARBOR_NOTES.md).

The serve oracle starts a detached server; its shared verifier checks `/health` and
`/server_info` (fallback `/get_server_info`), gating bench at reward 1.0. The bench
healthcheck requires the server to remain running. One benchmark invocation per
`BENCH_CONC_LIST` entry appends a JSONL point. A separate CPU-only verifier checks
completeness, finite metrics and matching load, without performance thresholds.
The trial uses the final step reward; a bench healthcheck failure has no verifier reward.

`/app/results/{server.log,server.pid,run_meta.txt,bench.jsonl,bench.log}` is collected
after each step. Verifier outputs include `reward.txt`, serve `server_info.json` and
bench `metrics.json`: throughput (total and per GPU), TTFT/TPOT/ITL/E2E latency
(mean/p50/p90/p99) and interactivity (1000/TPOT). Benchmark oracle and verifier
scripts are byte-identical across tasks.
