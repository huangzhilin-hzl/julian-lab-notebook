# molou/dsv41-flash-h200-fp4-lowlatency

sglang cookbook DeepSeek-V4.1 Flash, H200 · FP4 · Low-Latency · single node, as a multi-step Harbor
task. Built from `../sglang-bench-template` (see its README for the step layout and caveats).
Sources and line refs: `SOURCES.md`.

- `steps/serve`: the cookbook launch command (`SGLANG_LAUNCH_CMD`), shared verifier on `/health` and
  `/server_info`, `min_reward = 1.0`.
- `steps/bench`: one `bench_serving` pass, graded in a separate CPU-only container. The load is
  `TODO(load)`: the cookbook gives none for this cell.

Before running: fill every `TODO(load)` in `[environment.env]` (and the matching `EXPECTED_*` in the
bench step's `[steps.verifier.env]`), then

```bash
harbor lint .
harbor run -p . -a oracle -e <env>   # needs 8x H200; see ../HARBOR_NOTES.md for k8s
```

Open points: the server surviving from `serve` into `bench` is unverified until the first real run;
the cookbook's `docker run` flags `--shm-size 32g --ipc=host` have no `task.toml` field (SOURCES.md).
