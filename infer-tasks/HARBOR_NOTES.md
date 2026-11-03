# Harbor task format notes (sglang inference benchmarks)

Version read: `github.com/harbor-framework/harbor` (formerly `laude-institute/harbor`), commit
`00dc6a7003be8202e4e5ef6ac79c347c1b26033f` (2026-11-02 11:12 UTC+8), package `harbor 0.25.0`.
Doc paths below are relative to `docs-mintlify/` at that commit; source paths to `src/harbor/`.

## 1. Layout

Single-step (`tasks/overview.mdx`):

```
my-task/
├── instruction.md      # agent prompt (required, non-empty)
├── task.toml           # config + metadata
├── environment/        # Dockerfile / docker-compose.yaml; optional if task.toml sets docker_image
├── solution/solve.sh   # optional oracle (tasks/solution.mdx)
└── tests/test.sh       # verifier entry point; may carry tests/Dockerfile in separate mode
```

Multi-step (`tasks/multi-step.mdx`), used by the tasks in this directory: the task-root
`instruction.md`, `solution/`, and `tests/` are replaced by `steps/<name>/{instruction.md,
solution/solve.sh, tests/test.sh, workdir/}`, declared in order as `[[steps]]` in `task.toml`.
An optional task-level `tests/` holds shared helpers, overlaid by each step's `tests/`.

## 2. task.toml fields (`tasks/configuration.mdx`, `models/task/config.py`)

- Top level: `schema_version` (`"1.4"`), `artifacts = [...]`, `source`, `multi_step_reward_strategy`
  (`"mean"` | `"final"`, `MultiStepRewardStrategy`).
- `[task]`: `name` (`org/name`), `description`, `authors`, `keywords`, `version`.
- `[metadata]`: free-form. `[metadata.harbor.lint]` configures lint (`ignore`, `extend-select`).
- `[agent]`: `timeout_sec`, `user`, `network_mode`, `allowed_hosts`.
- `[verifier]`: `timeout_sec` (default 600), `env`, `user`, `environment_mode = "shared" | "separate"`,
  `[verifier.environment]` (same schema as `[environment]`; implies `"separate"`, conflicts with `"shared"`).
- `[environment]`: `docker_image`, `cpus`, `memory_mb`, `storage_mb`, `gpus`, `gpu_types`,
  `network_mode` (public / allowlist / no-network), `env`, `healthcheck`, `workdir`, ...
- `[environment.env]`: persistent env for every exec in the agent sandbox (agent, oracle,
  healthchecks, shared verifier). Values support `${VAR}` / `${VAR:-default}`, resolved on the host
  running harbor (`environments/base.py` L391-402).
- `[[steps]]` (`StepConfig`): `name`, `agent`, `verifier`, `min_reward` (number gates key `reward`;
  a table gates each key), `healthcheck` (`HealthcheckConfig`: `command`, `interval_sec`,
  `timeout_sec`, `start_period_sec`, `start_interval_sec`, `retries`), `artifacts`.
- `harbor task schema` prints the full JSON Schema.

## 3. Rewards and verifiers (`tasks/verifier.mdx`, `verifier/verifier.py`)

- `test.sh` writes `/logs/verifier/reward.txt` (one number) or `/logs/verifier/reward.json` (flat
  label -> number); json wins if both exist. (`harbor task init` comments say `rewards.json`; the
  code path is `reward.json`, `models/trial/paths.py`.)
- Shared mode: tests are uploaded to `/tests/` in the agent container. Separate mode: a new
  sandbox; image precedence `[verifier.environment].docker_image` > `tests/Dockerfile` > agent image
  (`tasks/separate-verifier.mdx`). Agent outputs reach it only via `artifacts` or `/logs/artifacts/`,
  restored at their original paths (`tasks/artifacts.mdx`).

## 4. Multi-step behavior (`trial/multi_step.py`, `models/task/verifier_mode.py`)

- All steps run in one agent environment; it is not restarted between steps. The docs promise
  filesystem persistence only. Process persistence (our server surviving from `serve` into
  `bench`) is not documented; it held on Modal 1×L4 (section 9). Each step's oracle runs
  `solve.sh` through `environment.exec`, so the server must be detached (`setsid nohup`, all
  std streams redirected) for that exec to return.
- Step verifier mode: `[steps.verifier].environment_mode` if set; else `[steps.verifier.environment]`
  implies `"separate"`; else the task-level mode; default `"shared"` (`resolve_step_verifier_mode`).
  Mixed shared/separate steps are supported.
- Separate verifier on the **last** step: Harbor stops the agent environment before verifying, so
  the server is torn down there.
- Separate image for a step: step `tests/Dockerfile` > task verifier image > task `tests/Dockerfile`
  > agent image. A bundled step image does not receive task-level `tests/`.
- `min_reward`: a missing verifier result or key counts as `-inf`; failure skips the remaining
  steps. A step that errors (for example its healthcheck fails) without a verifier result also stops.
- Trial reward: `"final"` = last executed step's verifier result (`None` if that step has none);
  `"mean"` = per-key mean over steps that have a result (`trial/trial.py` `_select_multi_step_reward`).
- `steps.verifier.timeout_sec` defaults to 600 and does not inherit the task-level value; set it per step.
- Task-level `[verifier.env]` is merged into every step's verifier, so step-specific vars go in
  `[steps.verifier.env]`.
- Between steps Harbor empties `/logs/agent` (unless resuming) and, after a shared verifier,
  `/logs/verifier`. Keep server logs under `/app/results`.

## 5. CLI (`harbor --help`, 0.25.0)

- `harbor lint <task>`: static checks, no sandbox (`tasks/lint.mdx`). `shared-verifier` is a default
  rule; `root-agent` and `unpinned-image` are opt-in.
- `harbor check <task>`: LLM agent reviews the task against a rubric inside a sandbox (default
  docker + claude-code); needs docker and an API key.
- `harbor run -p <task> -a oracle -e <env>`: runs each step's `solve.sh` then its verifier.

## 6. GPUs, images, k8s (`tasks/resources.mdx`, `environments/`)

- `gpus` / `gpu_types` are passed to the provider. Providers with GPU allocation: `beam, daytona,
  gke, modal, opensandbox, prime, docker`.
  - docker: compose nvidia device reservation of `count: gpus`; ignores `gpu_types` (docker/docker.py L229).
  - gke: `nvidia.com/gpu` requests/limits, `gpu_types[0]` -> `cloud.google.com/gke-accelerator`
    (h200 -> `nvidia-h200-141gb`), 1200 s pod-ready timeout with GPUs. Requires `gcloud` and
    GCP registry config. Pods get no volume mounts (no PVC) and no shm/ipc settings.
  - openshift and ack are k8s-based but allocate no GPUs.
- No `task.toml` field for `--shm-size` / `--ipc=host` (cookbook `docker run` flags). For the docker
  provider an `environment/docker-compose.yaml` overlay on service `main` may work (`tasks/multi-container.mdx`);
  untested.
- Custom sandbox: implement `BaseEnvironment`, use `-e module.path:ClassName` (`sandboxes/custom-sandboxes.mdx`).

## 7. Design choices in these tasks

- Every knob lives in `[environment.env]`; scripts hardcode nothing. The humming A/B pair differs at runtime only in
  the MoE runner token of `SGLANG_LAUNCH_CMD`; identity, metadata and docs may also differ.
  `../infer-experiments/dsv41-humming-ab/check_arms.py` checks the allowed differences.
- Steps: `serve` (shared verifier: `/health` + `/server_info`, `min_reward = 1.0`, long agent timeout
  for weight loading) then `bench` (step healthcheck on `/health`, separate CPU-only verifier on the
  `/app/results` artifact). Lint rule `shared-verifier` is ignored per task for the serve step.
- `multi_step_reward_strategy = "final"`: the trial reward is the benchmark's 0/1. `"mean"` would give
  0.5 for "server up, benchmark broken" and 1.0 if bench never got a verifier result.
- Load points: one `bench` step loops `BENCH_CONC_LIST` (ascending) against the same warm server,
  one `bench_serving` run per concurrency (as InferenceX's multi-node `srt-sweep` does). One step per
  point would duplicate instruction/solve/tests per point, and `final` reward would see only the last.
  `--flush-cache` clears the radix cache after each run's warmup.
- The bench verifier checks "ran, JSON complete, load matches" per concurrency; **no performance
  thresholds**. Metrics go to `/logs/verifier/metrics.json` as `{"points": [...]}`, one per concurrency,
  with InferenceX-style `*_tput_per_gpu` and `*_intvty = 1000/tpot_ms` (InferenceX names); `reward.txt` is 0/1.
- Result keys from sglang `python/sglang/benchmark/serving.py` L1855-1908 (main `4ae54e9`). `--output-file`
  appends JSONL, so the verifier requires exactly one line per `EXPECTED_CONC_LIST` entry, in order,
  and `solve.sh` deletes `bench.jsonl` first.
- `--dataset-name random` downloads ShareGPT from HF (datasets/random.py L84-92); `random-ids` does not.
  `sglang.bench_serving` is deprecated on main in favor of `sglang.benchmark.serving`; the old name works.
- `/get_server_info` is deprecated on sglang main (`d952161`) in favor of `/server_info`; the serve
  verifier tries `/server_info` first.

## 8. Validation (2026-11-02, harbor 0.25.0)

- `harbor lint sglang-bench-template dsv41-flash-h200-fp4-lowlatency`: no findings, exit 0
  (re-run after the concurrency loop change).
- `harbor lint --extend-select ALL`: only opt-in findings (root-agent per step, unpinned
  `docker_image`, unpinned `FROM` in `steps/bench/tests/Dockerfile`) plus the ignored `shared-verifier`.
- Harbor's own `Task` / `resolve_step_verifier_mode` parse both tasks: serve -> shared, bench -> separate
  built from `steps/bench/tests/Dockerfile` with `gpus = 0`.
- Local smoke test without docker/GPU, using a fake HTTP server as `SGLANG_LAUNCH_CMD` and a fake bench
  module: serve `solve.sh` returns while the detached server keeps running; serve verifier, bench
  healthcheck, bench `solve.sh` and verifier all pass; after killing the server the healthcheck and
  serve verifier fail. `check_result.py` synthetic fixtures (not benchmark data) fail as expected on
  load mismatch, two lines, null metric, missing file. After the loop change: the dsv41 env produces 5
  lines and passes; a missing concurrency, wrong order, wrong `completed`, wrong ISL/OSL, ratio, or a
  non-finite p90 all fail.
- Not run: `harbor check` (needs docker + LLM API key). `harbor run` on Modal: section 9.

## 9. Running on Modal

Job configs: `jobs/modal-template.yaml`, `jobs/modal-dsv41.yaml` (task paths are relative; run from
`infer-tasks/`).

```bash
pip install 'harbor[modal]'   # plain `modal` lacks dockerfile-parse: separate verifier fails with
                              # ModuleNotFoundError: No module named 'dockerfile_parse'
export MODAL_TOKEN_ID=ak-... MODAL_TOKEN_SECRET=as-...   # or `modal token set`
harbor run -c jobs/modal-template.yaml -y
```

- `gpu_types[0]` is a Modal GPU name; the sandbox requests `<gpu_types[0]>:<gpus>` (e.g. `H200:8`).
  GPU sandboxes run under gVisor; there is no shm/ipc setting.
- The default environment start timeout (600 s, `[environment].build_timeout_sec`) is too short for
  GPU queueing plus the sglang image pull (~15 GB). The job configs set
  `environment_build_timeout_multiplier` (2 for the template, 3 for dsv41); on the CLI use
  `--environment-build-timeout-multiplier 2` or more.
- `environment.kwargs.sandbox_timeout_secs` caps each sandbox's lifetime (Harbor default 24 h).
- Weights: `modal-dsv41.yaml` mounts the Modal Volume `hf-cache` at `/root/.cache/huggingface`
  so reruns skip the download. Harbor does not create it: run `modal volume create hf-cache` once.
  The model is public, so no HF token secret is configured.
- Verified: the server process survives `serve` -> `bench` (same PID, one server start in the log);
  the shared-mode serve verifier reaches `127.0.0.1`; the separate-mode bench verifier receives
  `/app/results` and `check_result.py` passes.
- Not verified: NCCL TP=8 under gVisor on H200:8 (the dsv41 task).
- Smoke run (2026-11-02, filled copy of the template): 1×L4, Qwen2.5-0.5B-Instruct, sglang v0.5.23
  (`lmsysorg/sglang:v0.5.23-runtime`), reward 1.0.
