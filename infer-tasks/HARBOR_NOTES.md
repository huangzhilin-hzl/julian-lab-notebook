# Harbor 任务格式笔记（面向 sglang 推理 benchmark）

读的版本：`github.com/laude-institute/harbor`（clone 后 docs 内链接已指向 `github.com/harbor-framework/harbor`），
commit `00dc6a7003be8202e4e5ef6ac79c347c1b26033f`（2026-11-02 11:12 UTC+8），包版本 `harbor 0.25.0`（PyPI 同版本）。
文档源：仓库 `docs-mintlify/`（下文路径都相对该 commit）。本机副本：`/workspace/harbor`。

## 1. 目录结构（`docs-mintlify/tasks/overview.mdx`）

```
my-task/
├── instruction.md      # 给 agent 的任务说明（必需，不能为空）
├── task.toml           # 配置 + 元数据
├── environment/        # Dockerfile / docker-compose.yaml / Apptainer.def；若 task.toml 设了 docker_image 可省略
├── solution/solve.sh   # 可选，oracle agent 用来证明任务可解（tasks/solution.mdx）
└── tests/test.sh       # verifier 入口（必需）；separate 模式下可带 tests/Dockerfile
```
- 还支持多步任务 `steps/step-N/{instruction.md,tests,solution,workdir/setup.sh}`（tasks/multi-step.mdx）。
- `tests/`、`solution/` 内可放 `.harborignore`（git ignore 语法）。
- `harbor task init org/name` 生成的脚手架：上述文件 + `tests/Dockerfile` + `README.md` + `.gitignore`，并默认 `verifier.environment_mode = "separate"`（本机实测）。

## 2. task.toml 字段（`tasks/configuration.mdx`，源码 `src/harbor/models/task/config.py`）

- 顶层：`schema_version`（当前 `"1.4"`）、`artifacts = [...]`（必须写在所有 section 之前）、`source`、`multi_step_reward_strategy`。
- `[task]`：可选；有则 `name`（`org/name`）必填，另有 `description`、`authors`、`keywords`、`version`。
- `[metadata]`：任意键值。`[metadata.harbor.lint]` 用来配置 lint 规则。
- `[agent]`：`timeout_sec`、`user`、`network_mode`、`allowed_hosts`。
- `[verifier]`：`timeout_sec`（默认 600）、`env`、`user`、`environment_mode = "shared"|"separate"`、`[verifier.environment]`（与 `[environment]` 同 schema）。
- `[solution.env]`：只在 oracle 执行 solve.sh 时注入。
- `[environment]`：`docker_image`、`build_timeout_sec`（默认 600）、`os`、`cpus`、`memory_mb`、`storage_mb`、**`gpus`、`gpu_types`**、`tpu`、`network_mode`（public/allowlist/no-network）、`allowed_hosts`、`env`、`healthcheck`、`workdir`、`mcp_servers`、`skills_dir`。
- `[environment.env]` 是该 sandbox 内所有 exec 的持久环境变量，值支持 `${VAR}` / `${VAR:-default}`，从**运行 harbor 的宿主机**取值（config.py `EnvironmentConfig.env`，`environments/base.py` L391-402）。
- `harbor task schema` 可打印完整 JSON Schema。

## 3. 打分 / reward（`tasks/verifier.mdx`，源码 `src/harbor/verifier/verifier.py`、`models/trial/paths.py`）

- `tests/test.sh` 跑完后必须写 `/logs/verifier/reward.txt`（单个数字）或 `/logs/verifier/reward.json`（标签→有限数值的扁平对象）；两者都有时优先 json。
- 注意：`harbor task init` 生成的 test.sh 注释写的是 `rewards.json`，但代码里的路径是 `reward.json`（paths.py L48/L286）。以代码为准。
- shared 模式：tests 上传到 agent 容器的 `/tests/` 后执行。separate 模式：另起一个 verifier sandbox。镜像选择优先级为 `[verifier.environment].docker_image` > `tests/Dockerfile` > agent 镜像（`tasks/separate-verifier.mdx`）。agent 产物只能通过 `artifacts` 或 `/logs/artifacts/` 传过去，并保持原路径（`tasks/artifacts.mdx`）。
- separate 模式支持对已有 trial 重新打分（regrade，`jobs/regrade.mdx`）。

## 4. 校验与运行（CLI，`harbor --help`，0.25.0 本机实测）

- `harbor lint <task>`：静态检查，不起 sandbox（`tasks/lint.mdx`）。默认规则包括 invalid-task-config、unknown-config-field（会提示拼写，实测把 `gpus` 写成 `gpuz` 会报错）、invalid-task-files、shared-verifier、unpinned-python-dependency、verifier-runtime-install、agent-image-test-leak 等。`root-agent` 和 `unpinned-image` 需要 `--extend-select` 才开。
- `harbor check <task>`：用 LLM agent（默认 claude-code + claude-sonnet-4-6）按 rubric 审任务质量，需要 API key，本次没跑。
- `harbor run -p <task> -a oracle -e <env>`：用 solve.sh 跑一遍；`-a <agent> -m <model>` 则让真 agent 跑。
- `harbor task start-env`：只把环境拉起来，便于调试。

## 5. GPU / 镜像 / k8s（`tasks/resources.mdx`，`src/harbor/environments/`）

- `gpus` 和 `gpu_types` 没有 enforcement policy，会直接透传给 provider。文档列出的支持 GPU 分配的 provider：`beam, daytona, gke, modal, opensandbox, prime, docker`。
  - docker：用 compose 的 nvidia device request 预留 `count: gpus` 张卡，**忽略 `gpu_types`**（docker/docker.py L229）。
  - gke：设置 `nvidia.com/gpu` 的 requests/limits，用 `gpu_types[0]` 映射 `cloud.google.com/gke-accelerator` 节点选择器（h200→`nvidia-h200-141gb`），有 GPU 时 pod ready 超时为 1200s（gke.py L69-83、L780-805、L893）。它强依赖 GCP：preflight 要求有 `gcloud`，构造函数需要 `cluster_name/region/namespace/registry_location/registry_name`。设了 `docker_image` 时用预构建镜像（gke.py L2085）。pod 的 `volume_mounts=[]`，不挂 PVC，也不设 shm/ipc。
  - openshift 和 ack 也是基于 k8s 的 provider，但不分配 GPU（openshift.py L179 `gpus=False`；resources.mdx 的 GPU 列表里没有它们）。
- 自定义镜像：`[environment].docker_image` 指定任意镜像，或者用 `environment/Dockerfile` 构建。
- 自定义 sandbox：实现 `BaseEnvironment`（start/stop/exec/upload_*/download_*），通过 `-e module.path:ClassName` 使用（`sandboxes/custom-sandboxes.mdx`）。

## 6. 本模板的设计取舍

- 所有可调参数都放在 `[environment.env]`，solve.sh 里不写死任何值。做 A/B 时复制任务，只改 `SGLANG_LAUNCH_CMD`，review 时直接 diff 两份 task.toml。
- separate verifier 用 CPU 小容器（`gpus = 0`），只读 artifact `/app/results`。verifier 只验证"跑了、JSON 完整、负载参数对得上"，**不设性能阈值**。指标写到 `/logs/verifier/metrics.json`，`reward.txt` 是 0/1。
- bench 结果字段取自 sglang `python/sglang/benchmark/serving.py` L1855-1908（sglang main `4ae54e9`，2026-11-02）。`--output-file` 是**追加**写 JSONL，所以 verifier 要求文件里恰好有 1 行。
- `--dataset-name random` 会从 HF 下载 ShareGPT（datasets/random.py L84-92），`random-ids` 不用下载（datasets/__init__.py L39、random.py L39）。`sglang.bench_serving` 在 main 上已标为 deprecated，实现迁到了 `sglang.benchmark.serving`，旧名仍然能用。

## 7. 本次校验结果（2026-11-02，harbor 0.25.0）

- `harbor lint sglang-bench-template dsv41-flash-h200-fp4-lowlatency`：默认规则全部通过，exit 0。
- `harbor lint --extend-select ALL`：只剩 3 条 opt-in 警告，分别是 root-agent、`docker_image` 没有 pin digest、`tests/Dockerfile` 的 FROM 没有 pin digest。
- 用 harbor 自己的 `TaskConfig` 能正常解析两份 task.toml，verifier 解析为 SEPARATE、gpus=0。
- `check_result.py` 用合成 fixture 自测过（不是 benchmark 数据）：完整结果 → 0；completed 不符、指标为 null、文件缺失 → 1。
- 没跑 `harbor run`（box 上没有 docker 和 GPU），也没跑 `harbor check`（需要 LLM API key）。
