# KDA B300 性能与精度复现

核查日期：2026-09-30。本文对应 NVIDIA 的 [KDA 优化 Kimi Delta Attention 文章](https://nvlabs.github.io/kda/blog/2026-09-27-kda-for-kda/)，整理发布 kernel 的性能与精度复测步骤，以及六层正确性验收的含义。

推荐使用当前修复后的发布版本，在 B300 上测量六个 workload，并单独回放真实输入。原环境有一个已确认无法获取的依赖 SHA；本文提供候选修复，但尚未安装、编译或运行 GPU 测试。文中的参考数字来自官方报告，不能作为本地实测结果。

## 复现范围与参考结果

KDAgent 指 Kernel Design Agents，KDAttn 指 Kimi Delta Attention 算子。本文的主要目标是复测已生成的 KDAttn forward kernel；重新运行 agent 搜索另见后文。

| 实现 | 博客报告的几何平均加速 | 当前公开 benchmark 报告 | 当前内部 judge 报告 |
| --- | ---: | ---: | ---: |
| TIRx | 2.96× | 2.963× | 2.950× |
| CuTe DSL | 2.85× | 2.799× | 2.786× |
| 静态 PTX | 2.94× | 2.927× | 2.914× |

以上加速均以 FlashKDA forward 为基线。算子 GPU 时间的变化不能直接换算成完整模型的吞吐变化。当前发布版补充了精度、状态交接和支持范围修复，复测时应对照所用版本的结果，而不是要求所有历史数字完全一致。

依据：[当前发布说明](https://github.com/NVlabs/kda/blob/ea37ebaff74c88a2545751dcb8ea8ef6c6251b67/README.md)、[早期发布说明](https://github.com/NVlabs/kda/blob/6c7068bc4ce0b6616ac1ec9b842b70c0ea2746ed/README.md)。

## 硬件与软件条件

以下操作在 Linux GPU 节点执行，推荐 Ubuntu 24.04。

| 条件 | 要求 |
| --- | --- |
| GPU | 对照博客性能使用 B300，架构 sm_103a |
| 其他 GPU | 发布说明支持 Blackwell sm_100a / sm_103a；B200 的 CuTe/TIRx 性能需重新测量 |
| 静态 PTX | 发布版本仅支持 B300 |
| NVIDIA 环境 | 支持 CUDA 13 的驱动，以及包含 nvcc 的 CUDA 13 toolkit |
| Python | 3.12；项目约束为 >=3.12,<3.13 |
| 构建工具 | C++17 编译器、CMake >=3.18、LLVM 18 开发文件 |
| 环境管理 | uv >=0.8 |

不要只安装 PyTorch 的 CUDA wheel：FlashKDA 和 TVM 的源码构建还需要系统 toolkit 和构建工具。

安装系统构建依赖，并确认 uv、驱动和 toolkit：

```bash
sudo apt-get install -y llvm-18-dev cmake build-essential \
  libxml2-dev libzstd-dev zlib1g-dev

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export FLASH_KDA_CUDA_ARCHS=103a

uv --version
nvidia-smi
nvcc --version
llvm-config-18 --version
```

如果 toolkit 不在 /usr/local/cuda，先将 CUDA_HOME 改为实际路径。FLASH_KDA_CUDA_ARCHS=103a 是 B300 配置；B200 对应 100a。

## 固定发布版本

建议将源码放在 GPU 节点的实验工作目录，日志和生成图也留在该 checkout 中。

```bash
git clone --branch 260927-kda-for-kda \
  https://github.com/NVlabs/kda.git kda-repro
cd kda-repro
git checkout ea37ebaff74c88a2545751dcb8ea8ef6c6251b67

mkdir -p results
git rev-parse HEAD > results/release_commit.txt
```

该版本中的主要依赖如下。实际安装以 pyproject.toml 和 uv.lock 为准。

| 依赖 | 原发布版本 |
| --- | --- |
| torch | 2.12.1+cu130 |
| nvidia-cutlass-dsl | 4.7.0 |
| FlashKDA | 7afb9f454f160a6c4bbc0999beca0a8c40a38934 |
| apache-tvm | ed5e2fed3f32030d9407e72751550b26653c5fac |
| apache-tvm-ffi | be35ec1297d6ddf388b7feb4f2e89d2cf030dda8 |
| tirx-kernels | 65d9a075a589e17b519d70f22c664a715dcc1ffa |
| fla-core | 0.5.2 |
| flashinfer-python | 0.6.13 |
| cupti-python | 13.0.1 |
| nvidia-cuda-nvcc / ptxas | 原 lock 为 13.2.86；PTX 路径要求 ptxas >=13.2 |

依据：[原发布依赖配置](https://github.com/NVlabs/kda/blob/ea37ebaff74c88a2545751dcb8ea8ef6c6251b67/pyproject.toml)。

## 原环境安装与当前阻碍

严格使用原 lock 的安装命令：

```bash
uv sync --locked --python 3.12
```

截至核查日期，tirx-kernels 的原 SHA 无法从公开 upstream 获取。实际执行 git fetch 后返回：

```text
fatal: remote error: upload-pack: not our ref 65d9a075a589e17b519d70f22c664a715dcc1ffa
```

fresh clone、公开历史和 GitHub commit API 的检查均未找到该提交；TVM、TVM FFI 和 FlashKDA 的锁定提交可以 fetch。原 SHA 不可获取的原因没有官方说明。

项目会一起安装各实现的依赖，因此即使只准备测 CuTe，直接使用这份原 lock 仍会遇到该阻碍。严格重建原环境，需要维护者恢复原 SHA，或提供与它对应的源码归档、wheel 及版本证据。

## 候选依赖修复

如果接受修改依赖后的复测，可将 tirx-kernels 固定到公开 v0.1.2 对应的提交：

```text
909d8e0b9889dc49ac69abf4fda7a88a48389def
```

原 kernel 使用的主要 tirx_lite API 和调用签名已做静态核对。公开库中还有注明源自本次 release 的 KDA 实现，但其中包含调整。因此，这个 pin 是候选修复，不能称为原环境的保真重建，也不能据此宣称已经兼容。

候选包新增了 kcoral 依赖，原 lock 中没有它。必须重新解析依赖，不能只替换 uv.lock 中的 SHA。

在发布 checkout 的根目录执行：

```bash
cp pyproject.toml pyproject.toml.release
cp uv.lock uv.lock.release

python3 - <<'PY'
from pathlib import Path

path = Path("pyproject.toml")
text = path.read_text()
old = "65d9a075a589e17b519d70f22c664a715dcc1ffa"
new = "909d8e0b9889dc49ac69abf4fda7a88a48389def"
assert text.count(old) == 1
path.write_text(text.replace(old, new))
PY

uv lock --python 3.12 --upgrade-package tirx-kernels
git diff -- pyproject.toml uv.lock
```

检查 diff，确认 TVM、FFI、FlashKDA 等原有 pins 是否保留，并记录新增依赖或其他版本变化。新增约束可能造成连带更新。确认后再安装：

```bash
set -o pipefail
uv sync --locked --python 3.12 2>&1 | tee results/install.log

git diff -- pyproject.toml uv.lock > results/dependency_patch.diff
cp pyproject.toml results/pyproject.used.toml
cp uv.lock results/uv.used.lock
```

官方发布说明估计首次源码构建需要 30–60 分钟；实际时间取决于网络、CPU 和缓存。

依据：[候选包依赖](https://github.com/mlc-ai/TIRx-kernels/blob/909d8e0b9889dc49ac69abf4fda7a88a48389def/pyproject.toml)、[tirx_lite API](https://github.com/mlc-ai/TIRx-kernels/blob/909d8e0b9889dc49ac69abf4fda7a88a48389def/tirx_kernels/tirx_lite/entry.py)、[KDA 实现来源记录](https://github.com/mlc-ai/TIRx-kernels/blob/909d8e0b9889dc49ac69abf4fda7a88a48389def/tirx_kernels/kda/kda_forward_portfolio_multishape.py#L4039)、[uv 更新锁定依赖的说明](https://docs.astral.sh/uv/concepts/projects/sync/#upgrading-locked-package-versions)。

## 安装检查与环境记录

```bash
uv run --locked python - <<'PY'
import cutlass
import flash_kda
import fla
import torch
import tvm
import tirx_kernels.tirx_lite

print("imports OK")
print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name())
print("capability:", torch.cuda.get_device_capability())
print("CuTe:", cutlass.__version__)
print("TVM:", tvm.__version__)
PY

nvidia-smi > results/nvidia-smi.txt
nvcc --version > results/nvcc.txt
uv --version > results/uv.txt
uv pip freeze > results/python_packages.txt
git status --short > results/source_status.txt
```

导入成功只证明环境能加载。kernel 编译、正确性和性能仍由后续运行验证。测量时避免其他作业争用同一 GPU，并记录 GPU 功率、时钟和负载条件。

## 六个性能 workload

公开 benchmark 的配置为 B=1、K=V=128、H=64 或 96。每项总 token 数都是 8192。

| workload | H | 各序列长度 |
| --- | ---: | --- |
| h96-fixed | 96 | [8192] |
| h96-mixed_varlen | 96 | [1300, 547, 2048, 963, 271, 3063] |
| h96-uniform_varlen | 96 | [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024] |
| h64-fixed | 64 | [8192] |
| h64-mixed_varlen | 64 | [1300, 547, 2048, 963, 271, 3063] |
| h64-uniform_varlen | 64 | [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024] |

输入 q/k/v/g/beta 为 BF16，initial_state 和 final_state 为 FP32，varlen 的 cu_seqlens 为 int64。输出既包含 attention output，也包含最终 recurrent state。

基线通过 FLA 调用 FlashKDA，包含 q/k L2 normalization、gate 激活、beta sigmoid 和 final-state 写回。公开 benchmark 的正确性参考使用 FLA Triton 路径，避免将计时基线同时作为唯一正确性标准。

计时采用 CUPTI、cold L2、CUDA Graph、3 次 warmup，以及每 trial 30 次迭代、3 个 trial。先取迭代时间中位数，再取各 trial 的中位数。prepare 阶段的编译和 host planning 不计入 launch 时间；TIRx 已录制私有 graph，不能嵌套 capture 时，计时器会尝试按已有 launch 计时。

依据：[workload 定义](https://github.com/NVlabs/kda/blob/ea37ebaff74c88a2545751dcb8ea8ef6c6251b67/bench.py#L37)、[基线调用](https://github.com/NVlabs/kda/blob/ea37ebaff74c88a2545751dcb8ea8ef6c6251b67/bench.py#L89)、[计时实现](https://github.com/NVlabs/kda/blob/ea37ebaff74c88a2545751dcb8ea8ef6c6251b67/bench.py#L143)。

## 运行性能复测

先移除可能禁用 FLA backend dispatch 的环境变量，再分别运行三个实现：

```bash
set -o pipefail
unset FLA_DISABLE_BACKEND_DISPATCH

uv run --locked python bench.py tirx \
  --warmup 3 --iters 30 --trials 3 --seed 0 \
  2>&1 | tee results/tirx_seed0.log

uv run --locked python bench.py cute \
  --warmup 3 --iters 30 --trials 3 --seed 0 \
  2>&1 | tee results/cute_seed0.log

uv run --locked python bench.py ptx \
  --warmup 3 --iters 30 --trials 3 --seed 0 \
  2>&1 | tee results/ptx_seed0.log
```

每个日志应包含六项 workload 的 FlashKDA ms、kernel ms、speedup，以及 output/state 的正确性结果。最后一行是几何平均：

```text
speedup_i = FlashKDA_ms_i / candidate_ms_i
geomean = exp(sum(log(speedup_i)) / 6)
```

确认全部六行是 PASS，最后一行是 all pass。该脚本即使打印 FAIL，也没有显式设置非零退出码；shell 命令成功不能代替结果检查。

公开检查要求 output 和 final_state 都返回、值有限、逐元素误差通过，并且整体 relative L2 <=0.03。逐元素条件约等价于：

```text
abs(candidate_i - reference_i)
    <= max(0.5 * RMS(reference), 0.05 * abs(reference_i))
```

这是该公开脚本的实际容差，不应理解成每个元素都必须满足纯 5% 相对误差。来源：[正确性实现](https://github.com/NVlabs/kda/blob/ea37ebaff74c88a2545751dcb8ea8ef6c6251b67/bench.py#L127)、[结果输出](https://github.com/NVlabs/kda/blob/ea37ebaff74c88a2545751dcb8ea8ef6c6251b67/bench.py#L232)。

若有新失败或结果明显偏离，再针对原因检查基线 dispatch、依赖变化、GPU 负载和计时路径。不要仅为得到目标数字而挑选最快的一次运行。

## 真实输入精度复测

官方脚本下载约 1 GB 的公开 trace：math500-multi-8192 样本中 layers 00、14、25，各层 32 heads，拼成 H96；实际序列长度为 8183 tokens。每个 kernel 与逐 token FP64 recurrence 比较。

```bash
set -o pipefail

uv run --locked --with huggingface_hub --with safetensors --with matplotlib \
  python scripts/real_workload_error_plots.py \
  2>&1 | tee results/real_workload_accuracy.log

cp figures/real_workload_error.json results/
cp figures/real_workload_accuracy.png results/
```

该脚本默认执行 FlashKDA、CuTe、TIRx 和 PTX，因此上述完整命令要求 B300。B200 不能直接执行其中的 PTX 路径。

| 实现 | 当前发布图的 final-state relative RMSE |
| --- | ---: |
| FlashKDA | 3.98% |
| CuTe | 0.23% |
| TIRx | 0.31% |
| PTX | 0.19% |

指标定义为 RMS(candidate - FP64 reference) / RMS(FP64 reference)。博客旧图的数值略有不同，复测时以固定版本的图和 JSON 为对照。

数据来自真实模型，但脚本将原始 softplus gate 转成 K3 ABI：A_log 和 dt_bias 设为 0，g 由 logit(clamp(-decay/5)) 生成；beta 也转换成 logits。数据说明指出约 2–3% 的原始 gate 条目深于 -5 nats，会被裁剪。因此这里验证转换后的算子，完整模型质量需要另测。

公开脚本只处理一个真实样本，不能代表博客的完整 151-case suite。依据：[精度脚本](https://github.com/NVlabs/kda/blob/ea37ebaff74c88a2545751dcb8ea8ef6c6251b67/scripts/real_workload_error_plots.py)、[公开数据说明](https://huggingface.co/datasets/humanfia-lab/kda-datasets/blob/main/kda-forward/README.md)。

### 固定真实数据版本

原脚本的 snapshot_download 没有指定 revision，会读取数据仓库的 main。需要固定数据时，可生成一份脚本副本，将 revision 固定为本次核查的 dataset SHA；不修改原始脚本。

```bash
python3 - <<'PY'
from pathlib import Path

source = Path("scripts/real_workload_error_plots.py")
target = source.with_name("real_workload_error_plots_pinned.py")
text = source.read_text()
old = 'snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=files, local_dir=data_dir)'
new = (
    'snapshot_download(HF_REPO, repo_type="dataset", '
    'revision="ee67fe82e1ccc41b01fcf7a864319f05cdb8e311", '
    'allow_patterns=files, local_dir=data_dir)'
)
assert text.count(old) == 1
assert not target.exists()
target.write_text(text.replace(old, new))
PY

set -o pipefail
uv run --locked --with huggingface_hub --with safetensors --with matplotlib \
  python scripts/real_workload_error_plots_pinned.py \
  2>&1 | tee results/real_workload_accuracy_pinned.log

cp scripts/real_workload_error_plots_pinned.py results/
cp figures/real_workload_error.json results/
cp figures/real_workload_accuracy.png results/
```

精度脚本和新增绘图库等依赖尚未在本次核查中执行；归档时同时记录实际使用的辅助包版本。

## 六层验收的含义与公开复测的边界

文章中的 acceptance gauntlet 用于防止 agent 通过测试漏洞获得虚假加速。全部关卡通过才支持完整验收结论；它们不是六个性能 workload 的另一种写法。

| 关卡 | 要验证什么 | 典型失效方式 |
| --- | --- | --- |
| 动态随机输入与隐藏分布 | 换随机种子，也换输入分布和序列布局 | 用高斯分布的预期范数代替实际 q/k 归一化；记住固定序列边界 |
| 严格输入规范 | 每个合法输入都遵循数学定义，允许按 shape 选择不同实现 | 忽略初始状态；丢弃历史；只支持测试中恰好出现的布局 |
| CUDA Graph replay 检查 | 计时后写入新的输入内容，重放同一 graph 并验证新结果 | 缓存旧答案；根据 tensor 地址返回预先计算的结果 |
| 极端输入压力测试 | 深 gate 衰减、饱和 gate、非整块长度、重复 key | 下溢、溢出、NaN、边界掩码错误 |
| 严格逐元素容差 | output 和 final state 的每个元素都满足规定容差 | 用平均误差或 99.9% 通过率掩盖少数严重错误 |
| 真实模型 trace 回放 | 检查真实数值范围，并将 evaluator 与生成 kernel 的环境隔离 | 合成数据过拟合；agent 接触测试数据后针对性优化 |

Graph 通常绑定捕获时的地址。实现新输入回放检查时，应将新数据写入已绑定的 input buffer，或使用合法的 graph 参数更新；只更换 Python 变量不足以证明 graph 已读取新数据。候选应重新计算新 output 和 final_state。

公开 bench.py 使用固定种子的六个合成 workload，并在计时前检查候选输出；它没有实现这里列出的全部隐藏分布、极端 probe、计时后换输入 replay 和隔离评测。公开精度脚本也只回放一个样本。

官方 README 报告内部 judge 为 24/24，但内部 judge 本体没有随本次 kernel release 一起提供。文档记录结果时，应使用“六项公开性能测试通过”“一个真实样本精度复测完成”等具体表述，不能直接写成“完整六层验收通过”。

## 重新运行 agent 优化

TIRx Harness 的 agent 流程现已公开。当前固定版本的 kda_forward 任务要求 B200，六个 shape 各有 standard 和 saturated 两种输入，共 12 个计分 case，并另外检查 stress、FP64 exact probes 和 salted holdout。

这个任务的硬件和评分口径与博客 B300 六项结果不同。它可用于重新开展优化实验；原 B300 搜索的全部 prompt、配置和逐次轨迹没有在 kernel release 中提供。

在符合该任务契约的 B200 环境中，准备 Linux x86_64、Python 3.12 或 3.13、pip >=25.1、CUDA toolkit、Cargo/Rust >=1.89 及源码构建工具，再执行：

```bash
git clone https://github.com/mlc-ai/TIRx-harness.git
cd TIRx-harness
git checkout ccd7a04a72aa121d98fd90b2a4a3fe97ca5a6726

git submodule update --init thirdparty/tvm-rust-ext
python -m pip install -r evolution/preparation/requirements.txt
python evolution/setup.py --task kda_forward --name kda-repro
```

setup 会输出 run_dir、进入 worktree 和激活环境的命令。按输出设置后，可参照官方 Humanize Flame Chase 入口启动。下面的 agent 字段和 run_dir 是占位符，需要换成已配置的值；24h 是文档示例的预算，不是博客实验的原始配置。

```bash
uv tool install git+https://github.com/humanfia/humanize.git

run_dir=/absolute/path/printed/as/run_dir
cd "$run_dir/worktree"
source .venv/bin/activate

hmz exec -f 'git+https://github.com/humanfia/flowverse#flame_chase' \
  -a 'first_chaser=<CLI/MODEL:EFFORT>' \
  -a 'second_chaser=<CLI/MODEL:EFFORT>' \
  -b duration=24h "$(cat "$run_dir/PROMPT.md")"
```

该命令没有锁定 Humanize 和 flowverse 版本；开展可重复的研究时，需要另外固定并记录它们、模型版本、预算和起始候选。使用生成的 PROMPT.md 中的正式评分命令，保留 manifest、frontier 和每次评测记录。

依据：[优化运行流程](https://github.com/mlc-ai/TIRx-harness/blob/ccd7a04a72aa121d98fd90b2a4a3fe97ca5a6726/docs/optimization-runs.md)、[安装条件](https://github.com/mlc-ai/TIRx-harness/blob/ccd7a04a72aa121d98fd90b2a4a3fe97ca5a6726/docs/installation.md)、[当前 KDA 任务契约](https://github.com/mlc-ai/TIRx-harness/blob/ccd7a04a72aa121d98fd90b2a4a3fe97ca5a6726/evolution/tasks/kda_forward.yaml)。

## 结果归档与完成标准

一轮发布 kernel 复测至少保留以下记录：

| 记录 | 内容 |
| --- | --- |
| 源码版本 | release commit、工作树修改、候选依赖 patch |
| 依赖环境 | 实际 pyproject、uv.lock、安装日志、包版本 |
| GPU 环境 | GPU 型号、驱动、toolkit、功率与时钟、其他负载 |
| 性能 | 六项 workload 的基线时间、候选时间、speedup、PASS 状态、geomean |
| 精度 | dataset revision、所用脚本、FP64 对照结果、JSON 和图 |
| agent 实验 | 若执行，记录任务契约、Harness、Humanize、flow、模型和预算 |

完成标准是：环境可构建，六项公开 workload 的 output/state 均通过检查，性能日志完整，真实样本完成 FP64 对照，并明确标注是否采用候选依赖修复。完整六层验收和模型端到端效果需要额外评测证据。

本次整理已经核查源码与版本，确认原依赖下载阻碍，检查候选 API 和命令参数；未执行依赖解析、源码构建、kernel 编译、GPU benchmark 或真实精度计算。
