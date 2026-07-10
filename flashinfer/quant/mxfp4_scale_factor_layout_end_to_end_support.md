# FlashInfer MXFP4 Scale-Factor Layout 端到端支持与 SM100 验收

## 1. 概要

分支 `molou/mxfp4-quantize-layout` 为 `mxfp4_quantize` 和 `mxfp4_dequantize` 增加 `sfLayout` 参数，支持以下 scale-factor layout：

- `SfLayout.layout_128x4`
- `SfLayout.layout_8x4`
- `SfLayout.layout_linear`

处理流程：

1. `mxfp4_quantize` 将 `sfLayout` 转换为 CUDA 或 CuTe-DSL 使用的 layout 参数。
2. 量化 backend 输出 packed MXFP4 tensor 和对应 layout 的 scale tensor。
3. `mxfp4_dequantize` 按相同 layout 读取 scale tensor。
4. 反量化结果为 `torch.float32` tensor。

默认 layout 为 `layout_128x4`，与原有接口行为一致。

## 2. 实现变更

| 模块 | 变更 |
| --- | --- |
| `flashinfer/quantization/fp4_quantization.py` | `mxfp4_quantize` 和 `mxfp4_dequantize` 增加尾部参数 `sfLayout`；CUDA backend 映射为 `is_sf_swizzled_layout` 和 `is_sf_8x4_layout` |
| `flashinfer/quantization/fp4_quantization.py` | 通用 `e2m1_and_ufp8sf_scale_to_float` 增加默认关闭的 `is_sf_8x4_layout`，并传递至 FFI |
| `flashinfer/quantization/kernels/mxfp4_quantize.py` | CuTe-DSL 接入三种 layout；按 row tile 和 scale-column tile 计算物理输出 shape |
| `csrc/nv_internal/tensorrt_llm/thop/fp4Op.cpp` | `computeSFIndex` 分别实现 128x4、8x4 和 linear 地址映射 |
| `csrc/nv_internal/tensorrt_llm/thop/fp4Op.cpp` | 反量化前校验 `sfVecSize`、hidden dimension、layout flags 和 scale buffer 大小 |
| `benchmarks/routines/quantization.py` | `mxfp4_quantize` routine 接入 `--sf_layout`；refcheck 使用相同 layout 反量化 |
| `benchmarks/bench_mxfp4_quantize_backend_comparison.py` | 三种 layout 的 CUDA/CuTe-DSL 正确性与性能对比 |
| `tests/utils/test_fp4_quantize.py` | 新增一个公共 API 功能测试，参数化三种 layout |

## 3. Scale-Factor Layout 规则

设输入 shape 为 `(M, K)`，MXFP4 scale vector size 为 32：

```text
G = K / 32
C = ceil(G / 4) * 4
```

| Layout | Scale shape |
| --- | --- |
| `layout_128x4` | `(ceil(M / 128) * 128, C)` |
| `layout_8x4` | `(ceil(M / 8) * 8, C)` |
| `layout_linear` | `(M, G)` |

约束与输出：

- `K % 32 == 0`。
- quantized tensor shape 为 `(M, K / 2)`，dtype 为 `torch.uint8`。
- scale tensor dtype 为 `torch.uint8`。
- dequantized tensor shape 为 `(M, K)`，dtype 为 `torch.float32`。
- `mxfp4_dequantize` 使用与量化阶段相同的 `sfLayout`。

## 4. 兼容性与范围

- 未传 `sfLayout` 时使用 `layout_128x4`。
- 通用 dequant 新增参数默认为 `False`，保持现有调用兼容。
- NVFP4 行为不变。
- 仓库既有 MXFP4 quantize trace template 和 decorator 不变；本分支不增加 layout trace schema、reference、example 或生成 JSON。
- 本分支不增加 MXFP4 dequantize trace。
- 测试代码只新增一个三 layout 公共 API 功能测试，不包含额外边界 shape、buffer 越界或拆分 backend 测试。

## 5. SM100 验收

### 5.1 测试版本

| 项目 | 要求 |
| --- | --- |
| Feature 仓库 | `https://github.com/huangzhilin-hzl/flashinfer` |
| Feature 分支 | `molou/mxfp4-quantize-layout` |
| Target 仓库 | `https://github.com/flashinfer-ai/flashinfer` |
| Target 分支 | `main` |
| GPU | NVIDIA SM100，compute capability `(10, 0)` |
| CUDA | `>= 12.8` |
| 输入 dtype | `float16`、`bfloat16` |
| Backend | `cuda`、`cute-dsl` |

验收使用两个 revision：

- `TARGET_SHA`：合并前的目标分支版本。
- `FEATURE_SHA`：待合并功能分支版本。

Feature 分支应包含当前 target 分支。以下命令动态记录两个 SHA，并校验 merge base：

```bash
set -euo pipefail

git clone \
  --branch molou/mxfp4-quantize-layout \
  --single-branch \
  --recursive \
  https://github.com/huangzhilin-hzl/flashinfer.git \
  flashinfer-mxfp4-quantize-layout

cd flashinfer-mxfp4-quantize-layout
git remote get-url upstream >/dev/null 2>&1 || \
  git remote add upstream https://github.com/flashinfer-ai/flashinfer.git
git fetch upstream main

TARGET_SHA="$(git rev-parse upstream/main)"
FEATURE_SHA="$(git rev-parse HEAD)"
MERGE_BASE_SHA="$(git merge-base "$FEATURE_SHA" "$TARGET_SHA")"

printf 'TARGET_SHA=%s\nFEATURE_SHA=%s\nMERGE_BASE_SHA=%s\n' \
  "$TARGET_SHA" "$FEATURE_SHA" "$MERGE_BASE_SHA"
test "$MERGE_BASE_SHA" = "$TARGET_SHA"

export RUN_ROOT="$(cd .. && pwd)"
export BASE_TREE="$RUN_ROOT/flashinfer-mxfp4-base"
export FEATURE_TREE="$RUN_ROOT/flashinfer-mxfp4-feature"
export RESULT_DIR="$RUN_ROOT/flashinfer-mxfp4-results"

mkdir -p "$RESULT_DIR"
git worktree add --detach "$BASE_TREE" "$TARGET_SHA"
git worktree add --detach "$FEATURE_TREE" "$FEATURE_SHA"
```

GPU 环境检查：

```bash
python3 - <<'PY'
import torch

cc = torch.cuda.get_device_capability()
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name())
print("compute capability:", cc)
assert cc == (10, 0)
PY
```

### 5.2 合并前后完整回归

Baseline 和 feature 分别运行各自 revision 的完整 FP4 测试：

```bash
run_full_fp4() {
  label="$1"
  tree="$2"
  (
    set -euo pipefail
    git -C "$tree" submodule update --init --recursive
    python3 -m pip install --no-build-isolation -e "$tree" -v
    cd "$tree"
    EXPECTED_FLASHINFER_ROOT="$tree" python3 - <<'PY'
import os
from pathlib import Path
import flashinfer
from flashinfer.cute_dsl import is_cute_dsl_available

expected = Path(os.environ["EXPECTED_FLASHINFER_ROOT"]).resolve()
loaded = Path(flashinfer.__file__).resolve()
print("flashinfer:", loaded)
print("CuTe-DSL:", is_cute_dsl_available())
assert expected in loaded.parents
assert is_cute_dsl_available()
PY
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
      python3 -m pytest -vv -rs tests/utils/test_fp4_quantize.py \
      --junitxml "$RESULT_DIR/${label}_fp4.xml" \
      2>&1 | tee "$RESULT_DIR/${label}_fp4.log"
  )
}

run_full_fp4 baseline "$BASE_TREE"
run_full_fp4 feature "$FEATURE_TREE"
```

对比项：

- 两个 revision 的完整测试均无 failed 或 error。
- Baseline 已有 case 在 feature 中不得出现回归或新增 skip。
- Feature 新增 case 全部通过。
- Feature 包含新增测试时，总 case 数可以高于 baseline。
- JUnit XML 和完整日志保存在 `RESULT_DIR`。

### 5.3 公共 API 用例矩阵

| Shape | 覆盖点 |
| --- | --- |
| `(9, 96)` | 8-row tile 边界；3 个 scale blocks padding 到 4 |
| `(129, 160)` | 128-row tile 边界；5 个 scale blocks padding 到 8 |
| `(2048, 8192)` | 生产规模 smoke test |

```bash
cd "$FEATURE_TREE"

for case_spec in 9:96 129:160 2048:8192; do
  m="${case_spec%%:*}"
  k="${case_spec##*:}"
  for dtype in bfloat16 float16; do
    for layout in 128x4 8x4 linear; do
      python3 benchmarks/flashinfer_benchmark.py \
        --routine mxfp4_quantize \
        --m "$m" \
        --k "$k" \
        --input_dtype "$dtype" \
        --sf_layout "$layout" \
        --backends cuda cute-dsl \
        --refcheck \
        --no_cuda_graph \
        -vv \
        --dry_run_iters 2 \
        --num_iters 5
    done
  done
done
```

预期 shape：

| 输入 | Layout | Quant | Scale |
| --- | --- | --- | --- |
| `(9, 96)` | `128x4` | `(9, 48)` | `(128, 4)` |
| `(9, 96)` | `8x4` | `(9, 48)` | `(16, 4)` |
| `(9, 96)` | `linear` | `(9, 48)` | `(9, 3)` |
| `(129, 160)` | `128x4` | `(129, 80)` | `(256, 8)` |
| `(129, 160)` | `8x4` | `(129, 80)` | `(136, 8)` |
| `(129, 160)` | `linear` | `(129, 80)` | `(129, 5)` |

### 5.4 CUDA Graph 与 PDL

```bash
cd "$FEATURE_TREE"

run_graph_case() {
  layout="$1"
  shift
  python3 benchmarks/flashinfer_benchmark.py \
    --routine mxfp4_quantize \
    --m 2048 \
    --k 8192 \
    --input_dtype bfloat16 \
    --sf_layout "$layout" \
    --backends cuda cute-dsl \
    --refcheck \
    -vv \
    --dry_run_iters 5 \
    --num_iters 30 \
    "$@"
}

for layout in 128x4 8x4 linear; do
  run_graph_case "$layout"
  run_graph_case "$layout" --enable_pdl
done
```

### 5.5 性能

性能对比分为两类：

| 范围 | 对比方式 |
| --- | --- |
| 既有 `128x4` 公共路径 | 使用 target revision 的固定 benchmark runner，对 baseline 和 feature 实现进行 A/B 对比 |
| 新增 `8x4`、`linear` 公共路径 | Feature 记录绝对 latency 和 CUDA/CuTe-DSL 对比；baseline 无同等公共 API，不计算前后误差 |

测试条件：

- 使用同一张 GPU、相同 CUDA/PyTorch 环境，测试期间无其他 GPU workload。
- 使用 CUDA Events，关闭 CUDA Graph；每个 case warmup 20 次、测量 100 次。
- Baseline 和 feature 各执行 5 轮，交替运行顺序。
- 每轮保存 CSV；汇总 median、min、max 和相对变化。

```bash
export PERF_RUNNER="$BASE_TREE/benchmarks/flashinfer_benchmark.py"

run_perf_round() {
  label="$1"
  tree="$2"
  round="$3"
  (
    set -euo pipefail
    python3 -m pip install --no-build-isolation -e "$tree" -v
    cd "$RUN_ROOT"
    for case_spec in 128:4096 2048:8192 8192:16384; do
      m="${case_spec%%:*}"
      k="${case_spec##*:}"
      for dtype in bfloat16 float16; do
        FLASHINFER_DISABLE_VERSION_CHECK=1 \
          python3 "$PERF_RUNNER" \
          --routine mxfp4_quantize \
          --m "$m" \
          --k "$k" \
          --input_dtype "$dtype" \
          --sf_layout 128x4 \
          --backends cuda cute-dsl \
          --refcheck \
          --no_cuda_graph \
          --use_cuda_events \
          --dry_run_iters 20 \
          --num_iters 100 \
          --output_path \
          "$RESULT_DIR/${label}_perf_r${round}_${dtype}_m${m}_k${k}.csv"
      done
    done
  )
}

for round in 1 2 3 4 5; do
  if ((round % 2 == 1)); then
    run_perf_round baseline "$BASE_TREE" "$round"
    run_perf_round feature "$FEATURE_TREE" "$round"
  else
    run_perf_round feature "$FEATURE_TREE" "$round"
    run_perf_round baseline "$BASE_TREE" "$round"
  fi
done
```

相对变化：

```text
delta_pct = (feature_median_ms - baseline_median_ms) / baseline_median_ms * 100
speedup   = baseline_median_ms / feature_median_ms
```

结果表：

| M | K | Dtype | Backend | Baseline median | Feature median | Baseline min/max | Feature min/max | delta_pct | Speedup |
| ---: | ---: | --- | --- | ---: | ---: | --- | --- | ---: | ---: |
|  |  |  |  |  |  |  |  |  |  |

- `delta_pct > 0`：feature 变慢。
- `delta_pct < 0`：feature 变快。
- 同时报告 5 轮结果的 min/max。正向变化超过轮次波动范围时，复测并定位性能回归。

Feature 的三 layout sweep：

```bash
cd "$FEATURE_TREE"
python3 benchmarks/bench_mxfp4_quantize_backend_comparison.py --dtype bfloat16
python3 benchmarks/bench_mxfp4_quantize_backend_comparison.py --dtype float16
```

## 6. 准入条件

1. 记录 `TARGET_SHA`、`FEATURE_SHA` 和 `MERGE_BASE_SHA`；feature 包含当前 target 分支。
2. Baseline 和 feature 均完成各自 revision 的 `tests/utils/test_fp4_quantize.py` 全量测试，无 failed 或 error。
3. Baseline 已有 case 在 feature 中保持通过，且不新增 skip；feature 新增 case 全部通过。
4. Baseline 与 feature 的实际 case 数和 JUnit 结果均已记录。
5. 三种 layout 在 CUDA 和 CuTe-DSL backend 上完成 quantize/dequantize roundtrip。
6. FP16、BF16 及用例矩阵中的 shape 全部通过，输出 shape 符合第 3 节规则。
7. 量化输出无非法值；反量化结果无 NaN/Inf。
8. CUDA/CuTe-DSL quant 和 scale exact-match 比例均大于 95%。
9. Backend comparison 中反量化 cosine similarity 不低于 0.9。
10. Eager、CUDA Graph 和 PDL 路径无异常或输出不一致。
11. Baseline 与 feature 使用固定 benchmark runner 完成同 workload 性能对比，并记录 median、min、max、`delta_pct` 和 speedup；超出轮次波动范围的性能下降已完成复测和分析。
