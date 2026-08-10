# SM90 Humming MXFP4A8 MegaMoE 测试

- 远程分支：[huangzhilin-hzl/DeepGEMM:molou/support_sm90_humming_mxfp4afp8_megamoe](https://github.com/huangzhilin-hzl/DeepGEMM/tree/molou/support_sm90_humming_mxfp4afp8_megamoe)
- 硬件：SM90（H20/H100/H200）

## 测试契约

| 测试 | 入口 | 口径 |
|---|---|---|
| 精度 | `tests/test_mega_moe_sm90.py` | PyTorch oracle、跨 rank 路由和计数；`diff < 0.01` |
| 预处理 | `tests/test_mega_moe_mxfp4_preprocess.py` | packed E2M1、K32 UE8M0、processed triplet/API |
| 性能 | `tests/bench_mega_moe_sm90.py` | persistent kernel-only；主指标 `max_rank_median_us` |
| Profile | benchmark `--profile-only` | 单 case、单次 launch，不作为稳态延迟 |

性能执行入口始终使用当前分支文件；数据记录格式参考 [DeepGEMM PR #383](https://github.com/deepseek-ai/DeepGEMM/pull/383)，保留逐轮 observation、rank-0/max-rank、median/range 和 JSON summary。当前脚本只有 MXFP4，不据此声明相对 FP8 speedup。

## 准备

```bash
set -o pipefail
DG_REPO=~/security_inference/DeepGEMM
cd "$DG_REPO"

SM90_RUN_ID="$(date +%Y%m%d-%H%M%S)"
SM90_RESULT_ROOT="$DG_REPO/results/sm90-humming-mxfp4afp8-megamoe/$SM90_RUN_ID"
mkdir -p "$SM90_RESULT_ROOT"/{environment,correctness,benchmark,nsys,ncu,jit-cache}
export DG_JIT_CACHE_DIR="$SM90_RESULT_ROOT/jit-cache"

git branch --show-current > "$SM90_RESULT_ROOT/environment/branch.txt"
git rev-parse HEAD > "$SM90_RESULT_ROOT/environment/head.txt"
git status --short > "$SM90_RESULT_ROOT/environment/git-status.txt"
git diff HEAD > "$SM90_RESULT_ROOT/environment/working-tree.diff"
nvidia-smi > "$SM90_RESULT_ROOT/environment/nvidia-smi.txt"
```

CUDA header、heuristic 或 launch 配置变更后重编译：

```bash
MAX_JOBS=8 python3 setup.py build_ext --inplace --force \
  2>&1 | tee "$SM90_RESULT_ROOT/environment/build.log"
```

## 精度

```bash
pytest -q tests/test_mega_moe_mxfp4_preprocess.py \
  2>&1 | tee "$SM90_RESULT_ROOT/correctness/preprocess.log"

for FM in 0 1; do
  CUDA_VISIBLE_DEVICES=0,1 \
  python3 tests/test_mega_moe_sm90.py \
    --num-processes 2 --suite full --fast-math "$FM" \
    --diff-tolerance 0.01 --fail-fast \
    2>&1 | tee "$SM90_RESULT_ROOT/correctness/full-2rank-fastmath${FM}.log"

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python3 tests/test_mega_moe_sm90.py \
    --num-processes 8 --suite smoke --fast-math "$FM" \
    --diff-tolerance 0.01 --fail-fast \
    2>&1 | tee "$SM90_RESULT_ROOT/correctness/smoke-8rank-fastmath${FM}.log"
done

CUDA_VISIBLE_DEVICES=0 \
compute-sanitizer --tool memcheck --error-exitcode 1 \
  python3 tests/test_mega_moe_sm90.py \
    --num-processes 1 --no-dist --suite smoke --fast-math 0 --fail-fast \
  2>&1 | tee "$SM90_RESULT_ROOT/correctness/memcheck.log"
```

通过条件：非零 `planned`、`success == planned`、`failed == 0`；Sanitizer 还需 `ERROR SUMMARY: 0 errors`。

## 性能与数据记录

同时采集 strict/fast math，以及显式冷 L2（`1`）和不显式冲刷 L2（`0`）：

```bash
for FM in 0 1; do
  for FLUSH_L2 in 0 1; do
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python3 tests/bench_mega_moe_sm90.py \
      --num-processes 8 --implementation mxfp4 \
      --model-config flash pro --batches 64 128 256 \
      --num-max-tokens-per-rank 256 \
      --num-warmups 5 --repeats 20 --num-tests 20 \
      --fast-math "$FM" --flush-l2 "$FLUSH_L2" --seed 0 \
      2>&1 | tee \
        "$SM90_RESULT_ROOT/benchmark/8rank-fastmath${FM}-flushl2${FLUSH_L2}.log"
  done
done

rg --no-filename '^BENCH_PLAN_JSON ' "$SM90_RESULT_ROOT/benchmark"/*.log \
  | sed 's/^BENCH_PLAN_JSON //' > "$SM90_RESULT_ROOT/benchmark/plans.jsonl"
rg --no-filename '^BENCH_OBS_JSON ' "$SM90_RESULT_ROOT/benchmark"/*.log \
  | sed 's/^BENCH_OBS_JSON //' > "$SM90_RESULT_ROOT/benchmark/observations.jsonl"
rg --no-filename '^BENCH_SUMMARY_JSON ' "$SM90_RESULT_ROOT/benchmark"/*.log \
  | sed 's/^BENCH_SUMMARY_JSON //' > "$SM90_RESULT_ROOT/benchmark/summaries.jsonl"
jq -s '.' "$SM90_RESULT_ROOT/benchmark/summaries.jsonl" \
  > "$SM90_RESULT_ROOT/benchmark/summaries.json"
```

报告 `max_rank_median_us` 和 `[max_rank_min_us, max_rank_max_us]`；保留原始日志、JSONL/JSON、HEAD、dirty diff 和 GPU 信息。该指标不含权重预处理和 input-buffer copy。

## NSYS / NCU

固定单卡 Flash `M=128, H=4096, I=2048, E=32, topk=6`：

```bash
DG_PROFILE_ARGS=(
  tests/bench_mega_moe_sm90.py
  --num-processes 1 --no-dist --profile-only
  --model-config flash --batches 128 --num-max-tokens-per-rank 128
  --num-experts-override 32 --fast-math 1
)

CUDA_VISIBLE_DEVICES=0 python3 "${DG_PROFILE_ARGS[@]}" \
  2>&1 | tee "$SM90_RESULT_ROOT/nsys/warmup.log"

CUDA_VISIBLE_DEVICES=0 \
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --force-overwrite=true \
  --output="$SM90_RESULT_ROOT/nsys/mxfp4-flash-m128-e32" \
  python3 "${DG_PROFILE_ARGS[@]}" \
  2>&1 | tee "$SM90_RESULT_ROOT/nsys/profile.log"

nsys stats --report cuda_gpu_kern_sum,cuda_gpu_trace --format csv \
  --output "$SM90_RESULT_ROOT/nsys/mxfp4-flash-m128-e32" \
  "$SM90_RESULT_ROOT/nsys/mxfp4-flash-m128-e32.nsys-rep" \
  2>&1 | tee "$SM90_RESULT_ROOT/nsys/stats.log"

CUDA_VISIBLE_DEVICES=0 \
ncu --target-processes all --set full --kernel-name-base function \
  --kernel-name 'regex:sm90_fp8_mxfp4_mega_moe_persistent_impl.*' \
  --launch-count 1 --force-overwrite \
  --export "$SM90_RESULT_ROOT/ncu/mxfp4-flash-m128-e32" \
  python3 "${DG_PROFILE_ARGS[@]}" \
  2>&1 | tee "$SM90_RESULT_ROOT/ncu/profile.log"

ncu --import "$SM90_RESULT_ROOT/ncu/mxfp4-flash-m128-e32.ncu-rep" \
  --page raw --csv > "$SM90_RESULT_ROOT/ncu/full-details.csv"
ncu --import "$SM90_RESULT_ROOT/ncu/mxfp4-flash-m128-e32.ncu-rep" \
  --page source --csv > "$SM90_RESULT_ROOT/ncu/source-sass.csv"
```
