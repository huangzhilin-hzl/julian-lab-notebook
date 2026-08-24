# DeepGEMM PR #411 FP8×MXFP4 更新前后对比（H20，fast-math=1）

## 结论

本次在 8× NVIDIA H20 上对 DeepGEMM PR #411 的性能优化提交执行同环境 A/B 测试。更新前后分别使用独立源码目录与 JIT cache；Flash、Pro 各测试 11 个 M，共 22 个 case，全部运行成功。

- 全部 22 个 shape：rank0 几何平均提升 **1.04%**，max-rank 几何平均提升 **1.68%**。
- 9 个主要优化目标（Flash `M>=1024`、Pro `M>=512`）：rank0 几何平均提升 **1.60%**，max-rank 几何平均提升 **1.67%**。
- Flash `M>=1024` 的 rank0 提升 **1.7%～2.6%**；Pro `M>=512` 提升 **0.9%～1.8%**。
- Flash M32/M128/M256 和 Pro M128/M256 出现 **0.4%～1.7%** 的轻微 rank0 回退。

## 测试对象

| 项目 | 内容 |
|---|---|
| PR | [deepseek-ai/DeepGEMM#411](https://github.com/deepseek-ai/DeepGEMM/pull/411) |
| 更新前 | [`f026a32667a88aacef2ce4982c15ec17fec34b19`](https://github.com/deepseek-ai/DeepGEMM/commit/f026a32667a88aacef2ce4982c15ec17fec34b19) |
| 更新后 | [`5d32216686b982a39eabccca9419af430a60cfc2`](https://github.com/deepseek-ai/DeepGEMM/commit/5d32216686b982a39eabccca9419af430a60cfc2) |
| Benchmark | [`tests/bench_mega_moe_sm90.py`](https://github.com/deepseek-ai/DeepGEMM/blob/5d32216686b982a39eabccca9419af430a60cfc2/tests/bench_mega_moe_sm90.py) |
| GPU | NVIDIA H20 × 8 |
| 软件栈 | Python 3.12.3，PyTorch 2.11.0+cu130，CUDA Toolkit 13.0，NCCL 2.28.9，NVIDIA Driver 550.127.08 |
| 日期 | 2026-08-24 |

`f026a32` 是 `5d32216` 的直接父提交，已经包含 FP8×MXFP4 功能，但尚未包含本次性能优化，因此用于更新前基线。

## 测试配置

```bash
python3 tests/bench_mega_moe_sm90.py \
  --fast-math 1 \
  --flush-l2 1 \
  --report-rank-times \
  --report-route-stats
```

| 参数 | 值 |
|---|---|
| ranks | 8 |
| workload | `flash`、`pro` |
| M | 8、16、32、64、128、256、512、1024、2048、4096、8192 |
| num-max-tokens-per-rank | 8192 |
| shared experts | 0 |
| fast-math | 1 |
| flush-L2 | 1（cold L2） |
| seed | 0 |
| activation clamp | 10.0 |
| masked ratio | 0.0 |
| warmups | 1 |
| tests | 每个 observation 20 次 kernel launch |
| observations | `M<=128`: 50；`M>=256`: 3 |
| 计时范围 | `sm90_fp8_mxfp4_mega_moe_persistent_impl` kernel only |

指标定义：

- `rank0 median`：每个 observation 通过 Kineto 取得 rank 0 persistent kernel 时间，再对 observations 取中位数。
- `max-rank median`：每个 observation 先取 8 个 rank 中的最大 kernel 时间，再对 observations 取中位数。
- `提升`：`更新前 / 更新后 - 1`，正值表示更新后更快。
- 权重预处理和 input-buffer copy 不计入 kernel 时间。

## Flash workload

| M | 更新前 rank0 (µs) | 更新后 rank0 (µs) | rank0 提升 | 更新前 max-rank (µs) | 更新后 max-rank (µs) | max-rank 提升 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 276.4 | 272.2 | +1.6% | 296.9 | 286.5 | +3.6% |
| 16 | 323.0 | 311.1 | +3.8% | 336.4 | 318.1 | +5.8% |
| 32 | 317.7 | 323.1 | -1.7% | 329.2 | 331.9 | -0.8% |
| 64 | 343.7 | 333.9 | +2.9% | 358.4 | 349.6 | +2.5% |
| 128 | 404.5 | 408.5 | -1.0% | 411.8 | 417.4 | -1.3% |
| 256 | 467.9 | 471.7 | -0.8% | 504.6 | 476.1 | +6.0% |
| 512 | 884.8 | 881.3 | +0.4% | 892.8 | 902.6 | -1.1% |
| 1024 | 1487.0 | 1449.0 | +2.6% | 1497.0 | 1453.0 | +3.0% |
| 2048 | 2714.0 | 2665.0 | +1.8% | 2726.0 | 2691.0 | +1.3% |
| 4096 | 5106.0 | 5005.0 | +2.0% | 5121.0 | 5017.0 | +2.1% |
| 8192 | 9907.0 | 9741.0 | +1.7% | 9917.0 | 9759.0 | +1.6% |

## Pro workload

| M | 更新前 rank0 (µs) | 更新后 rank0 (µs) | rank0 提升 | 更新前 max-rank (µs) | 更新后 max-rank (µs) | max-rank 提升 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 707.0 | 700.2 | +1.0% | 728.1 | 711.2 | +2.4% |
| 16 | 934.9 | 913.4 | +2.3% | 949.3 | 924.9 | +2.6% |
| 32 | 988.2 | 985.2 | +0.3% | 1009.5 | 998.1 | +1.1% |
| 64 | 1023.0 | 1013.0 | +1.0% | 1033.5 | 1025.0 | +0.8% |
| 128 | 1169.0 | 1178.5 | -0.8% | 1186.5 | 1190.0 | -0.3% |
| 256 | 1587.0 | 1593.0 | -0.4% | 1623.0 | 1608.0 | +0.9% |
| 512 | 2500.0 | 2474.0 | +1.1% | 2532.0 | 2485.0 | +1.9% |
| 1024 | 3895.0 | 3852.0 | +1.1% | 3904.0 | 3864.0 | +1.0% |
| 2048 | 6924.0 | 6865.0 | +0.9% | 6932.0 | 6881.0 | +0.7% |
| 4096 | 13065.0 | 12832.0 | +1.8% | 13080.0 | 12842.0 | +1.9% |
| 8192 | 25261.0 | 24912.0 | +1.4% | 25289.0 | 24912.0 | +1.5% |

## 有效性与脱敏说明

- 更新前后使用相同硬件和软件环境，使用独立源码目录及独立 JIT cache；route seed 和 route statistics 一致。
- 两轮各产生 22 个 benchmark summary，没有 traceback、runtime error、timeout 或 non-finite output。
- benchmark 执行 finite-output 检查，不替代完整数值正确性测试。
- 小 M 的 max-rank observation 存在偶发离群点，因此表中报告 50 个 observation 的中位数。
- 仅发布聚合 benchmark 数据和必要配置；不发布节点、Pod、网络地址、私有镜像、本地路径、原始日志或环境快照。

机器可读数据：[`results/20260824-h20-deepgemm-pr411-fastmath1/comparison.csv`](results/20260824-h20-deepgemm-pr411-fastmath1/comparison.csv)
