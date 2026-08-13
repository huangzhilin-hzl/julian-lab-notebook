# SM90 Humming MXFP4A8 MegaMoE H20 测试结果

测试 commit：`3a1ea396f006fe7c406d6610cd9d7f7e936fba6d`

## 精度结果

本次采用 `fast_math=1`。

| 测试范围 | 结果 | 最大 diff |
|---|---:|---:|
| MXFP4 preprocess（14 cases） | 14/14 PASS | - |
| 2 ranks，full | 34/34 PASS | 0.000716 |
| 8 ranks，smoke | 6/6 PASS | 0.000601 |
| Compute Sanitizer，routed Flash M=8192，E=32 | 1 launch PASS，0 errors | - |
| 2 ranks，routed Pro M=256，per-tensor | PASS | 0.000050 |
| 8 ranks，routed Pro M=256，per-tensor | PASS | 0.000050 |
| Compute Sanitizer，routed Pro M=256，per-tensor | 1 launch PASS，0 errors | 0.000049 |
| 8 ranks，routed Flash M=128，per-tensor | PASS | 0.000037 |
| Compute Sanitizer，routed Flash M=128，per-tensor | 1 launch PASS，0 errors | 0.000037 |

所有测试均通过，最大 diff 为 `0.000716`，低于测试阈值 `0.01`。

## 性能结果

测试条件：H20、8 ranks、`fast_math=1`、activation per-tensor、FC1/FC2 dequant scale 均为 `1.0`、`num_max_tokens_per_rank=8192`、5 次 warmup、每轮 20 次 observation、每次 Kineto 20 tests。指标为所有 rank 最慢耗时的中位数 `max_rank_median_us`；MXFP4 采用 3 轮 run-median，且不显式 flush L2。下表为 routed-only（`num_shared_experts=0`）结果。

| 模型 | M | PR #383 FP8 MegaMoE (µs) | MXFP4 per-tensor (µs) | MXFP4 / FP8 |
|---|---:|---:|---:|---:|
| Flash | 64 | 340.7 | 415.595 | 1.220× |
| Flash | 128 | 414.4 | 450.652 | 1.087× |
| Flash | 256 | 569.5 | 553.712 | 0.972× |
| Flash | 8192 | 9749.0 | 10417.000 | 1.069× |
| Pro | 64 | 1059.9 | 1178.000 | 1.111× |
| Pro | 128 | 1201.0 | 1218.000 | 1.014× |
| Pro | 256 | 1639.9 | 1314.000 | 0.801× |
| Pro | 8192 | 24777.0 | 26839.000 | 1.083× |

FP8 数据来自 [DeepGEMM PR #383](https://github.com/deepseek-ai/DeepGEMM/pull/383)。`MXFP4 / FP8` 仅为两组耗时的数值比值；两者量化格式和测试代码不同，不作为严格 A/B 加速比。

### Flash swap-AB descriptor 增量更新

commit `f45c6003c0fb43e2e909d8062b644c6b2d97760c` 仅在 Flash small-M swap-AB 路径复用 WGMMA base descriptor，并按 K32 增量更新起始地址。Pro 和非 swap-AB 路径保持原机器码。

| 模型 | M | baseline A1（µs） | candidate B（µs） | baseline A2（µs） | baseline 中心（µs） | B 相对中心 |
|---|---:|---:|---:|---:|---:|---:|
| Flash | 64 | 436.257 | 432.299 | 452.190 | 444.224 | -2.68% |
| Flash | 128 | 451.124 | 448.085 | 442.945 | 447.035 | +0.23% |

收窄为 Flash-only 后的独立复测为 M64 `425.092 µs`、M128 `439.814 µs`。M64 在正式 A/B/A 中超过 `0.5%` 接受阈值；M128 正式 A/B/A 差异在噪声范围内，独立复测未观察到回退。候选资源为 Flash `REG=128, STACK=0`，且 Compute Sanitizer memcheck 报告 `0 errors`。

### Flash M128 合并 swap-AB WGMMA group

commit `dd572d45f27e8f62c8cad66cb9cc680451708b10` 在 Flash M128 small-M swap-AB 路径中，将两个 N64 weight half 的 WGMMA 合并到同一个 group，减少每个 K stage 的 `wgmma.fence` 和 `wgmma.commit_group`。该优化通过 JIT 编译单元宏隔离；未启用的 Flash M64、Pro M256 等路径与 `f45c600` 的同源基线 SASS 完全一致。

| 模型 | M | baseline A1（µs） | candidate B（µs） | baseline A2（µs） | baseline 中心（µs） | B 相对中心 |
|---|---:|---:|---:|---:|---:|---:|
| Flash | 128 | 482.623 | 447.647 | 548.622 | 515.623 | -13.18% |

最终作用域下的独立复测为 `463.298 µs`，相对 baseline 中心改善 `10.15%`。候选资源为 `REG=128, STACK=0`；MXFP4 preprocess `14/14`、2 ranks full `34/34`、8 ranks smoke `6/6`、8 ranks Flash M128 精确形状均通过，Compute Sanitizer memcheck 报告 `0 errors`。

### Flash M128 swap-AB L2 warp-scatter

commit `3a1ea396f006fe7c406d6610cd9d7f7e936fba6d` 仅在 routed Flash M128 small-M swap-AB 编译单元启用新的 L2 epilogue。每个 warp 先把自己负责的 32 列从寄存器转置到 SMEM，再按连续 16-byte span 直接写入远端 combine buffer，去掉散射前的 128-thread barrier，并把每行元数据读取从 16 次降到 4 次。任务末尾的 128-thread barrier 保留，确保下一任务复用 SMEM 前所有 scatter 已完成。

| 模型 | M | baseline A1（µs） | candidate B（µs） | baseline A2（µs） | baseline 中心（µs） | B 相对中心 |
|---|---:|---:|---:|---:|---:|---:|
| Flash | 128 | 484.336 | 418.747 | 491.434 | 487.885 | -14.17% |

独立候选复测为 `438.055 µs`；两次候选中位数的中心为 `428.401 µs`，相对 baseline 中心改善 `12.19%`。最终 16-case 矩阵中的 Flash M128 为 `450.652 µs`，相对上一接受版本 `dd572d4` 的正式矩阵 `487.612 µs` 改善 `7.58%`。候选资源由 `REG=128` 降至 `REG=125`，且 `STACK=0, LOCAL=0`；MXFP4 preprocess `14/14`、2 ranks full `34/34`、8 ranks smoke `6/6`、8 ranks Flash M128 精确形状均通过，Compute Sanitizer memcheck 报告 `0 errors`。

### 共享专家性能

测试条件与上表相同，`num_shared_experts=1`。采用 per-tensor activation scale 和 3 轮 run-median。

| 模型 | M | MXFP4 per-tensor (µs) |
|---|---:|---:|
| Flash | 64 | 553.808 |
| Flash | 128 | 572.975 |
| Flash | 256 | 611.447 |
| Flash | 8192 | 13288.000 |
| Pro | 64 | 1877.000 |
| Pro | 128 | 1907.000 |
| Pro | 256 | 1955.000 |
| Pro | 8192 | 33685.000 |

shared-expert 编译单元不会启用本次 routed-only 宏，因此与上一接受版本之间的差异按测试噪声处理，不归因为本次优化。
