# PR383 FP8 MegaMoE vs FlashInfer PR4113 SM90 MegaMoE：H20/H20-3e 对比

## 结论

- 本文按 [DeepGEMM PR #383](https://github.com/deepseek-ai/DeepGEMM/pull/383) 的 H20 HT/LL 表格展示：保留官方 `MegaMoE / DeepEP / Speedup` 三列，并追加 FlashInfer PR4113 实测及相对倍率。
- FlashInfer PR4113 在 8× NVIDIA H20-3e 上完成 22/22 case、66/66 observation，失败数为 0。
- HT 中，FlashInfer 延迟是 PR383 MegaMoE 的 1.067–1.791×、PR383 DeepEP HT 的 0.794–1.322×；相对 DeepEP HT 的结果随模型和 M 混合分布。
- LL 中，FlashInfer 延迟是 PR383 MegaMoE 的 1.265–1.791×、PR383 DeepEP LL 的 0.846–1.020×；Flash 与 DeepEP LL 基本接近，Pro 的 5 个点数值均更低。
- 上述倍率是跨机器、跨 commit、跨计时边界的描述性对照，不是严格 A/B：PR383 表标注为 H20，本次 FlashInfer 复测是 H20-3e。

## Workload 与测试口径

| Shape | Hidden | Intermediate | Experts | Top-k | EP |
|:---|---:|---:|---:|---:|---:|
| Flash | 4096 | 2048 | 256 | 6 | 8 |
| Pro | 7168 | 3072 | 384 | 6 | 8 |

| 项目 | HT | LL |
|:---|:---|:---|
| PR383 DeepEP | grouped-contiguous FP8 dispatch/combine | low-latency grouped-masked FP8 dispatch/combine |
| FlashInfer | 同一个 `sm90_pull_fp8` MegaKernel compute 路径 | 同一个 `sm90_pull_fp8` MegaKernel compute 路径 |
| M | 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 | 8, 16, 32, 64, 128 |
| Capacity | `cap=M` | `cap=M` |
| FlashInfer 计时内 | `backend.compute(output=None)`：fused dispatch、FC1、SwiGLU、FC2、默认 TopkReduce | 与 HT 表相同；使用相同小 M 实测值 |
| FlashInfer 计时外 | 权重准备、backend plan、JIT warmup | 权重准备、backend plan、JIT warmup |

每个 case 固定 `seed=101`、3 个 observation；每个 observation 先 warmup 5 次，再采集 20 个 sample。每个 sample 前冲刷 8,000,000,000 bytes L2 buffer。单个 observation 取各 rank CUDA Event 中位数的最大值，表中 FlashInfer 数值再取 3 个 observation 的中位数。

PR383 官方 `Speedup` 按 `DeepEP / MegaMoE - 1` 计算。本文新增的两列是延迟倍率：

- `FlashInfer / MegaMoE`：FlashInfer 实测除以 PR383 MegaMoE；小于 1 表示 FlashInfer 数值更低。
- `FlashInfer / PR383 DeepEP`：FlashInfer 实测除以 PR383 DeepEP；小于 1 表示 FlashInfer 数值更低。

FlashInfer PR4113 这里只有一个 `sm90_pull_fp8` compute backend。将其 `M=8–128` 数据同时放进 HT、LL 对照表，是为了对齐 PR383 的表格分组，不代表 FlashInfer 存在独立的 HT/LL 两种执行模式。

## 测试环境

| 项目 | 值 |
|:---|:---|
| 测试拓扑 | 单机 NVIDIA H20-3e ×8，EP8 |
| FlashInfer PR4113 head | `28483960d7a56dd6a77e735f2c874b8e4dbd9d44` |
| Backend | `sm90_pull_fp8`、blockwise FP8 scale、`swap_ab`、tile `(256, 32, 128)`、`atomic_counter` |
| 其他配置 | `token_back`、复用 dispatch warps、`fast_math=true` |
| Benchmark SHA256 | `94cc0faf21d66cacb3eb1442a533924a9fa767437a785de58a94abfbefd8cb9e` |
| 完成日期 | 2026-08-11 |

## High Throughput：PR383 H20 vs FlashInfer PR4113 H20-3e

| Model | M | PR383 MegaMoE FP8 (us) | PR383 DeepEP HT (us) | PR383 Speedup | FlashInfer PR4113 (us) | FlashInfer / MegaMoE | FlashInfer / PR383 DeepEP |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Flash | 8 | 273.1 | 566.7 | +107.5% | 489.136 | 1.791× | 0.863× |
| Flash | 16 | 304.4 | 619.2 | +103.4% | 515.936 | 1.695× | 0.833× |
| Flash | 32 | 302.0 | 618.6 | +104.8% | 531.456 | 1.760× | 0.859× |
| Flash | 64 | 340.7 | 622.6 | +82.7% | 518.160 | 1.521× | 0.832× |
| Flash | 128 | 414.4 | 638.5 | +54.1% | 553.616 | 1.336× | 0.867× |
| Flash | 256 | 569.5 | 688.7 | +20.9% | 910.480 | 1.599× | 1.322× |
| Flash | 512 | 922.0 | 1057.3 | +14.7% | 1287.776 | 1.397× | 1.218× |
| Flash | 1024 | 1516.6 | 1983.4 | +30.8% | 2513.120 | 1.657× | 1.267× |
| Flash | 2048 | 2735.1 | 3419.5 | +25.0% | 2952.704 | 1.080× | 0.863× |
| Flash | 4096 | 5116.0 | 6087.7 | +19.0% | 5585.488 | 1.092× | 0.918× |
| Flash | 8192 | 9749.0 | 11779.2 | +20.8% | 11008.384 | 1.129× | 0.935× |
| Pro | 8 | 768.0 | 1349.2 | +75.7% | 1070.896 | 1.394× | 0.794× |
| Pro | 16 | 950.3 | 1585.8 | +66.9% | 1327.184 | 1.397× | 0.837× |
| Pro | 32 | 1026.3 | 1770.2 | +72.5% | 1495.648 | 1.457× | 0.845× |
| Pro | 64 | 1059.9 | 1787.6 | +68.7% | 1501.136 | 1.416× | 0.840× |
| Pro | 128 | 1201.0 | 1803.9 | +50.2% | 1519.344 | 1.265× | 0.842× |
| Pro | 256 | 1639.9 | 1857.7 | +13.3% | 2106.096 | 1.284× | 1.134× |
| Pro | 512 | 2599.0 | 2898.8 | +11.5% | 2863.632 | 1.102× | 0.988× |
| Pro | 1024 | 4036.0 | 5412.7 | +34.1% | 5440.016 | 1.348× | 1.005× |
| Pro | 2048 | 6986.0 | 8067.9 | +15.5% | 7451.168 | 1.067× | 0.924× |
| Pro | 4096 | 12932.0 | 14614.0 | +13.0% | 13886.656 | 1.074× | 0.950× |
| Pro | 8192 | 24777.0 | 28184.2 | +13.8% | 27537.552 | 1.111× | 0.977× |

HT 的数值分布：

- Flash 相对 DeepEP HT：`M≤128` 为 0.832–0.867×，`M=256/512/1024` 为 1.218–1.322×，`M≥2048` 为 0.863–0.935×。
- Pro 相对 DeepEP HT：除 `M=256`（1.134×）和 `M=1024`（1.005×）外均低于 1；`M=512` 为 0.988×，数值接近。

## Low Latency：PR383 H20 vs FlashInfer PR4113 H20-3e

| Model | M | PR383 MegaMoE FP8 (us) | PR383 DeepEP LL (us) | PR383 Speedup | FlashInfer PR4113 (us) | FlashInfer / MegaMoE | FlashInfer / PR383 DeepEP |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Flash | 8 | 273.1 | 479.7 | +75.7% | 489.136 | 1.791× | 1.020× |
| Flash | 16 | 304.4 | 528.4 | +73.6% | 515.936 | 1.695× | 0.976× |
| Flash | 32 | 302.0 | 531.8 | +76.1% | 531.456 | 1.760× | 0.999× |
| Flash | 64 | 340.7 | 552.2 | +62.1% | 518.160 | 1.521× | 0.938× |
| Flash | 128 | 414.4 | 552.8 | +33.4% | 553.616 | 1.336× | 1.001× |
| Pro | 8 | 768.0 | 1265.6 | +64.8% | 1070.896 | 1.394× | 0.846× |
| Pro | 16 | 950.3 | 1493.3 | +57.1% | 1327.184 | 1.397× | 0.889× |
| Pro | 32 | 1026.3 | 1667.2 | +62.4% | 1495.648 | 1.457× | 0.897× |
| Pro | 64 | 1059.9 | 1683.8 | +58.9% | 1501.136 | 1.416× | 0.892× |
| Pro | 128 | 1201.0 | 1714.1 | +42.7% | 1519.344 | 1.265× | 0.886× |

LL 的数值分布：Flash 相对 DeepEP LL 为 0.938–1.020×，Pro 为 0.846–0.897×。这仍是跨环境的数值对照，不能据此直接宣称同机端到端加速。

## 结果边界

- PR383 数值来自 PR 作者发布的 H20 表；本文没有在同一台 H20-3e 上重新运行 PR383 FP8 MegaMoE/DeepEP baseline，因此相对倍率不能作为同机严格加速比。
- PR383 MegaMoE 使用 Kineto 汇总两个指定 DeepGEMM kernel；FlashInfer 使用 CUDA Event 包围 `MegaKernelBackend.compute(output=None)`，其中包含 fused dispatch、FC1、SwiGLU、FC2 和默认 TopkReduce。workload 对齐不等于计时范围完全相同。
- FlashInfer 的 HT/LL 表使用同一个 compute 路径和同一组小 M 数据；表格分组只用于匹配 PR383 的展示方式。
- benchmark 验证执行成功和结构约束，但没有逐元素 PyTorch cross-backend oracle，因此本数据不是完整数值精度证明。
- 这是单机 EP8、H20-3e、指定 FlashInfer PR4113 commit 的结果，不能直接外推到其他 GPU、多机或 serving 并发场景。

## 原始证据

- PR383 官方 H20 数据：[DeepGEMM PR #383 Benchmark Results](https://github.com/deepseek-ai/DeepGEMM/pull/383)
- 测试入口：[bench_flashinfer_pr4113_megamoe_sm90.py](./bench_flashinfer_pr4113_megamoe_sm90.py)
- 本文保留聚合后的性能数据、公开 commit、benchmark 哈希和复现参数；包含 Pod、节点、内网地址、进程清单及环境快照的原始日志不随公开报告发布。
