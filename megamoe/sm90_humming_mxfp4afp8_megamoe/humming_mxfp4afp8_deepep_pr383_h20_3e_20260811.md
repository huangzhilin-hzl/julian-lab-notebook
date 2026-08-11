# PR383 FP8 MegaMoE vs Humming MXFP4A-FP8 + DeepEP：H20/H20-3e 对比

## 结论

- 本文按 [DeepGEMM PR #383](https://github.com/deepseek-ai/DeepGEMM/pull/383) 的 H20 HT/LL 表格展示：保留官方 `MegaMoE / DeepEP / Speedup` 三列，并追加 Humming 实测及相对倍率。
- Humming 在 8× NVIDIA H20-3e 上完成 32/32 case、96/96 observation，失败数为 0。
- HT 中，Humming 延迟是 PR383 MegaMoE 的 1.281–4.308×、PR383 DeepEP HT 的 1.097–2.403×，所有点都更慢。
- LL 中，Humming 延迟是 PR383 DeepEP LL 的 0.520–0.828×，10 个点都更低；相对 PR383 MegaMoE，Flash 为 1.105–1.215×，Pro 为 0.817–0.942×。
- 上述倍率是跨机器、跨 commit 的描述性对照，不是严格 A/B：PR383 表标注为 H20，本次 Humming 复测是 H20-3e。

## Workload 与测试口径

| Shape | Hidden | Intermediate | Experts | Top-k | EP |
|:---|---:|---:|---:|---:|---:|
| Flash | 4096 | 2048 | 256 | 6 | 8 |
| Pro | 7168 | 3072 | 384 | 6 | 8 |

| 项目 | HT | LL |
|:---|:---|:---|
| DeepEP | grouped-contiguous FP8 dispatch/combine | low-latency grouped-masked FP8 dispatch/combine |
| Humming | MXFP4A-FP8 grouped-contiguous L1/L2 | MXFP4A-FP8 grouped-masked L1/L2 |
| M | 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 | 8, 16, 32, 64, 128 |
| Capacity | `cap=M` | `cap=M` |
| 计时内 | dispatch、L1、SwiGLU/路由权重/FP8 重量化、L2、combine | dispatch、L1、masked SwiGLU/FP8 重量化、L2、combine/top-k weighting |
| 计时外 | 权重量化/transform、JIT warmup | 权重量化/transform、JIT warmup |

每个 case 固定 `seed=101`、3 个 observation；每个 observation 先 warmup 5 次，再采集 20 个 sample。每个 sample 前实际冲刷至少 8,000,000,000 bytes L2 buffer。

PR383 官方 `Speedup` 按 `DeepEP / MegaMoE - 1` 计算。本文新增的两列是延迟倍率：

- `Humming / MegaMoE`：Humming 实测除以 PR383 MegaMoE；小于 1 表示 Humming 数值更低。
- `Humming / PR383 DeepEP`：Humming 实测除以 PR383 DeepEP；小于 1 表示 Humming 数值更低。

## 测试环境

| 项目 | 值 |
|:---|:---|
| Pod | `molou/molou-deepep-humming-mxfp4-2089-0811` |
| Node | `lj-21d432895` / `10.13.2.89` |
| GPU | NVIDIA H20-3e ×8，单机 NV18 全互联 |
| DeepEP commit | `60d44037a702f651a6e18bd4aea65ed8409051c2` |
| Humming commit | `39a66bb86804f40cd45ce55f4df8657ce14ba7e9` |
| HT 环境 | `EP_DISABLE_GIN=1`、`NCCL_NVLS_ENABLE=0` |
| LL 环境 | NVSHMEM/RDMA、`NVSHMEM_QP_DEPTH=1024` |
| 完成时间 | 2026-08-11 16:45:29 +08:00 |

## High Throughput：PR383 H20 vs Humming H20-3e

| Model | M | PR383 MegaMoE FP8 (us) | PR383 DeepEP HT (us) | PR383 Speedup | Humming MXFP4A-FP8 + DeepEP (us) | Humming / MegaMoE | Humming / PR383 DeepEP |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Flash | 8 | 273.1 | 566.7 | +107.5% | 1162.240 | 4.256× | 2.051× |
| Flash | 16 | 304.4 | 619.2 | +103.4% | 1288.544 | 4.233× | 2.081× |
| Flash | 32 | 302.0 | 618.6 | +104.8% | 1301.072 | 4.308× | 2.103× |
| Flash | 64 | 340.7 | 622.6 | +82.7% | 1300.912 | 3.818× | 2.089× |
| Flash | 128 | 414.4 | 638.5 | +54.1% | 1304.224 | 3.147× | 2.043× |
| Flash | 256 | 569.5 | 688.7 | +20.9% | 1334.656 | 2.344× | 1.938× |
| Flash | 512 | 922.0 | 1057.3 | +14.7% | 1523.872 | 1.653× | 1.441× |
| Flash | 1024 | 1516.6 | 1983.4 | +30.8% | 2212.624 | 1.459× | 1.116× |
| Flash | 2048 | 2735.1 | 3419.5 | +25.0% | 3752.176 | 1.372× | 1.097× |
| Flash | 4096 | 5116.0 | 6087.7 | +19.0% | 6797.312 | 1.329× | 1.117× |
| Flash | 8192 | 9749.0 | 11779.2 | +20.8% | 13033.184 | 1.337× | 1.106× |
| Pro | 8 | 768.0 | 1349.2 | +75.7% | 3161.520 | 4.117× | 2.343× |
| Pro | 16 | 950.3 | 1585.8 | +66.9% | 3810.576 | 4.010× | 2.403× |
| Pro | 32 | 1026.3 | 1770.2 | +72.5% | 4131.472 | 4.026× | 2.334× |
| Pro | 64 | 1059.9 | 1787.6 | +68.7% | 3579.376 | 3.377× | 2.002× |
| Pro | 128 | 1201.0 | 1803.9 | +50.2% | 3594.448 | 2.993× | 1.993× |
| Pro | 256 | 1639.9 | 1857.7 | +13.3% | 3647.824 | 2.224× | 1.964× |
| Pro | 512 | 2599.0 | 2898.8 | +11.5% | 3719.824 | 1.431× | 1.283× |
| Pro | 1024 | 4036.0 | 5412.7 | +34.1% | 5985.664 | 1.483× | 1.106× |
| Pro | 2048 | 6986.0 | 8067.9 | +15.5% | 9276.496 | 1.328× | 1.150× |
| Pro | 4096 | 12932.0 | 14614.0 | +13.0% | 16940.944 | 1.310× | 1.159× |
| Pro | 8192 | 24777.0 | 28184.2 | +13.8% | 31739.088 | 1.281× | 1.126× |

## Low Latency：PR383 H20 vs Humming H20-3e

| Model | M | PR383 MegaMoE FP8 (us) | PR383 DeepEP LL (us) | PR383 Speedup | Humming MXFP4A-FP8 + DeepEP (us) | Humming / MegaMoE | Humming / PR383 DeepEP |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Flash | 8 | 273.1 | 479.7 | +75.7% | 320.160 | 1.172× | 0.667× |
| Flash | 16 | 304.4 | 528.4 | +73.6% | 364.320 | 1.197× | 0.689× |
| Flash | 32 | 302.0 | 531.8 | +76.1% | 367.008 | 1.215× | 0.690× |
| Flash | 64 | 340.7 | 552.2 | +62.1% | 399.520 | 1.173× | 0.724× |
| Flash | 128 | 414.4 | 552.8 | +33.4% | 457.920 | 1.105× | 0.828× |
| Pro | 8 | 768.0 | 1265.6 | +64.8% | 688.624 | 0.897× | 0.544× |
| Pro | 16 | 950.3 | 1493.3 | +57.1% | 775.936 | 0.817× | 0.520× |
| Pro | 32 | 1026.3 | 1667.2 | +62.4% | 892.464 | 0.870× | 0.535× |
| Pro | 64 | 1059.9 | 1683.8 | +58.9% | 947.072 | 0.894× | 0.562× |
| Pro | 128 | 1201.0 | 1714.1 | +42.7% | 1131.840 | 0.942× | 0.660× |

## 结果边界

- PR383 数值来自 PR 作者发布的 H20 表；本文没有在同一台 H20-3e 上重新运行 PR383 FP8 MegaMoE/DeepEP baseline，因此相对倍率不能作为同机严格加速比。
- PR383 MegaMoE 使用 `bench_kineto(..., with_multiple_kernels=True)` 统计 MegaMoE GPU kernels；PR383 DeepEP comparator 和 Humming comparator 使用 CUDA Event 包围完整 dispatch/compute/combine 路径。workload 对齐不等于计时实现完全相同。
- benchmark 在计时前检查输出 shape/dtype、finite 值和路由计数一致性，但没有逐元素 PyTorch oracle，因此本数据不是完整数值精度证明。
- 这是单机 EP8、H20-3e、指定 DeepEP/Humming commit 的结果，不能直接外推到 H100/H200、多机或 serving 并发。
- Pro 的 HT `M=8→32` 非单调、`M=64` 明显回落，是当前实测值；未做 profile 前不归因于具体 tactic 或通信行为。
- LL Flash `M=32` 的 observation range 较宽（363.664–444.304 us），报告保留该抖动，不用单次最优值替代中位数。

## 原始证据

- PR383 官方 H20 数据：[DeepGEMM PR #383 Benchmark Results](https://github.com/deepseek-ai/DeepGEMM/pull/383)
- 测试入口：[bench_deepep_humming_mxfp4afp8_ht_sm90.py](./bench_deepep_humming_mxfp4afp8_ht_sm90.py)、[bench_deepep_humming_mxfp4afp8_ll_sm90.py](./bench_deepep_humming_mxfp4afp8_ll_sm90.py)
- 完整汇总：[sm90_humming_mxfp4afp8_deepep_pr383_h203e_result.md](./results/20260811-h203e-pr383-r5-complete/sm90_humming_mxfp4afp8_deepep_pr383_h203e_result.md)
- 结构化数据：[CSV](./results/20260811-h203e-pr383-r5-complete/sm90_humming_mxfp4afp8_deepep_pr383_h203e_result.csv)、[JSON](./results/20260811-h203e-pr383-r5-complete/sm90_humming_mxfp4afp8_deepep_pr383_h203e_result.json)
- 运行参数：[run_config.txt](./results/20260811-h203e-pr383-r5-complete/run_config.txt)
- 环境快照与脚本哈希：[`environment/`](./results/20260811-h203e-pr383-r5-complete/environment/)
