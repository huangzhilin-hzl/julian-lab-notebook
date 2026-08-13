# PR383 FP8 MegaMoE vs FlashInfer PR4113/PR4069 SM90 MegaMoE：H20/H20-3e 对比

## 结论

- 本文按 [DeepGEMM PR #383](https://github.com/deepseek-ai/DeepGEMM/pull/383) 的 H20 HT/LL 表格展示，保留官方 `MegaMoE / DeepEP / Speedup` 三列，并追加 FlashInfer PR4113、PR4069 实测及相对倍率。
- FlashInfer PR4069 在 8× NVIDIA H20-3e 上完成 22/22 case、66/66 observation、528/528 rank stat，失败数为 0，`torchrun` exit 0。
- HT 中，PR4069 延迟是 PR383 MegaMoE 的 1.202–3.363×。相对 PR383 DeepEP HT，Flash 为 0.921–1.664×，Pro 为 0.962–1.894×；只有两个模型的 `M=8` 数值低于 DeepEP HT。
- LL 中，PR4069 相对 PR383 DeepEP LL：Flash 为 1.088–1.917×，Pro 为 1.026–2.012×，10 个小 M 对照点均高于 DeepEP LL。
- 相对此前 H20-3e PR4113 结果，PR4069 为 0.851–2.264×；只有 Flash `M=512/1024` 更低。该比较跨测试环境且接口计时边界不同，只用于描述数值分布。
- 上述倍率是跨机器、跨 commit、跨计时边界的描述性对照，不是严格 A/B：PR383 表标注为 H20，本次 PR4069 复测是 H20-3e。

## Workload 与测试口径

| Shape | Hidden | Intermediate | Experts | Top-k | EP |
|:---|---:|---:|---:|---:|---:|
| Flash | 4096 | 2048 | 256 | 6 | 8 |
| Pro | 7168 | 3072 | 384 | 6 | 8 |

| 项目 | HT | LL |
|:---|:---|:---|
| PR383 DeepEP | grouped-contiguous FP8 dispatch/combine | low-latency grouped-masked FP8 dispatch/combine |
| FlashInfer | 同一个 `sm90_push_fp8` MegaKernel forward 路径 | 同一个 `sm90_push_fp8` MegaKernel forward 路径 |
| M | 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 | 8, 16, 32, 64, 128 |
| Capacity | `cap=M` | `cap=M` |
| FlashInfer 计时内 | `stage_inputs(...) + compute(...)`：push dispatch、wait/compact、FC1、SwiGLU、FC2、grouped combine 与 round acknowledgement | 与 HT 表相同；使用相同小 M 实测值 |
| FlashInfer 计时外 | 权重预处理、workspace 分配、JIT、warmup、L2 flush、barrier | 与 HT 表相同 |

每个 case 固定 `seed=101`、3 个 observation；每个 observation 先 warmup 5 次，再采集 20 个 sample。每个 sample 前冲刷 8,000,000,000 bytes L2 buffer。单个 observation 先取每个 rank 的 20 次 CUDA Event 中位数，再取各 rank 最大值；表中 PR4069 数值为 3 个 observation 的中位数。

PR383 官方 `Speedup` 按 `DeepEP / MegaMoE - 1` 计算。PR4113 列来自 2026-08-11 的 H20-3e 实测；本文对 PR4069 展示三列延迟倍率：

- `PR4069 / PR4113`：PR4069 实测除以 PR4113 实测；小于 1 表示 PR4069 数值更低。
- `PR4069 / MegaMoE`：PR4069 实测除以 PR383 MegaMoE；小于 1 表示 PR4069 数值更低。
- `PR4069 / DeepEP`：PR4069 实测除以 PR383 DeepEP；小于 1 表示 PR4069 数值更低。

FlashInfer PR4069 这里只有一个 `sm90_push_fp8` forward backend。将其 `M=8–128` 数据同时放进 HT、LL 对照表，是为了对齐 PR383 的表格分组，不代表 PR4069 存在独立 HT/LL 两种执行模式。

## 测试环境

| 项目 | 值 |
|:---|:---|
| 测试拓扑 | 单机 NVIDIA H20-3e ×8，EP8 |
| FlashInfer PR4069 head | `301f8ce3dd42646bb12707251f50db619fb5c653` |
| FlashInfer PR4113 head | `28483960d7a56dd6a77e735f2c874b8e4dbd9d44` |
| Backend | `sm90_push_fp8`，FP8 payload/combine，deduplicated dispatch，grouped combine，fused FC1 epilogue |
| Runtime | Python 3.12.3，PyTorch 2.11.0+cu130，CUDA Toolkit 13.0，Driver 570.133.20 |
| 加载方式 | PR4069 source tree 通过 `PYTHONPATH` 覆盖系统 FlashInfer 0.6.12；PR4069 A2A/GEMM 均在测试环境内 JIT 编译 |
| Benchmark SHA256 | `35ab2a3049e184f46ae32ff4487dafe18a1a7a900c43007356b76efea0b6f2e7` |
| 完成日期 | 2026-08-13 |

## High Throughput：PR383 H20 vs FlashInfer PR4113/PR4069 H20-3e

| Model | M | PR383 MegaMoE FP8 (us) | PR383 DeepEP HT (us) | PR383 Speedup | FlashInfer PR4113 (us) | FlashInfer PR4069 (us) | PR4069 / PR4113 | PR4069 / MegaMoE | PR4069 / DeepEP HT |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Flash | 8 | 273.1 | 566.7 | +107.5% | 489.136 | 521.728 | 1.067× | 1.910× | 0.921× |
| Flash | 16 | 304.4 | 619.2 | +103.4% | 515.936 | 1013.104 | 1.964× | 3.328× | 1.636× |
| Flash | 32 | 302.0 | 618.6 | +104.8% | 531.456 | 1015.744 | 1.911× | 3.363× | 1.642× |
| Flash | 64 | 340.7 | 622.6 | +82.7% | 518.160 | 1035.872 | 1.999× | 3.040× | 1.664× |
| Flash | 128 | 414.4 | 638.5 | +54.1% | 553.616 | 1050.880 | 1.898× | 2.536× | 1.646× |
| Flash | 256 | 569.5 | 688.7 | +20.9% | 910.480 | 1104.624 | 1.213× | 1.940× | 1.604× |
| Flash | 512 | 922.0 | 1057.3 | +14.7% | 1287.776 | 1183.232 | 0.919× | 1.283× | 1.119× |
| Flash | 1024 | 1516.6 | 1983.4 | +30.8% | 2513.120 | 2139.568 | 0.851× | 1.411× | 1.079× |
| Flash | 2048 | 2735.1 | 3419.5 | +25.0% | 2952.704 | 3664.016 | 1.241× | 1.340× | 1.072× |
| Flash | 4096 | 5116.0 | 6087.7 | +19.0% | 5585.488 | 6646.416 | 1.190× | 1.299× | 1.092× |
| Flash | 8192 | 9749.0 | 11779.2 | +20.8% | 11008.384 | 12565.856 | 1.141× | 1.289× | 1.067× |
| Pro | 8 | 768.0 | 1349.2 | +75.7% | 1070.896 | 1298.288 | 1.212× | 1.690× | 0.962× |
| Pro | 16 | 950.3 | 1585.8 | +66.9% | 1327.184 | 3004.224 | 2.264× | 3.161× | 1.894× |
| Pro | 32 | 1026.3 | 1770.2 | +72.5% | 1495.648 | 3258.640 | 2.179× | 3.175× | 1.841× |
| Pro | 64 | 1059.9 | 1787.6 | +68.7% | 1501.136 | 3286.352 | 2.189× | 3.101× | 1.838× |
| Pro | 128 | 1201.0 | 1803.9 | +50.2% | 1519.344 | 3349.328 | 2.204× | 2.789× | 1.857× |
| Pro | 256 | 1639.9 | 1857.7 | +13.3% | 2106.096 | 3392.800 | 1.611× | 2.069× | 1.826× |
| Pro | 512 | 2599.0 | 2898.8 | +11.5% | 2863.632 | 3512.048 | 1.226× | 1.351× | 1.212× |
| Pro | 1024 | 4036.0 | 5412.7 | +34.1% | 5440.016 | 5730.672 | 1.053× | 1.420× | 1.059× |
| Pro | 2048 | 6986.0 | 8067.9 | +15.5% | 7451.168 | 8889.232 | 1.193× | 1.272× | 1.102× |
| Pro | 4096 | 12932.0 | 14614.0 | +13.0% | 13886.656 | 16103.889 | 1.160× | 1.245× | 1.102× |
| Pro | 8192 | 24777.0 | 28184.2 | +13.8% | 27537.552 | 29789.968 | 1.082× | 1.202× | 1.057× |

HT 的数值分布：

- Flash 相对 DeepEP HT：`M=8` 为 0.921×，其余 10 个点为 1.067–1.664×。
- Pro 相对 DeepEP HT：`M=8` 为 0.962×，其余 10 个点为 1.057–1.894×。
- 两个模型相对 PR383 MegaMoE 均高于 1；Flash 为 1.283–3.363×，Pro 为 1.202–3.175×。
- 相对 PR4113，PR4069 只有 Flash `M=512/1024` 更低（0.919× / 0.851×）；Flash 其余点为 1.067–1.999×，Pro 全部点为 1.053–2.264×。

## Low Latency：PR383 H20 vs FlashInfer PR4113/PR4069 H20-3e

| Model | M | PR383 MegaMoE FP8 (us) | PR383 DeepEP LL (us) | PR383 Speedup | FlashInfer PR4113 (us) | FlashInfer PR4069 (us) | PR4069 / PR4113 | PR4069 / MegaMoE | PR4069 / DeepEP LL |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Flash | 8 | 273.1 | 479.7 | +75.7% | 489.136 | 521.728 | 1.067× | 1.910× | 1.088× |
| Flash | 16 | 304.4 | 528.4 | +73.6% | 515.936 | 1013.104 | 1.964× | 3.328× | 1.917× |
| Flash | 32 | 302.0 | 531.8 | +76.1% | 531.456 | 1015.744 | 1.911× | 3.363× | 1.910× |
| Flash | 64 | 340.7 | 552.2 | +62.1% | 518.160 | 1035.872 | 1.999× | 3.040× | 1.876× |
| Flash | 128 | 414.4 | 552.8 | +33.4% | 553.616 | 1050.880 | 1.898× | 2.536× | 1.901× |
| Pro | 8 | 768.0 | 1265.6 | +64.8% | 1070.896 | 1298.288 | 1.212× | 1.690× | 1.026× |
| Pro | 16 | 950.3 | 1493.3 | +57.1% | 1327.184 | 3004.224 | 2.264× | 3.161× | 2.012× |
| Pro | 32 | 1026.3 | 1667.2 | +62.4% | 1495.648 | 3258.640 | 2.179× | 3.175× | 1.955× |
| Pro | 64 | 1059.9 | 1683.8 | +58.9% | 1501.136 | 3286.352 | 2.189× | 3.101× | 1.952× |
| Pro | 128 | 1201.0 | 1714.1 | +42.7% | 1519.344 | 3349.328 | 2.204× | 2.789× | 1.954× |

LL 的数值分布：Flash 相对 DeepEP LL 为 1.088–1.917×，Pro 为 1.026–2.012×。两个模型均在 `M=8` 最接近 DeepEP LL，`M=16–128` 的 PR4069 延迟约为 DeepEP LL 的 1.876–2.012×。相对 PR4113，Flash 为 1.067–1.999×，Pro 为 1.212–2.264×。

## 结果边界

- PR383 数值来自 PR 作者发布的 H20 表；本文没有在同一台 H20-3e 上重新运行 PR383 FP8 MegaMoE/DeepEP baseline，因此相对倍率不能作为同机严格加速比。
- PR383 MegaMoE 使用 Kineto 汇总两个指定 DeepGEMM kernel；PR4069 使用 CUDA Event 包围 `stage_inputs(...) + compute(...)` 的完整 push round。workload 对齐不等于计时范围完全相同。
- PR4113 使用 CUDA Event 包围 `backend.compute(output=None)` 的 pull-style compute 路径；PR4069 包含在 `stage_inputs(...)` 中启动的 push dispatch，因此 `PR4069 / PR4113` 也不是相同接口、相同计时边界的严格 A/B。
- PR4069 的 HT/LL 表使用同一个 forward 路径和同一组小 M 数据；表格分组只用于匹配 PR383 的展示方式。
- Benchmark 验证执行成功、输出 shape/dtype 和 finite，但没有逐元素 PyTorch cross-backend oracle，因此本数据不是完整数值精度证明。
- 测试使用共享 GPU 环境；测试前连续 5 次监测 GPU 利用率均为 0%，所有 point 的 8 GiB L2 flush 均成功分配。该环境不是严格独占 GPU，结果应按指定测试环境的实测理解。
- PR4069 通过 source-tree `PYTHONPATH` overlay 运行；系统 wheel 仍为 FlashInfer 0.6.12。A2A 与 GEMM 模块来自 PR4069 源码并在测试环境内成功 JIT，但这不是正式 PR4069 wheel 安装。

## 原始证据

- PR383 官方 H20 数据：[DeepGEMM PR #383 Benchmark Results](https://github.com/deepseek-ai/DeepGEMM/pull/383)
- PR4113 H20-3e 数据：[flashinfer_pr4113_megamoe_pr383_h20_3e_20260811.md](./flashinfer_pr4113_megamoe_pr383_h20_3e_20260811.md)
- 测试入口：[bench_flashinfer_pr4069_megamoe_sm90.py](./bench_flashinfer_pr4069_megamoe_sm90.py)
- 本文公开聚合后的性能数据、公开 commit、benchmark 哈希和复现参数；包含运行环境标识、进程信息及完整环境快照的原始日志不随公开报告发布。
