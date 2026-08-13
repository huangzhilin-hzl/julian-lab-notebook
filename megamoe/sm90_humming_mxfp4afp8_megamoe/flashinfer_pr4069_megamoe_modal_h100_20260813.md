# FlashInfer PR4069 SM90 Push MegaMoE：Modal H100 EP8 实测

## 结论

- FlashInfer PR4069 在 Modal 单机 8× NVIDIA H100 80GB HBM3 上完成 22/22 case、66/66 observation、528/528 rank stat，失败数为 0，`torchrun` exit 0。
- Flash workload 延迟为 560.704–4,680.960 us；Pro workload 延迟为 1,138.592–10,312.608 us。
- 相对同一 PR4069 bench 的 H20-3e 结果，H100 在 Flash `M=8` 为 1.075×，其余 Flash 点为 0.373–0.628×；Pro 全部点为 0.342–0.877×。
- 这是跨环境、跨 PyTorch 版本的描述性对照，不是严格的同机 A/B。

## Workload 与计时口径

| Shape | Hidden | Intermediate | Experts | Top-k | EP |
|:---|---:|---:|---:|---:|---:|
| Flash | 4096 | 2048 | 256 | 6 | 8 |
| Pro | 7168 | 3072 | 384 | 6 | 8 |

- `M/rank`：8、16、32、64、128、256、512、1024、2048、4096、8192，且每个 point 使用 `cap=M`。
- 每个 case 固定 `seed=101`、3 个 observation；每个 observation 先 warmup 5 次，再采集 20 个 sample。
- 每个 sample 前冲刷 8,000,000,000 bytes L2 buffer。单个 observation 取 8 个 rank 各自 CUDA Event 中位数的最大值，表中再取 3 个 observation 的中位数。
- 计时范围是 `stage_inputs(...) + compute(...)`，包含 push dispatch、wait/compact、FC1、SwiGLU、FC2、grouped combine 和 round acknowledgement。
- 权重预处理、workspace 分配、JIT、warmup、L2 flush 和 barrier 不在 CUDA Event 计时范围内。

## 测试环境

| 项目 | 值 |
|:---|:---|
| GPU | Modal 单机 NVIDIA H100 80GB HBM3 ×8，强制 `H100!:8` |
| FlashInfer PR4069 head | `301f8ce3dd42646bb12707251f50db619fb5c653` |
| FlashInfer version | `0.6.18+pr4069`，source-tree `PYTHONPATH` overlay |
| Backend | `sm90_push_fp8`，FP8 payload/combine，deduplicated dispatch，grouped combine，fused FC1 epilogue |
| Runtime | Python 3.12.13，PyTorch 2.12.0+cu130，CUDA 13.0.1 |
| P2P validation | 使用默认验证，未设置 `allow_unverified_p2p` |
| 完成日期 | 2026-08-13 |

## 结果

| Model | M | H100 median (us) | Obs min (us) | Obs max (us) | tokens/rank/s | H20-3e (us) | H100 / H20-3e |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Flash | 8 | 560.704 | 554.768 | 572.016 | 14267.8 | 521.728 | 1.075× |
| Flash | 16 | 610.256 | 603.792 | 617.344 | 26218.5 | 1013.104 | 0.602× |
| Flash | 32 | 632.096 | 623.456 | 633.264 | 50625.2 | 1015.744 | 0.622× |
| Flash | 64 | 635.616 | 617.232 | 635.952 | 100689.7 | 1035.872 | 0.614× |
| Flash | 128 | 644.256 | 634.816 | 649.872 | 198678.8 | 1050.880 | 0.613× |
| Flash | 256 | 668.240 | 659.104 | 669.792 | 383095.9 | 1104.624 | 0.605× |
| Flash | 512 | 743.440 | 725.648 | 744.368 | 688690.4 | 1183.232 | 0.628× |
| Flash | 1024 | 968.832 | 966.208 | 980.640 | 1056942.8 | 2139.568 | 0.453× |
| Flash | 2048 | 1492.448 | 1488.560 | 1493.696 | 1372242.1 | 3664.016 | 0.407× |
| Flash | 4096 | 2561.408 | 2555.840 | 2571.712 | 1599120.5 | 6646.416 | 0.385× |
| Flash | 8192 | 4680.960 | 4679.952 | 4684.544 | 1750068.4 | 12565.856 | 0.373× |
| Pro | 8 | 1138.592 | 1132.096 | 1139.440 | 7026.2 | 1298.288 | 0.877× |
| Pro | 16 | 1396.048 | 1391.456 | 1402.784 | 11460.9 | 3004.224 | 0.465× |
| Pro | 32 | 1432.064 | 1414.544 | 1432.160 | 22345.4 | 3258.640 | 0.439× |
| Pro | 64 | 1483.088 | 1469.296 | 1490.064 | 43153.2 | 3286.352 | 0.451× |
| Pro | 128 | 1459.744 | 1454.432 | 1481.536 | 87686.6 | 3349.328 | 0.436× |
| Pro | 256 | 1517.792 | 1512.720 | 1527.424 | 168666.1 | 3392.800 | 0.447× |
| Pro | 512 | 1622.384 | 1613.216 | 1624.912 | 315585.0 | 3512.048 | 0.462× |
| Pro | 1024 | 2112.864 | 2110.688 | 2117.952 | 484650.2 | 5730.672 | 0.369× |
| Pro | 2048 | 3110.848 | 3109.904 | 3115.152 | 658341.4 | 8889.232 | 0.350× |
| Pro | 4096 | 5514.544 | 5511.296 | 5520.880 | 742763.1 | 16103.889 | 0.342× |
| Pro | 8192 | 10312.608 | 10296.480 | 10332.576 | 794367.4 | 29789.968 | 0.346× |

## 结果边界

- Modal 容器屏蔽了 `nvidia-smi topo -m` 的 NVML 拓扑矩阵，但 PR4069 backend 的默认 P2P validation 和完整 8-rank forward 均成功；本文不据此声明具体 NVLink 拓扑。
- H100 与 H20-3e 使用相同 bench 参数和计时范围，但运行时版本、机器与环境不同，因此 `H100 / H20-3e` 只描述本次测得的数值关系。
- benchmark 验证输出 shape、dtype 和 finite，但没有逐元素 PyTorch cross-backend oracle，因此本数据不是完整数值精度证明。

## 复现与证据

- Benchmark：[bench_flashinfer_pr4069_megamoe_sm90.py](./bench_flashinfer_pr4069_megamoe_sm90.py)
- H20-3e 对照：[flashinfer_pr4069_megamoe_pr383_h20_3e_20260813.md](./flashinfer_pr4069_megamoe_pr383_h20_3e_20260813.md)
- 本文仅公开聚合性能数据、公开 commit 和复现参数；包含设备标识及完整环境快照的原始日志不随报告发布。
