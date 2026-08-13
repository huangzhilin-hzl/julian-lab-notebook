# FlashInfer PR4113 SM90 Pull MegaMoE：Modal H100 EP8 实测

## 结论

- FlashInfer PR4113 在 Modal 单机 8× NVIDIA H100 80GB HBM3 上完成 22/22 case、536/536 observation、4,288/4,288 rank stat，失败数为 0，`torchrun` exit 0。
- Flash workload 延迟为 721.296–6,979.280 us；Pro workload 延迟为 1,504.600–17,958.128 us。
- 本次运行使用仓库 `main` 上的 benchmark 原文，固定到 notebook commit `982ac7335ad5eab144be381e2cc9aa972cd2d218`，脚本 SHA256 为 `45f8a57615c5d71c2adf416612fe51f009c43ba2fc84bc268f103338962edc16`。

## Workload 与计时口径

| Shape | Hidden | Intermediate | Experts | Top-k | EP |
|:---|---:|---:|---:|---:|---:|
| Flash | 4096 | 2048 | 256 | 6 | 8 |
| Pro | 7168 | 3072 | 384 | 6 | 8 |

- `M/rank`：8、16、32、64、128、256、512、1024、2048、4096、8192；所有 point 固定 workspace `cap=8192`。
- 固定 `seed=0`；`M<=128` 使用 50 个 observation，其余使用 3 个 observation。每个 observation 先 warmup 5 次，再采集 20 个 sample。
- 每个 sample 前冲刷 8,000,000,000 bytes L2 buffer。单个 observation 取 8 个 rank 各自 CUDA Event 中位数的最大值，表中再取各 observation 的中位数。
- `stage_inputs(...)` 在计时前仅执行一次；CUDA Event 只覆盖 `backend.compute(workspace, transformed, output=None)`，即 PR4113 fused dispatch、FC1、SwiGLU、FC2 与默认 TopkReduce。
- 权重预处理、workspace 分配、`stage_inputs(...)`、JIT、warmup、L2 flush 和 barrier 不在 CUDA Event 计时范围内。

## 测试环境

| 项目 | 值 |
|:---|:---|
| GPU | Modal 单机 NVIDIA H100 80GB HBM3 ×8，强制 `H100!:8` |
| FlashInfer PR4113 head | `28483960d7a56dd6a77e735f2c874b8e4dbd9d44` |
| FlashInfer version | `0.6.18+pr4113`，source-tree `PYTHONPATH` overlay |
| Backend | `sm90_pull_fp8`，blockwise FP8，`swap_ab`，tile `(256, 32, 128)`，atomic-counter load balance |
| Runtime | Python 3.12.13，PyTorch 2.12.0+cu130，CUDA 13.0 |
| 完成日期 | 2026-08-13 |

## 结果

| Model | M | H100 median (us) | Obs min (us) | Obs max (us) | tokens/rank/s | Observations |
|:---|---:|---:|---:|---:|---:|---:|
| Flash | 8 | 721.296 | 702.800 | 742.544 | 11091.1 | 50 |
| Flash | 16 | 747.032 | 734.624 | 765.792 | 21418.1 | 50 |
| Flash | 32 | 761.672 | 749.632 | 779.440 | 42012.8 | 50 |
| Flash | 64 | 762.264 | 751.280 | 774.176 | 83960.4 | 50 |
| Flash | 128 | 780.944 | 765.040 | 799.760 | 163904.2 | 50 |
| Flash | 256 | 983.792 | 979.088 | 1006.800 | 260217.6 | 3 |
| Flash | 512 | 1222.096 | 1213.824 | 1228.192 | 418952.3 | 3 |
| Flash | 1024 | 1573.248 | 1572.160 | 1597.296 | 650882.8 | 3 |
| Flash | 2048 | 2604.928 | 2586.320 | 2628.512 | 786202.1 | 3 |
| Flash | 4096 | 3727.168 | 3724.400 | 3728.736 | 1098957.7 | 3 |
| Flash | 8192 | 6979.280 | 6971.024 | 6990.976 | 1173760.0 | 3 |
| Pro | 8 | 1504.600 | 1486.992 | 1529.472 | 5317.0 | 50 |
| Pro | 16 | 1644.808 | 1628.416 | 1663.872 | 9727.6 | 50 |
| Pro | 32 | 1763.728 | 1745.872 | 1778.704 | 18143.4 | 50 |
| Pro | 64 | 1780.680 | 1768.368 | 1792.480 | 35941.3 | 50 |
| Pro | 128 | 1801.320 | 1789.216 | 1870.656 | 71059.0 | 50 |
| Pro | 256 | 2340.208 | 2338.944 | 2349.952 | 109392.0 | 3 |
| Pro | 512 | 3193.600 | 3186.064 | 3211.520 | 160320.6 | 3 |
| Pro | 1024 | 3985.392 | 3981.408 | 4042.352 | 256938.3 | 3 |
| Pro | 2048 | 6018.000 | 6015.488 | 6047.200 | 340312.4 | 3 |
| Pro | 4096 | 9845.008 | 9837.728 | 9859.136 | 416048.4 | 3 |
| Pro | 8192 | 17958.128 | 17933.184 | 17982.384 | 456172.3 | 3 |

## 结果边界

- Modal 容器屏蔽了 `nvidia-smi topo -m` 的 NVML 拓扑矩阵，因此本文不声明具体 NVLink 拓扑；完整 8-rank backend 初始化和全部 workload 均成功。
- NVSHMEM 的 IBRC transport 输出了 `get_device_list` warning，CuTeDSL 的 IKET tracing marker 也不可用；两者均未影响单机 GPU 路径、结果完整性或进程退出状态。
- 本 benchmark 只验证输出 shape、dtype 和 finite，没有逐元素 PyTorch cross-backend oracle，因此本数据不是完整数值精度证明。
- PR4069 报告测量 `stage_inputs(...) + compute(...)`，本文只测量 `compute(...)`；两组绝对延迟不能作为严格的 push/pull 同口径 A/B。

## 复现与证据

- Benchmark：[bench_flashinfer_pr4113_megamoe_sm90.py](./bench_flashinfer_pr4113_megamoe_sm90.py)
- Modal runner：[modal_bench_flashinfer_pr4113_megamoe_h100.py](./modal_bench_flashinfer_pr4113_megamoe_h100.py)
- Modal run：`pr4113-h100-full-20260813a`，App ID `ap-BLyA86FKZyzXqxHhwkgfj5`
- `full.log` SHA256：`e77c4476c3eed7b5831a551c757fc538eb6c9722687b63434a4a3f765c011834`
- 本文仅公开聚合性能数据、公开 commit 和复现参数；包含设备标识及完整环境快照的原始日志不随报告发布。
