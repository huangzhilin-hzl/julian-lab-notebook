# SM90 Humming MXFP4A-FP8 MegaMoE Auto 与 PR #383 对比（H20，fast-math=1）

## 结论

本次在 8× NVIDIA H20 上完成 Flash 与 Pro 两组 workload。routed-only（`shared=0`）和 fused shared-expert（`shared=1`）各 22 个 case，共 44/44 个 case 成功。

- routed-only 以 `max-rank median` 对比 PR #383：11 个 case 更快、11 个更慢；22 个 case 的整体几何平均为快 0.4%。
- Flash 的几何平均为慢 0.7%。`M=128`、`M=256` 分别快 1.8%、11.0%；其余 9 个点慢 0.5%～5.6%。
- Pro 的几何平均为快 1.5%。`M=8～2048` 全部快 0.3%～4.0%；`M=4096`、`M=8192` 分别慢 0.9%、1.8%。
- 加入 1 个 fused shared expert 后，Flash 相对 routed-only 在 `M<=128` 增加 31.2%～70.7%，在 `M>=256` 增加 9.2%～25.9%；Pro 对应增加 52.6%～83.8% 和 13.9%～27.0%。

需要注意，目标实现为 FP8 activation × MXFP4 weight，PR #383 表格为 FP8 MegaMoE routed-only 基线；workload shape 对齐，但数值格式并不完全相同。`shared=1` 额外执行 shared expert，不能直接与 PR #383 作等 workload 回归判断，其表格主要用于观察相对本分支 `shared=0` 的增量。

## 测试对象

| 项目 | 内容 |
|---|---|
| 目标分支 | [`molou/support_sm90_humming_mxfp4afp8_megamoe_auto`](https://github.com/huangzhilin-hzl/DeepGEMM/tree/molou/support_sm90_humming_mxfp4afp8_megamoe_auto) |
| 目标提交 | [`e21f81c78d0cf9d759cfec341253aaa2d25a8eb7`](https://github.com/huangzhilin-hzl/DeepGEMM/commit/e21f81c78d0cf9d759cfec341253aaa2d25a8eb7) |
| Benchmark | [`tests/bench_mega_moe_sm90.py`](https://github.com/huangzhilin-hzl/DeepGEMM/blob/e21f81c78d0cf9d759cfec341253aaa2d25a8eb7/tests/bench_mega_moe_sm90.py) |
| 对照 | [DeepGEMM PR #383](https://github.com/deepseek-ai/DeepGEMM/pull/383) 中的 H20 MegaMoE 数据 |
| GPU | NVIDIA H20 × 8 |
| 软件栈 | PyTorch 2.11.0+cu130，CUDA 13.0 |
| 日期 | 2026-08-20 |

## 测试配置

| 参数 | 值 |
|---|---|
| ranks | 8 |
| workload | `flash`、`pro` |
| M | 8、16、32、64、128、256、512、1024、2048、4096、8192 |
| shared experts | routed-only: 0；fused: 1 |
| fast-math | 1 |
| flush-L2 | 1 |
| seed | 0 |
| activation clamp | 10.0 |
| masked ratio | 0.0 |
| warmups | 1 |
| tests | 20 |
| repeats | `M<=128`: 50；`M>=256`: 3 |
| 计时范围 | `sm90_fp8_mxfp4_mega_moe_persistent_impl` kernel only |

指标定义：

- `rank0 median`：每个 repeat 内由 `bench_kineto(num_tests=20)` 取得 rank 0 上指定 persistent kernel 的时间，再对 repeats 取中位数。
- `max-rank median`：每个 repeat 先取 8 个 rank 的最大 kernel 时间，再对 repeats 取中位数；本文以此作为主比较值。
- routed-only `delta`：`本次 shared0 max-rank median / PR383 - 1`。负值表示本次更快，正值表示本次更慢。
- shared `delta`：`本次 shared1 max-rank median / 本次 shared0 max-rank median - 1`，表示 fused shared expert 增加的 kernel 时间。
- 计时仅包含选中的 persistent kernel call；权重量化、权重预处理、输入 buffer copy、Python 和进程间初始化开销不计入表中耗时。

## Flash routed-only workload（shared=0）

| M | PR383 H20 MegaMoE (µs) | 本次 rank0 median (µs) | 本次 max-rank median (µs) | delta |
|---:|---:|---:|---:|---:|
| 8 | 273.1 | 271.7 | 280.4 | +2.7% |
| 16 | 304.4 | 292.1 | 306.6 | +0.7% |
| 32 | 302.0 | 306.1 | 318.9 | +5.6% |
| 64 | 340.7 | 337.4 | 358.9 | +5.3% |
| 128 | 414.4 | 390.4 | 406.8 | -1.8% |
| 256 | 569.5 | 503.6 | 506.8 | -11.0% |
| 512 | 922.0 | 928.6 | 928.6 | +0.7% |
| 1024 | 1516.6 | 1514.0 | 1555.0 | +2.5% |
| 2048 | 2735.1 | 2738.0 | 2768.0 | +1.2% |
| 4096 | 5116.0 | 5128.0 | 5144.0 | +0.5% |
| 8192 | 9749.0 | 9917.0 | 9928.0 | +1.8% |

## Pro routed-only workload（shared=0）

| M | PR383 H20 MegaMoE (µs) | 本次 rank0 median (µs) | 本次 max-rank median (µs) | delta |
|---:|---:|---:|---:|---:|
| 8 | 768.0 | 729.1 | 749.0 | -2.5% |
| 16 | 950.3 | 903.0 | 922.5 | -2.9% |
| 32 | 1026.3 | 976.2 | 985.2 | -4.0% |
| 64 | 1059.9 | 1013.0 | 1023.5 | -3.4% |
| 128 | 1201.0 | 1179.5 | 1194.0 | -0.6% |
| 256 | 1639.9 | 1632.0 | 1635.0 | -0.3% |
| 512 | 2599.0 | 2527.0 | 2557.0 | -1.6% |
| 1024 | 4036.0 | 3902.0 | 3923.0 | -2.8% |
| 2048 | 6986.0 | 6930.0 | 6952.0 | -0.5% |
| 4096 | 12932.0 | 13013.0 | 13044.0 | +0.9% |
| 8192 | 24777.0 | 25187.0 | 25225.0 | +1.8% |

## Fused shared-expert workload（shared=1）

shared weights 使用 replicated FP8 与 FP32 block-(128,128) scales，routed weights 保持 MXFP4；shared L1、SwiGLU、shared L2 与 routed MegaMoE 在同一个 persistent kernel 中执行。下表的 `delta` 仅比较本分支同机型、同参数的 `shared=1` 与 `shared=0`。

| Model | M | shared=0 max-rank (µs) | shared=1 rank0 (µs) | shared=1 max-rank (µs) | delta |
|---|---:|---:|---:|---:|---:|
| Flash | 8 | 280.4 | 467.4 | 478.6 | +70.7% |
| Flash | 16 | 306.6 | 505.2 | 510.4 | +66.5% |
| Flash | 32 | 318.9 | 504.9 | 520.8 | +63.3% |
| Flash | 64 | 358.9 | 522.6 | 528.1 | +47.1% |
| Flash | 128 | 406.8 | 528.3 | 533.8 | +31.2% |
| Flash | 256 | 506.8 | 549.9 | 553.6 | +9.2% |
| Flash | 512 | 928.6 | 1064.0 | 1070.0 | +15.2% |
| Flash | 1024 | 1555.0 | 1860.0 | 1864.0 | +19.9% |
| Flash | 2048 | 2768.0 | 3390.0 | 3398.0 | +22.8% |
| Flash | 4096 | 5144.0 | 6447.0 | 6473.0 | +25.8% |
| Flash | 8192 | 9928.0 | 12381.0 | 12497.0 | +25.9% |
| Pro | 8 | 749.0 | 1306.0 | 1316.0 | +75.7% |
| Pro | 16 | 922.5 | 1688.5 | 1696.0 | +83.8% |
| Pro | 32 | 985.2 | 1749.5 | 1753.5 | +78.0% |
| Pro | 64 | 1023.5 | 1770.0 | 1779.0 | +73.8% |
| Pro | 128 | 1194.0 | 1808.5 | 1822.0 | +52.6% |
| Pro | 256 | 1635.0 | 1853.0 | 1863.0 | +13.9% |
| Pro | 512 | 2557.0 | 2980.0 | 3003.0 | +17.4% |
| Pro | 1024 | 3923.0 | 4943.0 | 4981.0 | +27.0% |
| Pro | 2048 | 6952.0 | 8680.0 | 8692.0 | +25.0% |
| Pro | 4096 | 13044.0 | 16319.0 | 16321.0 | +25.1% |
| Pro | 8192 | 25225.0 | 31810.0 | 31817.0 | +26.1% |

## 有效性与脱敏说明

- routed-only 与 fused shared-expert 共 44/44 个正式 case 完成；所有结果均为正的有限值，四轮进程退出码均为 0。
- 正式测试前确认 8 张 H20 无其他 GPU 计算进程；每轮结束后均重新检查。最终检查为 0 个计算进程、最大显存占用 4 MiB、最大 GPU 利用率 0%。
- 每个正式小 M case 含 50 个 observation，大 M case含 3 个 observation；每个 observation 均含 20 次 cold-L2 kernel launch。
- 仅发布聚合后的 benchmark 数值和必要配置；未发布节点、Pod、网络、私有镜像、本地路径、原始日志或环境快照。

机器可读数据：

- routed-only 与 PR #383：[`results/20260820-h20-mxfp4-auto-fastmath1/summary.csv`](results/20260820-h20-mxfp4-auto-fastmath1/summary.csv)
- fused shared-expert 与 routed-only：[`results/20260820-h20-mxfp4-auto-fastmath1/shared1-comparison.csv`](results/20260820-h20-mxfp4-auto-fastmath1/shared1-comparison.csv)
