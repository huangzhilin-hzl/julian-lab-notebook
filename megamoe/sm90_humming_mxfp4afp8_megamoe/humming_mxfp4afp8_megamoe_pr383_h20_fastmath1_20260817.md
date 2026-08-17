# SM90 Humming MXFP4A-FP8 MegaMoE 与 PR #383 对比（H20，fast-math=1）

## 结论

本次在 8× NVIDIA H20 上完成 Flash 与 Pro 两组 workload。先跑 `shared=0` 的 routed-only 基线，再补跑 `shared=1` 的 fused shared-expert workload；两轮各 22 个 case，全部运行成功。

- routed-only 以更保守的 `max-rank median` 对比 PR #383：2 个 case 更快、1 个持平、19 个更慢。
- Flash：`M=256` 快 10.1%；`M=512` 基本持平（慢 0.2%）；`M>=1024` 慢 3.5%～5.6%。
- Pro：`M=512` 快 0.8%；`M=1024` 持平；`M>=2048` 慢 1.8%～4.6%。
- 小 M 回退明显：Flash `M=8～64` 慢 46.7%～64.2%，Pro `M=8～64` 慢 54.4%～64.6%。
- 加入 1 个 fused shared expert 后，22 个 case 的耗时相对 routed-only 增加 10.5%～31.1%。Flash 在 `M<=256` 增加 10.5%～18.9%，在 `M>=512` 增加 22.7%～31.1%；Pro 对应增加 13.9%～17.1% 和 21.9%～30.4%。

需要注意，本次目标实现为 FP8 activation × MXFP4 weight，PR #383 表格为 FP8 MegaMoE routed-only 基线；workload shape 对齐，但数值格式并不完全相同。`shared=1` 还额外执行 shared expert，不能与 PR #383 作等 workload 回归判断；shared-expert 表应主要看相对本分支 `shared=0` 的增量。

## 测试对象

| 项目 | 内容 |
|---|---|
| 目标分支 | [`molou/support_sm90_humming_mxfp4afp8_megamoe_opt`](https://github.com/huangzhilin-hzl/DeepGEMM/tree/molou/support_sm90_humming_mxfp4afp8_megamoe_opt) |
| 目标提交 | `5c060dc51427b497384d1b42f2b01263813c2d87` |
| Benchmark | `tests/bench_mega_moe_sm90.py` |
| 对照 | [DeepGEMM PR #383](https://github.com/deepseek-ai/DeepGEMM/pull/383) 中的 H20 MegaMoE 数据 |
| GPU | NVIDIA H20 × 8 |
| 软件栈 | PyTorch 2.11.0+cu130，CUDA 13.0 |
| 日期 | 2026-08-17 |

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
| warmups | 1 |
| tests | 20 |
| repeats | `M<=128`: 50；`M>=256`: 3 |
| 计时范围 | persistent kernel |

指标定义：

- `rank0 median`：每个 repeat 内由 `bench_kineto(num_tests=20)` 得到 rank 0 kernel 时间，再对 repeats 取中位数。
- `max-rank median`：每个 repeat 先取 8 个 rank 的最大 kernel 时间，再对 repeats 取中位数；本文以此作为主比较值。
- routed-only `delta`：`shared0 max-rank median / PR383 - 1`。负值表示本次更快，正值表示本次更慢。
- shared `delta`：`shared1 max-rank median / shared0 max-rank median - 1`，表示 fused shared expert 增加的端到端 kernel 时间。

## Flash routed-only workload（shared=0）

| M | PR383 H20 MegaMoE (µs) | 本次 rank0 median (µs) | 本次 max-rank median (µs) | delta |
|---:|---:|---:|---:|---:|
| 8 | 273.1 | 430.4 | 444.0 | +62.6% |
| 16 | 304.4 | 481.1 | 495.6 | +62.8% |
| 32 | 302.0 | 468.6 | 496.0 | +64.2% |
| 64 | 340.7 | 480.0 | 499.7 | +46.7% |
| 128 | 414.4 | 485.3 | 503.9 | +21.6% |
| 256 | 569.5 | 501.8 | 511.8 | -10.1% |
| 512 | 922.0 | 900.0 | 924.3 | +0.2% |
| 1024 | 1516.6 | 1584.0 | 1590.0 | +4.8% |
| 2048 | 2735.1 | 2794.0 | 2831.0 | +3.5% |
| 4096 | 5116.0 | 5283.0 | 5339.0 | +4.4% |
| 8192 | 9749.0 | 10293.0 | 10299.0 | +5.6% |

## Pro routed-only workload（shared=0）

| M | PR383 H20 MegaMoE (µs) | 本次 rank0 median (µs) | 本次 max-rank median (µs) | delta |
|---:|---:|---:|---:|---:|
| 8 | 768.0 | 1187.0 | 1202.5 | +56.6% |
| 16 | 950.3 | 1543.0 | 1564.0 | +64.6% |
| 32 | 1026.3 | 1605.5 | 1622.0 | +58.0% |
| 64 | 1059.9 | 1616.5 | 1637.0 | +54.4% |
| 128 | 1201.0 | 1625.0 | 1651.5 | +37.5% |
| 256 | 1639.9 | 1663.0 | 1692.0 | +3.2% |
| 512 | 2599.0 | 2563.0 | 2579.0 | -0.8% |
| 1024 | 4036.0 | 3997.0 | 4036.0 | +0.0% |
| 2048 | 6986.0 | 7060.0 | 7114.0 | +1.8% |
| 4096 | 12932.0 | 13321.0 | 13346.0 | +3.2% |
| 8192 | 24777.0 | 25892.0 | 25905.0 | +4.6% |

## Fused shared-expert workload（shared=1）

shared weights 使用 replicated FP8 与 FP32 block-(128,128) scales，routed weights 保持 MXFP4；shared L1、SwiGLU、shared L2 与 routed MegaMoE 在同一个 persistent kernel 中执行。下表的 `delta` 仅比较本分支同机型、同参数的 `shared=1` 与 `shared=0`。

| Model | M | shared=0 max-rank (µs) | shared=1 rank0 (µs) | shared=1 max-rank (µs) | delta |
|---|---:|---:|---:|---:|---:|
| Flash | 8 | 444.0 | 502.2 | 513.8 | +15.7% |
| Flash | 16 | 495.6 | 542.8 | 547.7 | +10.5% |
| Flash | 32 | 496.0 | 548.3 | 553.7 | +11.6% |
| Flash | 64 | 499.7 | 558.6 | 561.5 | +12.4% |
| Flash | 128 | 503.9 | 576.7 | 583.2 | +15.7% |
| Flash | 256 | 511.8 | 604.7 | 608.7 | +18.9% |
| Flash | 512 | 924.3 | 1124.0 | 1134.0 | +22.7% |
| Flash | 1024 | 1590.0 | 2012.0 | 2025.0 | +27.4% |
| Flash | 2048 | 2831.0 | 3690.0 | 3690.0 | +30.3% |
| Flash | 4096 | 5339.0 | 6921.0 | 6921.0 | +29.6% |
| Flash | 8192 | 10299.0 | 13500.0 | 13504.0 | +31.1% |
| Pro | 8 | 1202.5 | 1361.0 | 1369.5 | +13.9% |
| Pro | 16 | 1564.0 | 1772.5 | 1781.0 | +13.9% |
| Pro | 32 | 1622.0 | 1860.0 | 1866.5 | +15.1% |
| Pro | 64 | 1637.0 | 1871.5 | 1880.5 | +14.9% |
| Pro | 128 | 1651.5 | 1923.0 | 1927.0 | +16.7% |
| Pro | 256 | 1692.0 | 1971.0 | 1982.0 | +17.1% |
| Pro | 512 | 2579.0 | 3140.0 | 3145.0 | +21.9% |
| Pro | 1024 | 4036.0 | 5226.0 | 5230.0 | +29.6% |
| Pro | 2048 | 7114.0 | 9266.0 | 9279.0 | +30.4% |
| Pro | 4096 | 13346.0 | 17335.0 | 17356.0 | +30.0% |
| Pro | 8192 | 25905.0 | 33700.0 | 33725.0 | +30.2% |

## 有效性与脱敏说明

- routed-only 与 fused shared-expert 两轮均为 22/22 个 case 完成，结果中未出现非有限值或执行错误。
- Pro `M=8192` 在首轮测试窗口末尾观察到外部 GPU 活动，因此在无外部 GPU 活动的窗口补跑并用复测值替换；两次 `max-rank median` 相差 0.03%。
- fused shared-expert 正式测试前后均未检测到其他 GPU 计算进程，运行后最大 GPU 利用率为 0%。
- 仅发布聚合后的 benchmark 数值和必要配置；未发布内部运行环境标识、网络地址、私有镜像、本地路径、原始日志或环境快照。

机器可读数据：

- routed-only 与 PR #383：[`results/20260817-h20-mxfp4-opt-fastmath1/summary.csv`](results/20260817-h20-mxfp4-opt-fastmath1/summary.csv)
- fused shared-expert 与 routed-only：[`results/20260817-h20-mxfp4-opt-fastmath1/shared1-comparison.csv`](results/20260817-h20-mxfp4-opt-fastmath1/shared1-comparison.csv)
