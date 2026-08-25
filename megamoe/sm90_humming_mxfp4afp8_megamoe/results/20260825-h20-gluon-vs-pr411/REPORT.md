# Gluon MegaMoE H20 测试与 DeepGEMM PR #411 对比

测试日期：2026-08-25（Asia/Shanghai）

## 测试状态

- 运行环境：隔离的 Kubernetes 容器；namespace、工作负载名称、节点标识和网络地址已脱敏。
- 计算资源：单个专用节点上的 8 张 NVIDIA H20。
- 测试前确认 8 张卡空闲；所有 benchmark 串行执行，避免同节点任务干扰。
- GPU：8 × NVIDIA H20，SM90，每卡 97,871 MiB
- 软件基线：Python 3.12.3、PyTorch 2.11.0+cu130、CUDA 13.0、Triton 3.6.0
- 升级复测环境（隔离 venv）：PyTorch 2.13.0+cu130、CUDA 13.0、Triton 3.7.1
- Gluon 源码：`Triton-distributed@ed0be56e967474c289ab0b48097f9f33c030b994`；两个 vendored 文件 SHA256 校验通过。
- Pro 和 Flash 的 smoke、默认正式测试及 PR 对齐测试均成功；PyTorch 2.13.0 / Triton 3.7.1 的 Pro 默认矩阵复测也成功。原版 matrix runner、原版完整语雀命令和自定义每-M-独立-`torchrun` 三组 A/B 均成功。随后完成 72 组 child-harness 一级消融和 36 组 flush 二级消融。有效实验日志错误扫描为空，测试后 GPU 利用率均为 0%，无残留 compute process。

原始环境快照、Pod 定义、子进程日志和 rank samples 保存在本地私有归档中；本文不披露其内部路径和集群标识。可公开复核的数据汇总为同目录下的 CSV 文件。

## 默认语雀口径结果

参数：DSV4 Pro、EP8、固定 cap=64、random top-k、48 experts/wave、fast_math=1、8 GB cold-L2、30 warmups、100 active samples、E2E（pre-dispatch registration + persistent kernel）。`critical_path_median_us` 先逐 iteration 取 8 rank 最大值，再在 100 个样本上取 median。

### Pro

| M / rank | critical path (µs) | legacy max-rank (µs) |
|---:|---:|---:|
| 1 | 306.176 | 306.048 |
| 2 | 386.320 | 385.360 |
| 4 | 588.112 | 586.624 |
| 8 | 770.496 | 769.840 |
| 16 | 966.864 | 966.064 |
| 32 | 1051.984 | 1051.024 |
| 64 | 1086.128 | 1084.480 |

### Flash

参数除模型 shape 和每 wave expert 数量外与 Pro 相同：H=4096、I=2048、E=256、top-k=6、32 experts/wave。

| M / rank | critical path (µs) | legacy max-rank (µs) |
|---:|---:|---:|
| 1 | 133.376 | 132.064 |
| 2 | 193.072 | 192.208 |
| 4 | 270.224 | 269.632 |
| 8 | 290.448 | 289.664 |
| 16 | 320.768 | 318.736 |
| 32 | 335.056 | 333.888 |
| 64 | 351.808 | 350.864 |

## Runtime 版本 A/B（Pro E2E）

在同一隔离环境、同一 8×H20 节点、同一 Gluon 源码和同一评测参数下，将运行时从 PyTorch 2.11.0+cu130 / Triton 3.6.0 升级到 PyTorch 2.13.0+cu130 / Triton 3.7.1。为避免破坏原环境，新版本安装在独立虚拟环境中；正式测试仍采用 Pro、M=1–64、固定 cap=64、30 warmups、100 samples、8 GB L2 flush、E2E critical-path 口径。

| M / rank | Torch 2.11 / Triton 3.6 (µs) | Torch 2.13 / Triton 3.7.1 (µs) | 新版 vs 旧版 | 语雀 (µs) | 新版 vs 语雀 |
|---:|---:|---:|---:|---:|---:|
| 1 | 306.176 | 306.160 | -0.01% | 327.040 | -6.38% |
| 2 | 386.320 | 389.424 | +0.80% | 410.580 | -5.15% |
| 4 | 588.112 | 589.760 | +0.28% | 616.210 | -4.29% |
| 8 | 770.496 | 766.624 | -0.50% | 810.900 | -5.46% |
| 16 | 966.864 | 959.952 | -0.71% | 999.630 | -3.97% |
| 32 | 1051.984 | 1055.952 | +0.38% | 1109.700 | -4.84% |
| 64 | 1086.128 | 1090.736 | +0.42% | 1155.730 | -5.62% |

新版相对旧版只变化 -0.71% 到 +0.80%，方向也不一致，属于正常运行波动；升级后相对语雀仍快 3.97%–6.38%。因此 PyTorch / Triton 版本不是语雀与本次 bench 差异的主要原因。下节 runner A/B 进一步定位了差异来源。

## Runner / child harness 三组 A/B

所有组使用同一隔离环境、同一 8×H20 节点、同一 commit `ed0be56e`、PyTorch 2.13.0+cu130、Triton 3.7.1，以及相同 shape、路由、cap=64、30 warmups、100 active samples、8 GB L2 flush 和 critical-path median 口径。

- `custom continuous`：我们的 bench 在一个 `torchrun` 中连续运行 M=1–64，为上一节结果。
- `custom per-M`：我们的 bench 不改实现，只把每个 M 拆成独立 `torchrun`。
- `original Gluon/e2e`：原版 matrix runner，仅 `--implementations gluon_fused --scope e2e`。
- `original full`：完全复现语雀四实现、`--scope all` 命令；环境同时对齐 `sgl-deep-ep==0.1.0` 和 `sgl-deep-gemm==0.1.5.post2`。

| M | custom continuous (µs) | custom per-M (µs) | original Gluon/e2e (µs) | original full (µs) | 语雀 (µs) |
|---:|---:|---:|---:|---:|---:|
| 1 | 306.160 | 306.112 | 335.360 | 333.200 | 327.040 |
| 2 | 389.424 | 386.272 | 420.032 | 402.416 | 410.580 |
| 4 | 589.760 | 589.232 | 617.360 | 611.952 | 616.210 |
| 8 | 766.624 | 761.744 | 785.216 | 791.328 | 810.900 |
| 16 | 959.952 | 957.680 | 982.176 | 989.712 | 999.630 |
| 32 | 1055.952 | 1055.776 | 1086.640 | 1086.384 | 1109.700 |
| 64 | 1090.736 | 1089.168 | 1122.144 | 1126.912 | 1155.730 |

验证结果：

1. `custom per-M` 相对 `custom continuous` 只变化 -0.81% 到 -0.02%，因此“每个 M 独立 `torchrun`”本身不是主要原因。
2. `original full` 相对 `custom per-M` 慢 2.90%–8.85%，说明主要差异位于原版 child harness 内部执行方式，而不是 matrix wrapper 的进程拆分。
3. `original full` 相对语雀只差 -2.49% 到 +1.88%；原来的 custom runner 相对语雀快 3.97%–6.38%。原版 harness 已复现语雀数据的主要部分，剩余差异落在同 harness 的运行波动、节点频率和温度状态范围内。
4. `original Gluon/e2e` 与 `original full` 没有稳定单向差异，四实现交错运行不是主因。
5. 两边统计公式相同；同一批 custom raw samples 的 `median(per-iteration max-rank)` 与 `max(per-rank median)` 也只差 0.02%–0.29%，不能解释上述差距。

原版 child 与 custom child 的剩余方法差异包括：原版输入生成后调用 `torch.cuda.empty_cache()`、保留额外的原始 L1 FP8 weight、每个 active sample 调用 `torch.cuda.mem_get_info()` 后临时创建/释放 8 GB flush tensor并新建 CUDA Event；custom child 预先计算 flush 大小并预分配 tensor、复用 Event，而且在正式 warmup 前多执行一次输出验证。下面用两级消融逐项定位。

## Child harness 单变量消融

在同一节点和 PyTorch 2.13.0 / Triton 3.7.1 环境下选取 M=1、8、64。每种配置运行 3 个独立 `torchrun`，执行顺序在三轮中交错；每个子进程仍使用原版 Gluon/e2e child、固定 cap=64、30 warmups、100 active samples 和逐 iteration max-rank 后取 median 的统计方法。一级消融共 72 个子进程，72/72 metadata 完整且错误扫描为 0。

| M | 原版 (µs) | 仅持久 flush (µs) | 持久 flush + Event 复用 (µs) | 全部 custom-like (µs) | custom per-M 参考 (µs) |
|---:|---:|---:|---:|---:|---:|
| 1 | 329.136 | 306.384 | 304.560 | 304.176 | 306.112 |
| 8 | 800.608 | 769.168 | 765.152 | 766.112 | 761.744 |
| 64 | 1124.944 | 1091.344 | 1092.880 | 1091.696 | 1089.168 |

表中数值为三轮 latency 的中位数。`全部 custom-like` 同时使用持久 flush、Event 复用、去掉输入后的 `empty_cache`、释放原始 L1 weight 并增加一次 prewarm；它相对 custom per-M 参考只差 -0.63%、+0.57%、+0.23%，已经复现我们的 bench。

各单变量相对同一轮原版的配对百分比中位数如下；负值表示更快：

| 单变量 | M=1 | M=8 | M=64 | 判断 |
|---|---:|---:|---:|---|
| 持久 flush tensor | -6.91% | -3.91% | -2.89% | 唯一跨 M 稳定的大效应 |
| 复用 CUDA Event | +1.98% | +0.96% | -0.74% | 无稳定收益 |
| 不调用输入后的 `empty_cache` | +0.19% | -0.42% | -0.57% | 噪声量级 |
| 释放原始 L1 weight | +0.81% | -0.69% | -0.46% | 噪声量级 |
| 增加一次 prewarm | +3.16% | +0.54% | -0.42% | 无稳定收益，M=1 反而变慢 |
| 持久 flush + Event 复用 | -7.39% | -4.11% | -2.99% | 与持久 flush 单项接近 |
| 全部 custom-like | -7.58% | -3.73% | -2.96% | 与 custom per-M 对齐 |

这一步把差距定位到原版 `_flush_l2()` 路径，但“持久 flush”同时去掉了逐样本显存查询和逐样本 tensor 申请/析构，因此又做了正交二级消融。

## Flush 二级消融：精确到 `mem_get_info()`

二级消融共 36 个独立子进程：4 种模式 × 3 个 M × 3 轮交错运行，36/36 metadata 完整且错误扫描为 0。

- `transient`：原版，每个样本先 `torch.cuda.mem_get_info()`，再申请、清零并释放临时 tensor。
- `transient_fixed`：只在计时循环前计算一次 flush 大小，样本内仍申请、清零并释放临时 tensor。
- `persistent_query`：tensor 持久化，但每个样本仍调用 `torch.cuda.mem_get_info()`。
- `persistent`：大小只计算一次，tensor 持久化，即 custom bench 方式。

| M | transient (µs) | transient_fixed (µs) | persistent_query (µs) | persistent (µs) |
|---:|---:|---:|---:|---:|
| 1 | 336.096 | 304.736 | 332.000 | 305.024 |
| 8 | 797.008 | 767.120 | 792.432 | 765.920 |
| 64 | 1129.904 | 1092.336 | 1118.976 | 1091.728 |

配对差值把两个因素完全拆开：

| 因素 | M=1 | M=8 | M=64 |
|---|---:|---:|---:|
| 临时 tensor 下增加逐样本查询 | +28.112 µs / +9.13% | +29.888 µs / +3.90% | +37.568 µs / +3.44% |
| 持久 tensor 下增加逐样本查询 | +26.976 µs / +8.84% | +27.984 µs / +3.66% | +27.952 µs / +2.56% |
| 无查询时，临时相对持久 tensor | -0.544 µs / -0.18% | +1.360 µs / +0.18% | +0.080 µs / +0.01% |

因此，主要差距不是“每个样本申请/释放 8 GB tensor”，而是原版 `_flush_l2()` 在每个 active sample 中执行的：

```python
free_bytes, _ = torch.cuda.mem_get_info()
```

只要保留这行查询，无论 tensor 是否持久化都会慢约 27–38 µs；只要把 flush 大小移到循环外，无论 tensor 是否持久化都会回到 custom bench 的 305/767/1092 µs 区间。无查询时 tensor 生命周期的配对中位影响不超过 0.18%。

raw rank samples 还显示，有查询时每个 iteration 的 8-rank 延迟跨度中位数明显放大：

| M | transient 查询 (µs) | transient 无查询 (µs) | persistent 查询 (µs) | persistent 无查询 (µs) |
|---:|---:|---:|---:|---:|
| 1 | 43.424 | 12.352 | 38.304 | 13.104 |
| 8 | 37.440 | 6.496 | 31.104 | 5.696 |
| 64 | 44.416 | 6.768 | 30.096 | 4.880 |

机制上的最佳解释是：查询位于循环内的 `dist.barrier()` 之后、timed collective launch 之前，且查询后没有再次对齐 rank；驱动查询耗时的 rank 间抖动使各 rank 进入 persistent fused collective 的时间错开，先进入的 rank 在 Event 区间内等待晚到 rank，最终被 per-iteration max-rank 统计放大。这个解释由 rank-spread 数据支持，但不是对 CUDA 驱动内部行为的直接观测。曾尝试在 flush 后额外插入 NCCL barrier 做因果验证，但该 barrier 本身使 persistent target latency 近乎翻倍，改变了被测执行状态，因此该组结果判为无效并排除。

一级完整汇总见 `child-harness-ablation-pro-e2e.csv`，二级数据与 rank-spread 见 `flush-lifecycle-ablation-pro-e2e.csv`；原始 rank samples 和子进程日志保存在本地私有归档中。

## PR #411 对齐结果

参数：DSV4 Pro、EP8、M=cap、3 observations × 20 samples、5 warmups、20,000,000 cycle barrier sleep、8 GB cold-L2、kernel-only。PR #411 声明沿用 PR #383 的 H20 shape 和计时方法，因此直接比较使用 Gluon 的 `max_rank_median_us`（各 rank 先取 median，再取最大 rank），而非语雀主表的 `critical_path_median_us`。

`Gluon vs PR411` 为 `Gluon / PR411 - 1`，正数表示 Gluon 更慢。PR #411 与 PR #383 两列均来自 PR #411 公布的 H20 High Throughput 表。

### Pro

| M / rank | Gluon legacy (µs) | PR #411 MXFP4×FP8 (µs) | Gluon vs PR411 | PR #383 FP8×FP8 (µs) | Gluon vs PR383 |
|---:|---:|---:|---:|---:|---:|
| 8 | 768.240 | 700.2 | +9.72% | 768.0 | +0.03% |
| 16 | 957.808 | 913.4 | +4.86% | 950.3 | +0.79% |
| 32 | 1050.608 | 985.2 | +6.64% | 1026.3 | +2.37% |
| 64 | 1090.480 | 1013.0 | +7.65% | 1059.9 | +2.89% |
| 128 | 1282.864 | 1178.5 | +8.86% | 1201.0 | +6.82% |
| 256 | 1718.304 | 1593.0 | +7.87% | 1639.9 | +4.78% |
| 512 | 2547.184 | 2474.0 | +2.96% | 2599.0 | -1.99% |
| 1024 | 4513.904 | 3852.0 | +17.18% | 4036.0 | +11.84% |
| 2048 | 8338.912 | 6865.0 | +21.47% | 6986.0 | +19.37% |
| 4096 | 16088.000 | 12832.0 | +25.37% | 12932.0 | +24.40% |
| 8192 | 31435.168 | 24912.0 | +26.18% | 24777.0 | +26.87% |

### Flash

| M / rank | Gluon legacy (µs) | PR #411 MXFP4×FP8 (µs) | Gluon vs PR411 | PR #383 FP8×FP8 (µs) | Gluon vs PR383 |
|---:|---:|---:|---:|---:|---:|
| 8 | 292.400 | 272.2 | +7.42% | 273.1 | +7.07% |
| 16 | 323.344 | 311.1 | +3.94% | 304.4 | +6.22% |
| 32 | 333.760 | 323.1 | +3.30% | 302.0 | +10.52% |
| 64 | 355.936 | 333.9 | +6.60% | 340.7 | +4.47% |
| 128 | 448.480 | 408.5 | +9.79% | 414.4 | +8.22% |
| 256 | 618.576 | 471.7 | +31.14% | 569.5 | +8.62% |
| 512 | 1043.632 | 881.3 | +18.42% | 922.0 | +13.19% |
| 1024 | 1844.528 | 1449.0 | +27.30% | 1516.6 | +21.62% |
| 2048 | 3478.368 | 2665.0 | +30.52% | 2735.1 | +27.18% |
| 4096 | 6643.440 | 5005.0 | +32.74% | 5116.0 | +29.86% |
| 8192 | 13056.880 | 9741.0 | +34.04% | 9749.0 | +33.93% |

## 结论

1. Gluon 的 Pro 和 Flash 评测链路都已在 8×H20 上完整跑通，M=1–8192 没有 OOM、launch 或 collective 错误。
2. PyTorch 2.13.0 / Triton 3.7.1 的 Pro 默认矩阵也已完整跑通；相对旧运行时只变化 -0.71% 到 +0.80%，不能解释语雀的 3.97%–6.38% 差距。
3. 三组 runner A/B 证明统计公式和独立 `torchrun` 不是主因；108 组有效消融进一步将原版 child harness 的主要额外延迟精确定位到 active loop 内逐样本调用 `torch.cuda.mem_get_info()`。它稳定增加约 27–38 µs；无查询时，临时/持久 flush tensor 生命周期的影响不超过 0.18%，Event、`empty_cache`、L1 weight 和 prewarm 也都不是主因。
4. Pro 在 M≤512 时和 PR #411 表中的 PR #383 FP8×FP8 基线基本同量级：差值为 -1.99% 到 +6.82%，M=8 仅 +0.03%。M≥1024 后差距扩大到 +11.84%–+26.87%。
5. Flash 相对 PR #383 FP8×FP8，在 M≤512 慢 +4.47%–+13.19%；M≥1024 后差距扩大到 +21.62%–+33.93%。长序列吞吐扩展性同样是主要优化点。
6. 相对 PR #411 FP8×MXFP4，Pro 全部点慢 2.96%–26.18%，Flash 全部点慢 3.30%–34.04%。Flash 的最小差距出现在 M=32，最大差距出现在 M=8192。
7. 这不是纯 kernel 实现的同格式 A/B：Gluon 使用 FP8 activation × FP8 weight，PR #411 使用 FP8 activation × MXFP4 weight；后者权重带宽更低且解码路径不同。
8. 本报告比较的是同型号 8×H20 上的本次 Gluon 实测与 PR #411 公布表格，并没有在本次隔离环境中重新构建和执行 PR #411 commit；因此版本、路由 RNG、驱动和频率状态仍可能形成系统误差。

官方对照来源：https://github.com/deepseek-ai/DeepGEMM/pull/411
