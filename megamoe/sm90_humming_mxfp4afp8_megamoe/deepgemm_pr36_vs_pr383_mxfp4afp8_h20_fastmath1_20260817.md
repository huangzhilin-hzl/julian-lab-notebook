# DeepGEMM PR #36 与 PR #383、MXFP4A-FP8 MegaMoE 对比（H20，fast-math=1）

## 结论

在一台实际无其他 GPU 计算进程的 8× NVIDIA H20 环境上，Flash 与 Pro 共 22/22 个 case 完成。最终数据只采用干净环境的全量复跑；此前有隐性显存占用的测试轮不计入结果。

- 对 PR #383：Flash 仅 M=512 快 0.8%，其余 10 个点慢 0.1%～78.8%；Pro 11 个点全部慢 0.1%～56.9%。按 11 个 M 等权几何平均，PR #36 分别慢 26.5% 和 26.5%。
- 对 routed-only MXFP4A-FP8 MegaMoE：Flash 3/11 个点更快（M=512、4096、8192），几何平均慢 4.3%；Pro 6/11 个点更快（M=16～256、8192），几何平均慢 2.9%。
- 大 M 时 Flash 三者接近：M=4096 时 PR #36 比 PR #383 慢 2.6%、比 MXFP4A-FP8 快 1.7%；M=8192 时分别慢 1.4%、快 4.0%。
- Pro 在 M=512～4096 回退更明显：相对 PR #383 慢 6.8%～22.8%，相对 MXFP4A-FP8 慢 3.5%～22.8%；M=8192 时相对 MXFP4A-FP8 快 1.0%。

PR #36 路径是 FP8 activation × FP8 weight，MXFP4A-FP8 路径是 FP8 activation × MXFP4 weight，数值格式不同。本次 PR #36 脚本不包含 fused shared expert，因此 MXFP4A-FP8 对照采用 `shared=0` routed-only 数据；不能把 `shared=1` 数据直接并入等 workload 比较。

## 测试对象

| 项目 | 内容 |
|---|---|
| PR #36 | [sgl-project/DeepGEMM PR #36](https://github.com/sgl-project/DeepGEMM/pull/36) |
| PR #36 提交 | `3f9268b5c15d4b939957051a1b5d22d2ef3dcf4e` |
| Benchmark | `bench_deepgemm_pr36_megamoe_sm90.py` |
| 对照 1 | [DeepGEMM PR #383](https://github.com/deepseek-ai/DeepGEMM/pull/383) 的 H20 MegaMoE 数据 |
| 对照 2 | `molou/support_sm90_humming_mxfp4afp8_megamoe_opt`，routed-only `shared=0` |
| GPU | NVIDIA H20 × 8 |
| 软件栈 | Python 3.12.3，PyTorch 2.11.0+cu130，CUDA 13.0，driver 550.127.08 |
| 日期 | 2026-08-17 |

构建产物 wheel SHA-256：`f2bf15464ce3018d558c0e06da7722298e8d381887144a8e418c0d30b4ab16ee`。Benchmark 脚本 SHA-256：`967b7a5b90c262fa1094f5f378d54d7fb364aa417a96d19c98a816481b373b66`。

## 测试配置

| 参数 | 值 |
|---|---|
| ranks | 8 |
| workload | Flash：H=4096、I=2048、E=256、top-k=6；Pro：H=7168、I=3072、E=384、top-k=6 |
| M | 8、16、32、64、128、256、512、1024、2048、4096、8192 |
| shared experts | 0 |
| fast-math | 1 |
| seed | 101 |
| observations | 3 |
| warmups | 5 / observation |
| samples | 20 / observation |
| flush-L2 | 请求 8,000,000,000 bytes |
| 计时范围 | `deep_gemm.fp8_mega_moe(...)`；输入拷贝、权重预处理、weight transform 和 buffer allocation 不计时 |
| 主指标 | 每个 observation 先取各 rank CUDA-event median 的最大值，再对 3 个 observation 取中位数 |

`--match-cap-to-m` 请求 cap=M，但 PR #36 内部对称内存容量会对齐：M≤256 为 384，随后依次为 768、1152、2304、4224、8448。

脚本只做输出 shape、dtype、finite 与累计路由计数校验，`numerical_reference=false`；本报告是性能对比，不替代独立精度测试。

## Flash

| M | PR #36 (µs) | PR #383 (µs) | PR36 vs PR383 | MXFP4A-FP8 shared=0 (µs) | PR36 vs MXFP4A-FP8 |
|---:|---:|---:|---:|---:|---:|
| 8 | 473.1 | 273.1 | +73.2% | 444.0 | +6.5% |
| 16 | 514.2 | 304.4 | +68.9% | 495.6 | +3.8% |
| 32 | 540.1 | 302.0 | +78.8% | 496.0 | +8.9% |
| 64 | 541.0 | 340.7 | +58.8% | 499.7 | +8.3% |
| 128 | 548.4 | 414.4 | +32.3% | 503.9 | +8.8% |
| 256 | 570.1 | 569.5 | +0.1% | 511.8 | +11.4% |
| 512 | 914.7 | 922.0 | -0.8% | 924.3 | -1.0% |
| 1024 | 1679.5 | 1516.6 | +10.7% | 1590.0 | +5.6% |
| 2048 | 2880.6 | 2735.1 | +5.3% | 2831.0 | +1.8% |
| 4096 | 5248.8 | 5116.0 | +2.6% | 5339.0 | -1.7% |
| 8192 | 9884.6 | 9749.0 | +1.4% | 10299.0 | -4.0% |

## Pro

| M | PR #36 (µs) | PR #383 (µs) | PR36 vs PR383 | MXFP4A-FP8 shared=0 (µs) | PR36 vs MXFP4A-FP8 |
|---:|---:|---:|---:|---:|---:|
| 8 | 1204.7 | 768.0 | +56.9% | 1202.5 | +0.2% |
| 16 | 1453.9 | 950.3 | +53.0% | 1564.0 | -7.0% |
| 32 | 1579.6 | 1026.3 | +53.9% | 1622.0 | -2.6% |
| 64 | 1603.1 | 1059.9 | +51.2% | 1637.0 | -2.1% |
| 128 | 1628.9 | 1201.0 | +35.6% | 1651.5 | -1.4% |
| 256 | 1641.7 | 1639.9 | +0.1% | 1692.0 | -3.0% |
| 512 | 3056.7 | 2599.0 | +17.6% | 2579.0 | +18.5% |
| 1024 | 4955.5 | 4036.0 | +22.8% | 4036.0 | +22.8% |
| 2048 | 7672.8 | 6986.0 | +9.8% | 7114.0 | +7.9% |
| 4096 | 13807.6 | 12932.0 | +6.8% | 13346.0 | +3.5% |
| 8192 | 25656.0 | 24777.0 | +3.5% | 25905.0 | -1.0% |

表中 delta 为 `PR36 / 对照 - 1`；负值表示 PR #36 更快。

## 有效性与脱敏

- 正式轮 22/22 个 case 成功；Flash observation 最大相对跨度 7.6%，Pro 为 2.8%。
- 正式测试前后均未检测到其他 GPU 计算进程。
- 最终报告和机器可读数据不包含节点名、Pod 名、网络地址、私有镜像或远端工作目录。
- CUDA-event kernel timing 与 PR #383/MXFP4A-FP8 的 persistent-kernel 主指标尽量对齐，但实现、数值格式与计时工具并非完全相同。

机器可读数据：

- [comparison.csv](results/20260817-h20-deepgemm-pr36-fastmath1/comparison.csv)
- [summary.json](results/20260817-h20-deepgemm-pr36-fastmath1/summary.json)
- [clean-flash.log](results/20260817-h20-deepgemm-pr36-fastmath1/clean-flash.log)
- [clean-pro.log](results/20260817-h20-deepgemm-pr36-fastmath1/clean-pro.log)
