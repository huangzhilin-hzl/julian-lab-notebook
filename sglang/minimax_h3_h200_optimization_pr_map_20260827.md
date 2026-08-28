# MiniMax-H3 8×H200 优化点与 SGLang PR 映射

来源：[MiniMax-H3 on 8×H200: 1.95× Lossless, Up to 6.24× at 0.76–0.91 SSIM](https://www.lmsys.org/blog/2026-08-27-minimax-h3-h200)

GitHub 状态核对日期：2026-08-28。本文的 PR 对应关系由博客描述、PR 正文、改动文件和博客所报提交 `d90318b3e2` 的祖先关系交叉核对得到；除博客发布 PR 外，LMSYS 原文没有给出官方的一对一 PR 清单。

## 一页结论

| 目标 | 配置 | 相对 Diffusers 加速 | SSIM 相对 SGLang lossless | 结论 |
| --- | --- | ---: | ---: | --- |
| 无损默认 | SGLang dense/lossless | 1.85–1.95× | 1.0000 | 不减少 denoise step，也不稀疏 attention；收益来自更快的原生 runtime、并行和融合内核。 |
| 质量优先 | Cache-DiT conservative | 2.65–2.99× | 0.8986–0.9771 | 只做跨 step 复用，是博客最保守的近似加速档。 |
| 均衡档 | SubBlock 0.75 + Cache-DiT stride | 4.90–5.93× | 0.7713–0.9202 | 同时减少执行 step 数和每个已执行 step 的 attention 成本；T2VA 的相似度损失明显大于 FL2VA。 |
| 最快档 | SubBlock 0.80 + Cache-DiT stride | 5.06–6.24× | 0.7584–0.9144 | FL2VA 10 秒达到 6.24×；不能把该质量区间概括成所有任务都在 0.85 以上。 |

这些区间跨 T2VA/FL2VA 和 5/10 秒四个 workload 汇总，不代表任一固定 workload 的单点结果。

## 测试边界

| 项目 | 博客口径 |
| --- | --- |
| 硬件 | 8× NVIDIA H200 141 GB |
| 模型与输出 | MiniMax-H3，1344×768，24 FPS，5 秒/10 秒 |
| Denoise | 配置 50 inference steps；sigma 含两个端点，实际执行 49 次模型求值 |
| 并行 | Diffusers 使用 CP8；SGLang 使用 SP/Ulysses 8 |
| 版本 | 博客写作 `SGLang v0.5.18 (d90318b3e2)` |
| 测量日期 | 2026-08-18 |
| 延迟 | 排除服务启动、warmup、HTTP polling 和 MP4 下载；每个 task/duration 使用 3 个 prompt，报告中位数 |
| 质量 | 对同 workload、同 seed 的 SGLang lossless 输出逐帧计算 YUV420 SSIM |

## 三层优化如何叠加

```text
所有 49 个模型求值
        │
        ├─ 融合内核：降低每个实际执行 step 的固定开销
        │
        ├─ Cache-DiT：命中时复用 block-stack 输出，整个 step 的中间 blocks 不执行
        │
        └─ SubBlock：对未被 Cache-DiT 跳过的 step，减少 attention 读取的 KV blocks
```

三层作用在不同维度，因此可以组合；但算子 microbenchmark 的倍数不能相加，也不能直接等同于端到端倍数。

### 1. Dense/lossless runtime 与融合内核

MiniMax-H3 把 video/audio token 打包进同一个 DiT 序列。SGLang 的原生实现使用 Ulysses sequence parallelism，并在每个 block 的非 GEMM 热点减少中间 tensor、HBM 往返和 kernel launch。

| 算子 | Eager | SGLang kernel | 单点加速 | PR 归因 |
| --- | ---: | ---: | ---: | --- |
| indexed AdaLN scale-shift | 136.7 μs | 38.2 μs | 3.58× | H3 直接集成：[sglang#33275](https://github.com/sgl-project/sglang/pull/33275) |
| indexed gated residual | 93.2 μs | 46.6 μs | 2.00× | H3 直接集成：[sglang#33275](https://github.com/sgl-project/sglang/pull/33275)；Cache-DiT 所需的 out-of-place 首残差变体：[sglang#33827](https://github.com/sgl-project/sglang/pull/33827) |
| in-place SwiGLU | 364.5 μs | 105.2 μs | 3.46× | H3 直接集成和精确 rounding 路径：[sglang#33275](https://github.com/sgl-project/sglang/pull/33275) |
| QK RMSNorm | 334.0 μs | 76.9 μs | 4.35× | 基础 persistent fused kernel：[sglang#15835](https://github.com/sgl-project/sglang/pull/15835)；H3 调用路径：[sglang#33275](https://github.com/sgl-project/sglang/pull/33275) |
| QK RMSNorm + 3D RoPE | 1335.6 μs | 109.8 μs | 12.16× | 通用融合基础：[sglang#21440](https://github.com/sgl-project/sglang/pull/21440)；H3 的 3D RoPE、BF16 cache 和 `round_norm_before_rope=True` 路径：[sglang#33275](https://github.com/sgl-project/sglang/pull/33275) |

上述 shape 是 5 秒 T2VA、SP/Ulysses-8 后每 rank 的真实形状：4,722 rows、hidden 5,376、56 heads、head dim 128、RoPE dim 96、BF16。每个数字为 10 轮×20 次调用的 CUDA event 中位数，只是隔离的算子测试。

### 2. Cache-DiT：跨 denoise step 复用

H3 只有一个共享 `MiniMaxH3DiTModel` block stack，video/audio token 共用一次 DBCache 决策，不存在“video 命中、audio 重算”的独立缓存状态。

每个请求先执行 4 个 warmup steps；之后计算边界 blocks，比较当前 normalized residual 与上一次缓存状态：低于阈值且没有超过连续缓存上限时，中间 blocks 直接复用缓存输出，否则重算并刷新缓存。

| 档位 | `Fn` | `Bn` | warmup | RDT | 最大连续缓存 step (`MC`) |
| --- | ---: | ---: | ---: | ---: | ---: |
| conservative | 1 | 0 | 4 | 0.04 | 1 |
| stride | 1 | 0 | 4 | 0.08 | 3 |

关键 PR：

- [sglang#14234](https://github.com/sgl-project/sglang/pull/14234)：SGLang Diffusion 的通用 Cache-DiT 基础集成。
- [sglang#33275](https://github.com/sgl-project/sglang/pull/33275)：接入 MiniMax-H3 原生 block stack 和 H3 denoising pipeline。
- [sglang#33827](https://github.com/sgl-project/sglang/pull/33827)：本次 H3 数据真正依赖的关键修复。原先第一个 in-place gated residual 会改写 Cache-DiT 按引用保存的输入，使 residual diff 变成 `NaN`、0 次命中；PR 为 Cache-DiT 场景增加 out-of-place 首残差路径，同时保留第二个残差的 in-place 融合。
- [sglang#35339](https://github.com/sgl-project/sglang/pull/35339)：把 Cache-DiT 参数和 attention backend 切换做成 per-request 控制，属于部署/切换能力，不是新的算法加速来源。
- [sglang#34242](https://github.com/sgl-project/sglang/pull/34242) 与 [sglang#34848](https://github.com/sgl-project/sglang/pull/34848)：处理 breakable CUDA graph 会禁用 Cache-DiT 时的诊断和 H3 回归；属于可观测性/正确性保障。

### 3. SubBlock sparse attention：降低已执行 step 的 attention 成本

SubBlock 是无需训练的 block-sparse router：

1. 把 Q/K 序列切成 64-token blocks。
2. 每个 block 在 query/key 两侧各切为 4 个 16-token sub-block（`n_q=n_k=4`）。
3. 用 pooled Q/K 的点积和 log-sum-exp 估算每个 key block 对当前 query block/head 的未归一化 softmax mass。
4. 保留分数最高的 key blocks，把索引交给 block-sparse attention kernel；不会构造完整 attention matrix。

`sparsity` 是允许丢弃的 key-block 比例，不是保留比例：`0.75` 约保留 25%，`0.80` 约保留 20%。前 10 个 denoise steps 保持 dense，最小序列长度为 4096；短 segment、token refiner 和不支持的调用走 dense fallback。

直接 PR：

- [sglang#34148](https://github.com/sgl-project/sglang/pull/34148)：加入 SubBlock router、Triton pooling/scoring/top-k、dense fallback 和初始 SM100/FlashInfer block-sparse 路径。
- [sglang#34680](https://github.com/sgl-project/sglang/pull/34680)：把同一 64×64 routing plan 接到 SGLang CuTe-DSL block-sparse FlashAttention，新增 SM90/H200 支持；这是博客 8×H200 SubBlock 结果的硬件后端对应 PR。

### 4. SageAttention 对照项

详细结果表还有一行仅针对 FL2VA 的 `SubBlock 0.75 + SageAttention`：5 秒/10 秒分别为 2.63×/2.92×，SSIM 为 0.8827/0.9219。对应 [sglang#33703](https://github.com/sgl-project/sglang/pull/33703)，它为 MiniMax-H3 packed varlen attention 增加 SageAttention 路径。这是额外的近似 attention 对照，不是博客所强调的最快 Cache-DiT + SubBlock 主链路。

## 全部 profile 的结果范围

| Profile | 加速范围 | SSIM 范围 | 观察 |
| --- | ---: | ---: | --- |
| SGLang lossless | 1.85–1.95× | 1.0000 | 最稳妥的直接收益。 |
| Cache-DiT conservative | 2.65–2.99× | 0.8986–0.9771 | 质量优先。 |
| SubBlock 0.75 | 2.41–2.82× | 0.8006–0.9385 | 只减少 attention 工作量。 |
| SubBlock 0.75 + Cache-DiT conservative | 3.47–3.90× | 0.7936–0.9414 | 保守 cache 与 sparse 叠加。 |
| Cache-DiT stride | 3.99–4.46× | 0.8037–0.9248 | 单独使用时吞吐提升最大的近似档。 |
| SubBlock 0.75 + Cache-DiT stride | 4.90–5.93× | 0.7713–0.9202 | 博客推荐的均衡档。 |
| SubBlock 0.80 | 2.52–3.00× | 0.7858–0.9350 | 更激进的 sparse；T2VA 更敏感。 |
| SubBlock 0.80 + Cache-DiT stride | 5.06–6.24× | 0.7584–0.9144 | 最快档。 |
| SubBlock 0.75 + SageAttention（仅 FL2VA） | 2.63–2.92× | 0.8827–0.9219 | 表中额外对照，不能与四-workload 范围直接比较。 |

## PR 总表

| PR | 状态 | 与博客结果的关系 |
| --- | --- | --- |
| [sglang#33275: support MiniMax-H3](https://github.com/sgl-project/sglang/pull/33275) | Merged 2026-08-02 | 核心：原生 joint video/audio pipeline、packed DiT、并行和 H3 热点融合内核。 |
| [sglang#33827: make Cache-DiT actually cache on MiniMax-H3](https://github.com/sgl-project/sglang/pull/33827) | Merged 2026-08-13 | 核心：修复 0 命中的 aliasing 问题，使 H3 Cache-DiT 数据有效。 |
| [sglang#34148: SubBlock](https://github.com/sgl-project/sglang/pull/34148) | Merged 2026-08-11 | 核心：SubBlock 算法、router 和初始 block-sparse 后端。 |
| [sglang#34680: SubBlock on SM90](https://github.com/sgl-project/sglang/pull/34680) | Merged 2026-08-19 | 核心：H200/SM90 CuTe block-sparse kernel 接入。 |
| [sglang#33703: SageAttention packed varlen path](https://github.com/sgl-project/sglang/pull/33703) | Merged 2026-08-05 | 直接对应 FL2VA 表中的 SageAttention 对照项。 |
| [sglang#15835: JIT fused QK norm](https://github.com/sgl-project/sglang/pull/15835) | Merged 2025-12-28 | 基础内核：persistent fused QK RMSNorm。 |
| [sglang#21440: fused QK RMSNorm + RoPE](https://github.com/sgl-project/sglang/pull/21440) | Merged 2026-03-27 | 基础内核：diffusion 通用 QK RMSNorm + RoPE 融合。 |
| [sglang#14234: Cache-DiT support](https://github.com/sgl-project/sglang/pull/14234) | Merged 2025-12-05 | 基础能力：SGLang Diffusion 通用 Cache-DiT。 |
| [sglang#35339: per-request lossy accelerations](https://github.com/sgl-project/sglang/pull/35339) | Merged 2026-08-19 | 支撑：请求级 Cache-DiT/attention backend 控制。 |
| [sglang#34242](https://github.com/sgl-project/sglang/pull/34242), [sglang#34848](https://github.com/sgl-project/sglang/pull/34848) | Merged 2026-08-14 | 支撑：Cache-DiT 与 breakable CUDA graph 的诊断和 H3 回归修复。 |
| [lm-sys.github.io#412](https://github.com/lm-sys/lm-sys.github.io/pull/412) | Merged 2026-08-27 | 博客正文、图表和 demo 视频的发布 PR。 |
| [how-to-optim-algorithm-in-cuda#26](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/pull/26) | Closed, not merged | 原始 benchmark 报告、SVG 和 demo artifacts；博客把它作为完整数据与复现细节来源。 |

## 复现与归因注意事项

1. 博客写“测于 2026-08-18”，但所报提交 [`d90318b3e2`](https://github.com/sgl-project/sglang/commit/d90318b3e20fce682a74cbd2c9d6294b364b4eb9) 的提交时间是 2026-08-22；SM90 SubBlock PR #34680 也在 2026-08-19 合并。因而 `d90318b3e2` 更适合作为发布时固定的代码快照，不应解释成 8 月 18 日跑测时已经存在的公开 main SHA。
2. GitHub annotated tag [`v0.5.18`](https://github.com/sgl-project/sglang/releases/tag/v0.5.18) 指向 `71de97b264b0...`，不是 `d90318b3e2`。需要复现实验时应直接 pin 博客给出的完整 commit，而不是只 checkout tag。
3. 博客声称 cookbook 保存每种 profile 的精确 flags，但截至 2026-08-28，当前 MiniMax-H3 cookbook 已演进，页面中不再出现本文的 SubBlock 0.75/0.80 和 Cache-DiT RDT 0.04/0.08 历史 profile。历史配置应以博客正文和 benchmark PR #26 为准。
4. 本次数据只覆盖 fused kernels、Cache-DiT 和 SubBlock 三类主开关；量化、progressive resolution 等其他 lossy 路径不在测试矩阵中，不能从 6.24× 外推其组合收益。
5. SSIM 衡量的是与同 seed lossless 轨迹的相似度，不是绝对视觉质量，也没有覆盖联合生成音频的质量。
