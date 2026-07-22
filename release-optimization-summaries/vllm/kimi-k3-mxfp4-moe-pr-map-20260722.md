# Kimi K3 MXFP4 MoE 技术点与公开 PR 对应关系

- 更新时间：2026-07-22
- 原文：[A Preview of Production-Scale Kimi K3 Support on vLLM](https://vllm.ai/blog/2026-07-22-kimi-k3-preview)
- 聚焦范围：MXFP4 MoE、SiTU、TRTLLM-Gen 大规模 launch grid 分块，以及 AMD FlyDSL A16W4/A8W4

## 结论

截图中的实现并不对应一个已经公开的单一 vLLM PR，而是横跨 vLLM、FlashInfer 和 ROCm/AITER 三层。

截至 2026-07-22：

1. 尚未找到包含 Kimi K3 专属模型接入、`SiTU` 参数映射和 MXFP4 TRTLLM-Gen 分块逻辑的公开 vLLM PR。
2. 公开 PR 已覆盖 SiTU 内核、TRTLLM MoE 大批量分块，以及 AMD FlyDSL A16W4/A8W4 的主要基础能力。
3. vLLM 博客所说的 K3 专属 glue code、16-GPU DP16+EP16 验证代码和最终 backend selection，仍应视为 release branch 内容，等待随模型权重公开。

因此，下面的 PR 需要区分为“直接相关内核”“机制前置实现”和“尚未公开的 K3 集成”，不能把其中任意一个单独等同于完整 K3 MXFP4 MoE 支持。

## 公开 PR 对应表

| 原文技术点 | 公开 PR | 2026-07-22 状态 | 对应程度 | 边界 |
| --- | --- | --- | --- | --- |
| SiTU 激活公式与参数传播 | [FlashInfer #4009](https://github.com/flashinfer-ai/flashinfer/pull/4009) | Open | 直接相关内核 | 实现的是 Blackwell CuTe-DSL NVFP4 fused MoE 路径，不是完整的 vLLM MXFP4 TRTLLM-Gen glue |
| 大规模 token-by-top-k launch grid 安全分块 | [vLLM #43599](https://github.com/vllm-project/vllm/pull/43599) | Merged | 机制直接对应 | 已合入实现只对 TRTLLM NVFP4 MoE 启用；K3 所需 MXFP4 扩展尚未公开 |
| AMD FlyDSL A16W4：BF16 activation × MXFP4 weight | [ROCm/AITER #3254](https://github.com/ROCm/aiter/pull/3254) | Open | 直接相关内核 | 提供两阶段 A16W4 MoE GEMM、Python API 和正确性测试，不包含 K3 SiTU glue |
| AMD FlyDSL A8W4：FP8 activation × MXFP4 weight | [ROCm/AITER #2951](https://github.com/ROCm/aiter/pull/2951) | Merged | 直接相关基础能力 | 以 DeepSeek-V4 为接入对象，但提供的是可复用 FlyDSL A8W4 MoE 路径 |
| AMD gfx942 的 A16W4/A8W4 MXFP4 扩展 | [ROCm/AITER #3926](https://github.com/ROCm/aiter/pull/3926) | Open, Draft | 平台扩展 | 同时覆盖 A16W4/A8W4，但目标是 gfx942，不能代替 K3 release-branch 的完整 AMD 验证 |

## NVIDIA 路径

### SiTU

[FlashInfer #4009](https://github.com/flashinfer-ai/flashinfer/pull/4009) 是目前公开代码中与文章 SiTU 描述最直接对应的 PR。它增加：

- `situ_beta` 参数；
- `up * beta * tanh(gate / beta) * sigmoid(gate)` 计算；
- 可选的 linear/up 分支平滑截断；
- functional API、wrapper、autotuner、compiled-kernel cache 和 trace 参数传播；
- 与普通 SwiGLU 的数值对照和错误配置校验。

但这个 PR 修改的是 FlashInfer Blackwell CuTe-DSL NVFP4 fused MoE。文章描述的是 vLLM 将 K3 的 SiTU 参数映射到优化后的 **MXFP4 TRTLLM-Gen** expert path，因此仍缺少至少两层公开接入：

1. K3 模型配置将 SiTU 参数传入 vLLM fused-MoE backend；
2. vLLM/FlashInfer 的 MXFP4 TRTLLM-Gen runner 消费这些参数。

在这两层公开之前，不能把 FlashInfer #4009 单独称为完整 K3 MXFP4 MoE PR。

### 大规模 launch grid 分块

[vLLM #43599](https://github.com/vllm-project/vllm/pull/43599) 与博客中“large token-by-top-k launch grids by safely chunking the workload”的表述高度一致。该 PR 处理两类大 batch 问题：

- TRTLLM fused MoE 的 CUDA `grid.y` 超过约 64K 的硬限制；
- 极大 token 数下的 kernel 非法内存访问。

实现会根据 `top_k`、`num_experts` 和 grid 上限计算最大安全 token 数，并对 MoE 输入分块执行。不过，公开 PR 明确只给 `TrtLlmNvFp4ExpertsModular` 启用该逻辑。K3 的 MXFP4 TRTLLM-Gen 路径大概率复用了或扩展了这一机制，但截至本笔记日期，相应 MXFP4 PR 尚未公开。

## AMD FlyDSL 路径

### A16W4

[ROCm/AITER #3254](https://github.com/ROCm/aiter/pull/3254) 的定义与文章中的 A16W4 完全一致：BF16 activation × MXFP4 weight。它提供两阶段 FlyDSL MoE GEMM：

- stage 1：gate/up projection 与激活；
- stage 2：down projection 与累加；
- split-K、weight/scale preshuffle、Python dispatch 和正确性测试。

这是 AMD A16W4 最重要的公开内核 PR，但当前仍为 Open，并且激活仍以现有 SiLU/SwiGLU 路径为主，没有公开的 K3 `SiTU` 参数实现。

### A8W4

[ROCm/AITER #2951](https://github.com/ROCm/aiter/pull/2951) 已将 FlyDSL A8W4 MoE 接入 AITER，包含：

- FP8 activation × MXFP4 weight 两阶段 kernel；
- fused activation/quantization；
- MXFP4 scale/weight shuffle；
- AOT 和 tuned configuration 接入。

[ROCm/AITER #3926](https://github.com/ROCm/aiter/pull/3926) 则把 A16W4/A8W4 MXFP4 扩展到 gfx942，通过 FP4 dequant 加 legacy MFMA 实现；它是平台覆盖扩展，不是 K3 专属 PR。

### 尚缺的 SiTU 层

截至 2026-07-22，公开 AITER PR 和主分支中没有找到 K3 `SiTU` 或 `situ_beta` 的明确实现。现有 `silu`、`swiglu`、`swiglu_limit` 不能直接等同于 SiTU：SiTU 包含由 beta 控制的 tanh 变换，数学合同不同。

因此，博客中的“AMD FlyDSL A16W4/A8W4 fused operators and a SiTU activation implementation”应理解为：公开仓库已经具备 A16W4/A8W4 基础 kernel，但 K3 专属 SiTU glue/实现仍在 release branch 或待公开 PR 中。

## 容易误认但不直接对应的 PR

| PR | 为什么不是截图中的直接实现 |
| --- | --- |
| [vLLM #44400](https://github.com/vllm-project/vllm/pull/44400) | 面向 Kimi K2.5 INT4/W4A16 Compressed Tensors FlyDSL 路径，不是 K3 MXFP4 + SiTU |
| [ROCm/AITER #2863](https://github.com/ROCm/aiter/pull/2863) | 面向 Kimi K2 的 packed INT4 A16WI4 MoE，不是 K3 MXFP4 A16W4 |
| [ROCm/AITER #3767](https://github.com/ROCm/aiter/pull/3767) | 实现带 limit 的 clamped SwiGLU；公式不同，不能当作 SiTU |

## 后续核对重点

模型权重和 day-0 支持公开后，应优先确认：

1. vLLM Kimi K3 model/config PR 是否新增 SiTU beta 参数；
2. `TrtLlmMxfp4Experts*` 是否获得 SiTU 和大 batch chunking；
3. FlashInfer TRTLLM-Gen MXFP4 runner 是否新增对应 activation contract；
4. AITER FlyDSL 是否公开 K3 SiTU stage-1 epilogue；
5. 16-GPU DP16+EP16 验证实际使用的 GPU、backend、PR SHA 和启动参数。

## 来源

- [vLLM Kimi K3 preview blog](https://vllm.ai/blog/2026-07-22-kimi-k3-preview)
- [vLLM blog source PR #276](https://github.com/vllm-project/vllm-project.github.io/pull/276)
- [FlashInfer #4009](https://github.com/flashinfer-ai/flashinfer/pull/4009)
- [vLLM #43599](https://github.com/vllm-project/vllm/pull/43599)
- [ROCm/AITER #3254](https://github.com/ROCm/aiter/pull/3254)
- [ROCm/AITER #2951](https://github.com/ROCm/aiter/pull/2951)
- [ROCm/AITER #3926](https://github.com/ROCm/aiter/pull/3926)
