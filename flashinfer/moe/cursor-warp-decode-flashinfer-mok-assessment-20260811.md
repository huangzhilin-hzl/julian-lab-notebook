# Cursor Warp Decode、FlashInfer 与 Mixture-of-Kittens 实现评估

更新日期：2026-08-11

## 结论

截至本次核对：

1. Cursor 公开的 Warp Decode 是面向 Blackwell 小 batch 自回归解码的 output-centric MoE 算法。其关键不是简单融合已有 grouped GEMM，而是把工作所有权从 expert/token batch 翻转为“每个 warp 负责一个输出值”。
2. FlashInfer 当前 `main` 没有 Cursor 文中两个同名 kernel，也没有完全相同的 B200/SM100 MXFP8 Warp Decode 后端。
3. FlashInfer 已合入的 SM12x `MoEDirectMicroKernel` 属于最接近的算法家族：直接消费 `topk_ids/topk_weights`、不做 routing pre-pass、面向 tiny decode batch。但它是 SM12x NVFP4/W4A16 路线，使用单个 direct-micro kernel 和自己的 shared-memory/barrier/scale 合同，不能视作 Cursor 两-kernel MXFP8 实现。
4. FlashInfer PR [#4310](https://github.com/flashinfer-ai/flashinfer/pull/4310) 是当前最接近 Cursor 两-kernel 骨架的公开贡献：SM120、BF16、`M=1..8`、预路由输入、gate/up+SwiGLU 与 down+top-k reduction 两个 kernel；但截至 2026-08-11 仍为 open，且目标架构、精度和权重格式均不等同于 Cursor 的 B200 MXFP8 路线。
5. Cursor 官方开源仓库 `cursor/mixture-of-kittens` 也没有 Warp Decode。它是 NVL72 训练 megakernel，仍采用 expert-centric padding、dispatch、grouped GEMM 和 combine，只是把计算与通信融合进单个 persistent/mega kernel。
6. 当前最明确的第三方公开复现是 TokenSpeed PR [#403](https://github.com/lightseekorg/tokenspeed/pull/403)：它明确引用 Cursor 文章，在 AMD gfx950 上实现 `M<=16` 的两阶段 direct token/top-k Warp Decode；后续性能、正确性和格式扩展形成了多条 PR 链。TensorRT-LLM 有相同/相邻数据格式和阶段融合，但未发现声称或实现 Cursor exact output-centric Warp Decode 的 PR。

因此，应将当前状态表述为：**FlashInfer 已有相邻的 direct-routed 小-M 技术，但没有 Cursor Warp Decode 的 exact 对应实现；Mixture-of-Kittens 也不是其开源版本。**

## 1. Cursor Warp Decode 的判定合同

Cursor 文章公开的两个内部 kernel 名称为：

```text
moe_gate_up_3d_batched
moe_down_3d_batched
```

文章描述的执行模型如下：

```text
BF16 activation + top-k expert ids
  -> gate/up kernel
       每个 CTA 8 warps
       每个 warp 负责一个 token/expert 的一个 intermediate neuron
       流式读取 MXFP8 gate/up weight row
       FP32 register accumulation
       warp reduction + SiLU(gate) * up
  -> down kernel
       每个 warp 负责一个 token 的一个 output dimension
       在 warp 内遍历全部 top-k experts
       routing weight 直接折叠进 FP32 accumulator
       __shfl_xor_sync butterfly reduction
  -> BF16 output
```

它与传统 expert-centric MoE 的本质差异是：

- 不形成 per-expert token batch；
- 不做 expert padding；
- 不做 activation gather/scatter；
- 不物化每个 expert 的完整 down 输出；
- 不运行独立 top-k combine；
- warp 从开始到结束拥有唯一输出 scalar，无需跨 warp 协作。

官方描述见 [Better MoE model inference with warp decode](https://cursor.com/blog/warp-decode)。文章没有给出 kernel 源码或可构建仓库，只给出了算法、内部 kernel 名称和内部 B200 结果。

## 2. FlashInfer 当前实现状态

### 2.1 核对基线

| 项目 | 值 |
| --- | --- |
| 上游仓库 | `flashinfer-ai/flashinfer` |
| 默认分支 | `main` |
| 核对 SHA | [`42ea835`](https://github.com/flashinfer-ai/flashinfer/commit/42ea835cfde6aadbbee4d7c12187873cade5aaba) |
| 核对日期 | 2026-08-11 |
| 精确符号搜索 | 未找到 `moe_gate_up_3d_batched`、`moe_down_3d_batched`、`warp_decode` |
| 验证边界 | 静态源码与公开 PR 核对；本次未运行 B200/SM120 benchmark |

### 2.2 已合入的最近实现：SM12x DirectMicro

当前 `main` 的 [`moe_direct_micro_kernel.py:1`](https://github.com/flashinfer-ai/flashinfer/blob/42ea835cfde6aadbbee4d7c12187873cade5aaba/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_direct_micro_kernel.py#L1-L7) 明确说明：

```python
"""MoEDirectMicroKernel: direct-routed NVFP4 MoE decode kernel for SM12x.

Unlike MoEMicroKernel (Triton route pre-pass, expert-major packed A), this
kernel consumes raw per-token topk_ids/topk_weights with no routing
pre-pass and computes both GEMMs as software fp4 dot products on CUDA
cores, targeting tiny decode batches.
"""
```

调度器仅在很小的 routed-row 范围内选择它，主要约束包括：

```text
routed_rows = num_tokens * top_k
routed_rows < 32
intermediate_size <= 512
MoEDirectMicroKernel.is_supported(...) == true
```

对应选择逻辑见 [`moe_dispatch.py:1469`](https://github.com/flashinfer-ai/flashinfer/blob/42ea835cfde6aadbbee4d7c12187873cade5aaba/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py#L1469-L1520)。

它与 Cursor Warp Decode 的共同点：

- 面向 tiny decode batch；
- 直接消费预计算路由，不进行 expert-major routing pre-pass；
- CUDA core 软件 dot product，而不是为大 M 设计的 grouped Tensor Core GEMM；
- 使用 warp reduction；
- 通过避免排序、padding 和 grouped-GEMM 启动开销服务低 M。

关键差异：

| 维度 | Cursor Warp Decode | FlashInfer DirectMicro |
| --- | --- | --- |
| 架构 | B200/SM100 | SM12x |
| 主要权重格式 | MXFP8 | NVFP4/W4A16 |
| kernel 组织 | gate/up 与 down 两个 kernel | 单个 direct-micro kernel 内完成两层计算 |
| CTA 规模 | 8 warps | 16 warps，见 [`moe_direct_micro_kernel.py:64`](https://github.com/flashinfer-ai/flashinfer/blob/42ea835cfde6aadbbee4d7c12187873cade5aaba/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_direct_micro_kernel.py#L64-L70) |
| 工作所有权 | 每 warp 固定拥有一个 intermediate/output scalar | warp 处理一个或多个 FC1 row/chunk，并通过 CTA 内 scratch/barrier 协同完成后续阶段 |
| 中间激活 | BF16，不做中间 activation quantization | 具有 FP4/FP8 scale、down-input scale 及动态量化合同 |

结论：DirectMicro 是可复用的低-M direct-routing参考，但不是 Cursor exact kernel。

### 2.3 Open PR #4310

PR [#4310](https://github.com/flashinfer-ai/flashinfer/pull/4310) 增加 `sm120_direct_fused_moe`：

```text
precomputed expert ids + routing weights
  -> GateUpSwiGLUKernel
  -> DownFusedTopKKernel
  -> rank-local BF16 partial output
```

它已经覆盖了 Cursor 方案最重要的两点：

- 取消 token sorting/gather；
- 在 down kernel 内完成 FP32 top-k weighted reduction。

但适用边界是：

- SM120，而非 B200/SM100；
- BF16 input/weight/output，而非 MXFP8 weight streaming；
- `M=1..8`、`topk<=8`、`H<=8192`、`I<=1024`；
- EP 场景返回 rank-local partial，跨 rank collective 由调用方负责；
- 截至核对日期仍未合入 `main`。

PR 中的 SM120 测试和性能数字是作者报告，本次没有独立复现。因此它能证明公开代码和测试资产存在，不能作为 B200 生产性能证明。

## 3. Mixture-of-Kittens 是否包含 Warp Decode

### 3.1 核对基线

| 项目 | 值 |
| --- | --- |
| 仓库 | [`cursor/mixture-of-kittens`](https://github.com/cursor/mixture-of-kittens) |
| 默认分支 | `main` |
| 核对 SHA | [`6438bf4`](https://github.com/cursor/mixture-of-kittens/commit/6438bf48f88094d305972fbe0fa6deba0f7d4d1a) |
| 精确符号搜索 | `moe_gate_up_3d_batched`、`moe_down_3d_batched`、`warp_decode` 均为 0 命中 |
| 公开分支 | 仅 `main` |
| PR/commit 搜索 | 未发现 Warp Decode 相关提交或 PR |

### 3.2 为什么不是同一实现

MoK README 将其定义为 NVL72 的 deterministic MoE **training megakernel**，覆盖 forward 和 backward，见 [`README.md:1`](https://github.com/cursor/mixture-of-kittens/blob/6438bf48f88094d305972fbe0fa6deba0f7d4d1a/README.md#L1-L5)。

它保留了 Warp Decode 明确要消除的 expert-centric 数据组织：

1. Scheduler 将每个 expert 的 token 数 pad 到 256：[`scheduler.cuh:13`](https://github.com/cursor/mixture-of-kittens/blob/6438bf48f88094d305972fbe0fa6deba0f7d4d1a/csrc/scheduler.cuh#L13-L17)、[`scheduler.cuh:65`](https://github.com/cursor/mixture-of-kittens/blob/6438bf48f88094d305972fbe0fa6deba0f7d4d1a/csrc/scheduler.cuh#L65-L78)。
2. Functional API 要求 `minibatch_size` 是 256 的倍数：[`functional.py:388`](https://github.com/cursor/mixture-of-kittens/blob/6438bf48f88094d305972fbe0fa6deba0f7d4d1a/mok/functional.py#L388-L393)。README 建议的生产训练范围是 2048 到 16384。
3. Forward megakernel 内部仍把 routed gate、routed up、SwiGLU、routed down 分成 grouped-GEMM/tile task：[`mok_megakernel.cuh:1561`](https://github.com/cursor/mixture-of-kittens/blob/6438bf48f88094d305972fbe0fa6deba0f7d4d1a/csrc/mok_megakernel.cuh#L1561-L1576)、[`mok_megakernel.cuh:1771`](https://github.com/cursor/mixture-of-kittens/blob/6438bf48f88094d305972fbe0fa6deba0f7d4d1a/csrc/mok_megakernel.cuh#L1771-L1813)。
4. Megakernel 中仍存在显式 dispatch、combine、通信 SM、跨任务 barrier 和中间 activation buffer；“单 kernel”只是把传统执行图放进同一 persistent launch，不等于 output-centric warp independence。

### 3.3 相似点为何不足以判定同源

MoK 与 Warp Decode 都包含以下元素：

- Blackwell；
- MXFP8；
- 8-warps compute CTA；
- gate/up、SwiGLU、down；
- MoE routing weights。

这些只是硬件、数值格式和 MoE 数学上的交集。判定 Warp Decode 必须检查工作所有权、是否形成 per-expert batch、是否 padding、down 是否在 warp 内跨 top-k 累积，而不能只看“8 warps + MXFP8 + fused MoE”。

## 4. 相关 PR 与实现演进

以下状态均核对于 2026-08-11。这里把 PR 分成三类：

- **直接复现**：明确引用 Cursor Warp Decode，或者继续维护同一 direct token/top-k 两阶段实现；
- **算法相邻**：同样面向小 M、直接路由或融合 finalize，但工作所有权/架构/格式不完全相同；
- **基础设施相邻**：提供相同量化格式、grouped-MoE、MegaMoE 或通信能力，不能据此称为 Warp Decode。

### 4.1 TokenSpeed：明确的 Cursor-inspired Warp Decode PR 链

TokenSpeed 是目前找到的、最明确写明“as described by Cursor Warp Decode”的公开实现。需要注意：它运行在 AMD gfx950/CDNA4，硬件执行单元是 64-lane wavefront；`warp decode` 在这里是算法名称，不代表 NVIDIA 32-lane warp。

| PR | 状态 | 分类 | 与 Warp Decode 的关系 |
| --- | --- | --- | --- |
| [TokenSpeed #314](https://github.com/lightseekorg/tokenspeed/pull/314) | Merged | 前置基线 | 增加 GPT-OSS Gluon MoE，decode 仍采用普通 pipelined 路径，为后续 Warp Decode 提供权重预处理和 Gluon 基础。 |
| [TokenSpeed #363](https://github.com/lightseekorg/tokenspeed/pull/363) | Merged | 前置优化 | `M<=16` 的单-kernel routing fast path；只优化路由，不是 Warp Decode MLP 本体。 |
| [TokenSpeed #403](https://github.com/lightseekorg/tokenspeed/pull/403) | Merged | **直接复现起点** | 明确引用 Cursor；gfx950、FP8 activation × MXFP4 weight、`M<=16`。Stage 1 融合 top-k、gate/up、SwiGLU，Stage 2 在 kernel 内遍历 top-k 并直接写最终输出。 |
| [TokenSpeed #423](https://github.com/lightseekorg/tokenspeed/pull/423) | Merged | 直接后续 | Stage 1 改为 4-wave cooperative LDS pipeline；Stage 2 引入按 M 调优的 split-K/reduce，修正 gate/up interleave、scale swizzle 和 K-tail。 |
| [TokenSpeed #470](https://github.com/lightseekorg/tokenspeed/pull/470) | Merged | fallback/crossover | 优化 medium-batch decode，用于 Warp Decode 小-M 区间以外的性能衔接。 |
| [TokenSpeed #494](https://github.com/lightseekorg/tokenspeed/pull/494) | Closed, unmerged | 实验 | medium-batch decode WIP；不能作为已落地主线能力。 |
| [TokenSpeed #495](https://github.com/lightseekorg/tokenspeed/pull/495) | Merged | 正确性后续 | 修复 preshuffled W2 padded-N 下的越界地址形成，并同步 padding bias。 |
| [TokenSpeed #605](https://github.com/lightseekorg/tokenspeed/pull/605) | Merged | 正确性后续 | 修复小 K pipeline 预取越界和 top-k 索引恢复错误，并恢复此前因 import 问题被跳过的 Warp Decode 单测。 |
| [TokenSpeed #626](https://github.com/lightseekorg/tokenspeed/pull/626) | Merged | BF16 sibling | gfx950 BF16 两阶段 MoE；`num_tokens<=8` 自动走 pure-Gluon warp-reduce GEMV decode，作者报告 M=1–2 约 2.4x。不是 #403 的 FP8×MXFP4 格式。 |
| [TokenSpeed #670](https://github.com/lightseekorg/tokenspeed/pull/670) | Merged | A4W4 扩展 | 完整 MXFP4 activation × MXFP4 weight 包；`M<=8` 使用 direct/route-owned MFMA decode，`M>=9` 转 prefill/grouped 路线。 |
| [TokenSpeed #831](https://github.com/lightseekorg/tokenspeed/pull/831) | Closed, unmerged | 调优实验 | Kimi K3 Warp Decode re-tile，覆盖 M=2/4/8/16；作者报告 M=8/16 延迟下降，但没有合入。 |
| [TokenSpeed #878](https://github.com/lightseekorg/tokenspeed/pull/878) | Merged | Kimi K3 相邻路线 | Kimi K3 小 token 优化：M<=4 使用 warp-GEMV，M>=5 使用 grouped MFMA，并融合输入投影/路由；不是 GPT-OSS #403 的同一数学/格式合同。 |
| [TokenSpeed #1020](https://github.com/lightseekorg/tokenspeed/pull/1020) | Open, WIP | 代码重构 | 拆分 gfx950 MXFP4 decode kernels；当前不能视为稳定落地。 |
| [TokenSpeed #1021](https://github.com/lightseekorg/tokenspeed/pull/1021) | Merged | 清理 | 删除未引用的重复 `gluon_mxfp4_moe_decode` 和 legacy BF16-activation warp-GEMV。**没有删除生产路径**；生产仍调用 `_maybe_precomputed_mxfp4_direct_mfma_decode`。 |
| [TokenSpeed #1024](https://github.com/lightseekorg/tokenspeed/pull/1024) | Open, draft | 代码重构 | 将大文件拆成子包，其中 `fused/warp_decode.py` 承载 `M<=4` FP8×MXFP4 两阶段路径；PR 声明主要是 verbatim code move。 |

因此，TokenSpeed 的关键主线是：

```text
#314/#363 基础与小-M routing
  -> #403 Cursor-inspired Warp Decode
  -> #423 pipeline/split-K 优化
  -> #495/#605 正确性修复
  -> #626 BF16 sibling
  -> #670 A4W4 direct-decode 扩展
  -> #1021 清理重复旧入口
  -> #1024 当前模块化重构（尚未合入）
```

### 4.2 FlashInfer：DirectMicro、两-kernel 提案与 MegaMoE

| PR/Issue | 状态 | 分类 | 关系与边界 |
| --- | --- | --- | --- |
| [FlashInfer PR #3271](https://github.com/flashinfer-ai/flashinfer/pull/3271) | Merged | DirectMicro 格式扩展 | 增加 SM120 W4A16 b12x kernels，并接入 W4A16 direct-micro；EP remap 不支持时回退 static W4A16。 |
| [FlashInfer PR #4285](https://github.com/flashinfer-ai/flashinfer/pull/4285) | Merged | **当前 DirectMicro 主线** | 同步 SM12x NVFP4 kernels 到 b12x HEAD；让 `MoEDirectMicroKernel` 真正进入 dispatch，直接读取 top-k IDs，跳过 routing pre-pass。batch 1–2 是其主要收益区间。上游来源为 [b12x `f9be272`](https://github.com/local-inference-lab/b12x/commit/f9be2724953a5b412d19c20482aeb0a64fbd5d2a)。 |
| [FlashInfer PR #4310](https://github.com/flashinfer-ai/flashinfer/pull/4310) | Open | **最接近两-kernel 骨架** | SM120 BF16、`M=1..8`；direct gate/up+SwiGLU 与 down+FP32 top-k reduction 两个 kernel。不是 SM100/MXFP8，且尚未合入。 |
| [FlashInfer Issue #3110](https://github.com/flashinfer-ai/flashinfer/issues/3110) | Open | 模块化需求 | GB200 BF16 expert compute 与 NVLink all-to-all 解耦需求；是 #4310 关联背景，不是 Warp Decode PR。 |
| [FlashInfer PR #4113](https://github.com/flashinfer-ai/flashinfer/pull/4113) | Merged | MegaMoE 相邻路线 | SM90 pull-style FP8 MegaMoE：NVSHMEM dispatch+FC1+SwiGLU+FC2+combine 单 launch。它是 expert-centric persistent megakernel，不是 output-centric Warp Decode。 |
| [FlashInfer PR #4120](https://github.com/flashinfer-ai/flashinfer/pull/4120) | Closed, unmerged draft | MegaMoE 相邻路线 | Blackwell BF16 MegaMoE 集成；未合入，并保留 dispatch/compute/combine megakernel 结构。 |

### 4.3 TensorRT-LLM：相同格式与阶段融合，但不是 exact Warp Decode

对 TensorRT-LLM 当前公开代码和 PR 做了 `warp_decode`、`moe_gate_up_3d_batched`、`moe_down_3d_batched` 搜索，均未找到 exact 对应实现。下面这些 PR 与数据格式、低延迟或流水融合相关，但应标为相邻能力：

| PR | 状态 | 能力 | 为什么不是 Cursor Warp Decode |
| --- | --- | --- | --- |
| [TensorRT-LLM #5027](https://github.com/NVIDIA/TensorRT-LLM/pull/5027) | Merged | 开源 `low_latency_gemm`、`moe-gemm`、`fp4_gemm` 等内部 CUTLASS kernels | 提供 kernel primitives/传统 MoE GEMM；没有 direct token/top-k output-owned 两阶段合同。 |
| [TensorRT-LLM #4750](https://github.com/NVIDIA/TensorRT-LLM/pull/4750) | Merged | Blackwell MoE plugin 的 FP8×MXFP4 支持 | 相同/相邻数值格式，但仍是 MoE plugin/grouped execution；当时只暴露 C++ API。 |
| [TensorRT-LLM #5222](https://github.com/NVIDIA/TensorRT-LLM/pull/5222) | Merged | 开源 MXFP8×MXFP4 MoE CUTLASS 实现 | 是最重要的格式和 grouped-MoE baseline，不是 Cursor output-centric 复现。 |
| [TensorRT-LLM #3294](https://github.com/NVIDIA/TensorRT-LLM/pull/3294) | Merged | CUTLASS MoE FC2+Finalize fusion | 把 routing scale/permutation finalize 融入 GEMM2 epilogue，减少一个阶段；仍需要 expert routing/分组，不能等同于 Warp Decode 在 warp 内遍历全部 top-k。 |
| [TensorRT-LLM #6645](https://github.com/NVIDIA/TensorRT-LLM/pull/6645) | Merged | GPT-OSS 与 MXFP4 MoE backend 接入 | 模型与量化格式高度相关，但使用 TRTLLM/CUTLASS/Triton MoE backend，不是 Cursor 两-kernel。 |
| [TensorRT-LLM #7937](https://github.com/NVIDIA/TensorRT-LLM/pull/7937) | Merged | GPT-OSS SM120/SM121、mixed FP8×FP4 grouped GEMM | 面向消费级 Blackwell 的 grouped GEMM 支持；架构和执行组织不同。 |
| [TensorRT-LLM #7970](https://github.com/NVIDIA/TensorRT-LLM/pull/7970) | Merged | 更新 TRT-LLM Gen MoE kernels | 内部/生成式 MoE backend 演进，没有公开 Cursor Warp Decode 工作所有权合同。 |
| [TensorRT-LLM #8156](https://github.com/NVIDIA/TensorRT-LLM/pull/8156) | Merged | MXFP4 MoE cubin 与 tile-N autotune | 是 grouped-MoE tactic 调优，不是算法轴翻转。 |
| [TensorRT-LLM #9025](https://github.com/NVIDIA/TensorRT-LLM/pull/9025) | Merged | 减少 MXFP4 weight padding、收紧 TMA bound | 减少传统路径 padding/边界成本，但仍不是“不形成 expert batch”的 Warp Decode。 |
| [TensorRT-LLM #14550](https://github.com/NVIDIA/TensorRT-LLM/pull/14550) | Merged | 删除 MoE A2A kernel 的 one-warp-per-token policy | 这里的 warp policy 属于通信/A2A 调度，不是 gate/up/down MoE 计算，因此名称相似但不应纳入实现证明。 |
| [TensorRT-LLM #17059](https://github.com/NVIDIA/TensorRT-LLM/pull/17059) | Open | Kimi K3 decode routing+MXFP8 quantization 单-launch融合，支持 token<=64 | 优化的是 MoE 前置 routing/quantize；随后仍进入 TRTLLM-Gen W4A8 backend，不是 direct gate/up/down Warp Decode。 |
| [TensorRT-LLM #17063](https://github.com/NVIDIA/TensorRT-LLM/pull/17063) | Open | Kimi K3 SiTU DeepGEMM MegaMoE | 融合 dispatch、GEMM、activation、combine 的 expert-centric MegaMoE；不是小-M output-centric 路线。 |

需要特别区分两个术语：

- TensorRT-LLM/CUTLASS 文件名里的 `warp-specialized` 通常表示 producer/consumer warp specialization 或 TMA pipeline schedule；
- Cursor Warp Decode 的 `warp` 表示每个 warp 从开始到结束拥有一个 output scalar。

前者是一个 GEMM 内部的流水分工，后者是整个 MoE 数据流和并行轴的重构，不能由名称直接互推。

### 4.4 当前公开代码结论

| 仓库 | Exact Cursor 名称/源码 | 最接近公开路径 |
| --- | --- | --- |
| Cursor | 文章公开内部名称，未公开源码 | 内部 `moe_gate_up_3d_batched` + `moe_down_3d_batched` |
| TokenSpeed | 不同实现，但明确引用 Cursor | #403/#423 gfx950 Gluon Warp Decode |
| FlashInfer | 无 exact 名称 | merged #4285 DirectMicro；open #4310 两-kernel BF16 |
| TensorRT-LLM | 无 exact 名称 | #5222 MXFP8×MXFP4 grouped MoE；#3294 FC2+Finalize fusion |
| Mixture-of-Kittens | 无 exact 名称 | NVL72 training megakernel，非 decode |

没有发现 vLLM、SGLang 或 TensorRT-LLM 中明确声明为 Cursor Warp Decode 的公开 PR。它们可能集成相邻的 direct-MoE、CUTLASS、TRTLLM-Gen 或 FlashInfer backend，但集成关系不能替代 exact kernel 证据。

## 5. 横向结论

| 实现 | 是否公开 | 是否 exact Cursor Warp Decode | 适用场景 | 主要原因 |
| --- | --- | --- | --- | --- |
| Cursor 内部 Warp Decode | 仅文章公开 | 是 | B200 小 batch decode | 原始两-kernel、output-centric 设计 |
| FlashInfer SM12x DirectMicro | 已合入 | 否；算法家族接近 | SM12x NVFP4/W4A16 tiny decode | direct routing，但架构、格式、kernel/协同合同不同 |
| FlashInfer PR #4310 | Open | 否；当前最接近的两-kernel 骨架 | SM120 BF16 `M=1..8` | 两-kernel direct route，但非 SM100/MXFP8，且未合入 |
| Cursor Mixture-of-Kittens | 已开源 | 否 | NVL72 大规模训练 | expert-centric、padding=256、dispatch/combine、前反向 |

## 6. 对 FlashInfer 的实现建议

如果目标是在 FlashInfer 中补齐 B200/SM100 Warp Decode，建议作为独立小-M backend，而不是修改通用 grouped-MoE 或直接移植 MoK：

```text
public wrapper
  -> architecture/shape/format gate
       SM100
       M <= 8/16
       topk fixed/small
       supported H/I and weight layout
  -> direct gate/up kernel
  -> direct down + top-k reduction kernel
  -> grouped/CUTLASS/MegaMoE fallback
```

优先复用：

- DirectMicro 的 `topk_ids/topk_weights` API、workspace、JIT cache、shape gate 和 fallback 经验；
- PR #4310 的两-kernel API、CUDA Graph 与 rank-local EP partial 合同；
- FlashInfer 现有 MXFP8 weight preprocessing/scale-layout 工具。

不应直接复用：

- MoK 的 256-padding scheduler；
- expert-major dispatch/combine buffer；
- 训练 backward/replay pipeline；
- SM12x-specific FP4 dot/scale contract。

合并前至少需要分别证明：

1. **数值正确性**：所有支持的 `M/topk/H/I`，BF16 activation、MXFP8 weight decode、FP32 accumulation、top-k weight folding。
2. **API/布局正确性**：gate/up 顺序、weight stride、scale block/layout、global-to-local expert map。
3. **运行正确性**：CUDA Graph、非连续路由、重复 expert、无本地 expert、Compute Sanitizer。
4. **性能边界**：在目标 B200 上与相同权重格式、相同路由分布、相同输出精度的 grouped-MoE 基线比较；独立报告 M=1、2、4、8、16 的 crossover。
5. **fallback**：大 M、非 SM100、未知 shape/layout 必须回退，不能把小-M 优化当作通用 MoE 替代。

## 7. 证据边界

- 本文的“存在/不存在”结论来自固定 SHA 的公开源码、分支、PR 和符号搜索。
- Cursor 文章中的 B200 吞吐、带宽和精度数据属于作者内部系统结果，未公开完整代码和可匹配 benchmark。
- FlashInfer PR #4310 的性能与 sanitizer 结果属于 PR 作者报告，本次未在 SM120 上复现。
- 本次没有进行 B200、SM120 或 NVL72 runtime 测试；不能从静态相似性推导生产性能。

## 参考链接

- [Cursor Warp Decode](https://cursor.com/blog/warp-decode)
- [Cursor Mixture-of-Kittens](https://github.com/cursor/mixture-of-kittens)
- [FlashInfer main @ 42ea835](https://github.com/flashinfer-ai/flashinfer/commit/42ea835cfde6aadbbee4d7c12187873cade5aaba)
- [FlashInfer PR #4310: SM120 low-token BF16 fused MoE](https://github.com/flashinfer-ai/flashinfer/pull/4310)
- [TokenSpeed PR #403: Cursor-inspired Warp Decode](https://github.com/lightseekorg/tokenspeed/pull/403)
- [TokenSpeed PR #423: Warp Decode improvements](https://github.com/lightseekorg/tokenspeed/pull/423)
- [TensorRT-LLM PR #5222: MXFP8-MXFP4 MoE](https://github.com/NVIDIA/TensorRT-LLM/pull/5222)
- [TensorRT-LLM PR #3294: FC2+Finalize fusion](https://github.com/NVIDIA/TensorRT-LLM/pull/3294)
