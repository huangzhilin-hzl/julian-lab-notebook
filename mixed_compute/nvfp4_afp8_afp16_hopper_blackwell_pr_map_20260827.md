# Hopper / Blackwell NVFP4A8 与 NVFP4A16 实现 PR 总结

> 状态快照：2026-08-27，Asia/Shanghai。
>
> 范围：公开 GitHub PR 中与 NVIDIA Hopper/Blackwell 上的 FP4 权重、FP8 或 FP16/BF16 激活直接相关的 dense GEMM、Fused MoE、MegaMoE 内核和主要 serving 接入。

## 结论

1. **Hopper SM90 没有原生 NVFP4 Tensor Core MMA。** Hopper 上的高性能 W4A8 实现通常保存 MXFP4/FP4 packed 权重，在 kernel 内将权重展开为 FP8，再使用 FP8 WGMMA。严格 NVFP4 checkpoint 则主要通过在线/加载期转换、Marlin 或 QDQ emulation 运行。
2. **Blackwell SM100 已有原生 NVFP4×FP8 实现。** TensorRT-LLM 的 dense GEMM 和 Fused MoE 路线均已合入。
3. **Hopper 当前最成熟的是 MXFP4×BF16 和 MXFP4×FP8。** FlashInfer `#3084/#3738` 已进入主线；严格 NVFP4 checkpoint + W4A8 MegaMoE 仍以 FlashInfer `#4589` 为主要开放候选。
4. **Blackwell NVFP4 W4A16 已覆盖 dense 和 MoE。** SM100/103 可看 FlashInfer `#4048/#4466`，SM120/121 MoE 可看 `#3271/#3336`。

推荐优先级：

| 目标 | 当前优先实现 |
|---|---|
| Hopper W4A16 | FlashInfer [#3084](https://github.com/flashinfer-ai/flashinfer/pull/3084)；SGLang 使用 [#24816](https://github.com/sgl-project/sglang/pull/24816) |
| Hopper W4A8 | FlashInfer [#3738](https://github.com/flashinfer-ai/flashinfer/pull/3738) 或 Humming [#35](https://github.com/inclusionAI/humming/pull/35)/[#37](https://github.com/inclusionAI/humming/pull/37) |
| Hopper 严格 NVFP4 checkpoint W4A8 | FlashInfer [#4589](https://github.com/flashinfer-ai/flashinfer/pull/4589)，尚未合入 |
| Blackwell 严格 NVFP4×FP8 | TensorRT-LLM [#6809](https://github.com/NVIDIA/TensorRT-LLM/pull/6809)/[#7968](https://github.com/NVIDIA/TensorRT-LLM/pull/7968) |
| Blackwell 严格 NVFP4×BF16 | FlashInfer [#4048](https://github.com/flashinfer-ai/flashinfer/pull/4048)/[#4466](https://github.com/flashinfer-ai/flashinfer/pull/4466)；SM120 MoE 使用 [#3336](https://github.com/flashinfer-ai/flashinfer/pull/3336) |

## 精度口径

本文区分两种容易被混用的 FP4 格式：

| 格式 | FP4 数据 | Block scale | 常见表述 |
|---|---|---|---|
| NVFP4 | E2M1 | 每 16 个元素一组，E4M3 scale，通常另有 global scale | NVFP4、W4A4、NVFP4 W4A8/W4A16 |
| MXFP4 | E2M1 | 每 32 个元素一组，UE8M0/E8M0 scale | MXFP4、MXFP4×FP8、MXFP4×BF16 |

`A8` 表示 FP8 activation，`A16` 表示 FP16/BF16 activation。部分接口接收 BF16 输入后在 kernel 内量化为 FP8/FP4；本文会单独说明这种情况。

## 1. 严格 NVFP4 路线

### 1.1 Hopper SM90

| PR | 状态 | 精度与实现 | 判断 |
|---|---|---|---|
| [FlashInfer #4589](https://github.com/flashinfer-ai/flashinfer/pull/4589) | OPEN、REVIEW_REQUIRED；1 项 check 失败 | NVFP4 checkpoint + W4A8 MegaMoE。packed E2M1 权重常驻 HBM，kernel 内解码为 FP8 后进入 Hopper WGMMA；另有 load-time folded-FP8 和 hot/dual residency 策略。 | 当前最接近严格 Hopper NVFP4A8 高性能 MegaMoE 的公开候选，但尚未满足主线可用条件。 |
| [TensorRT-LLM #14009](https://github.com/NVIDIA/TensorRT-LLM/pull/14009) | OPEN、Draft、REVIEW_REQUIRED；1 项 check 失败 | NVFP4 W4A16 fallback。Dense 在执行时解量化为 BF16 后调用标准 linear；MoE expert 在加载期解量化为 BF16。 | 是 checkpoint 兼容方案，不是保持 4-bit compute 的 kernel；MoE 还会失去 FP4 权重常驻的显存优势。 |
| [vLLM #35733](https://github.com/vllm-project/vllm/pull/35733) → [#40033](https://github.com/vllm-project/vllm/pull/40033) → [#44667](https://github.com/vllm-project/vllm/pull/44667) | 全部 MERGED | 依次加入 Hopper/非原生平台 NVFP4 dense emulation、Triton NVFP4 dequant/QDQ kernel，以及 MoE `w13/w2` 的 fused NVFP4 dequant + compute。 | 能运行严格 NVFP4 checkpoint，但属于软件 emulation，不是 Hopper 原生 FP4 MMA。 |
| [vLLM #41769](https://github.com/vllm-project/vllm/pull/41769) → [#42566](https://github.com/vllm-project/vllm/pull/42566) | 全部 MERGED | ModelOpt `W4A16_NVFP4` dense 与 fused MoE 接入；FP16/BF16 activation，实际路由到 FP4 Marlin。 | 当前框架层较完整的 NVFP4 W4A16 路径。 |

### 1.2 Blackwell

| PR | 硬件 | 状态 | 精度与实现 |
|---|---|---|---|
| [TensorRT-LLM #6809](https://github.com/NVIDIA/TensorRT-LLM/pull/6809) | SM100 | MERGED | TRTLLM-Gen 原生 NVFP4×FP8 dense GEMM，包含 `sm100a` FP4×FP8 cubin 与量化/scale plumbing。 |
| [TensorRT-LLM #7968](https://github.com/NVIDIA/TensorRT-LLM/pull/7968) | SM100 | MERGED | 在 `#6809` 基础上加入 W4A8 NVFP4/FP8 Fused MoE。 |
| [FlashInfer #3597](https://github.com/flashinfer-ai/flashinfer/pull/3597) | SM100/103/110/120/121 | MERGED | `mm_bf16_fp4` dense GEMM：BF16 activation × NVFP4 weight；提供 cuDNN 与 CuTeDSL backend。 |
| [FlashInfer #4466](https://github.com/flashinfer-ai/flashinfer/pull/4466) → [#4686](https://github.com/flashinfer-ai/flashinfer/pull/4686) | SM100/103 | MERGED | SM100 CuTeDSL NVFP4 W4A16 dense kernel，以及 optimizer level/raster autotune 性能优化。 |
| [FlashInfer #4048](https://github.com/flashinfer-ai/flashinfer/pull/4048) | SM100 | MERGED | CuTeDSL NVFP4 MoE 的显式 `quant_mode="w4a16"`；保留 NVFP4 packed weight，在线 decode 为 BF16 后执行 MoE GEMM。 |
| [FlashInfer #3271](https://github.com/flashinfer-ai/flashinfer/pull/3271) → [#3336](https://github.com/flashinfer-ai/flashinfer/pull/3336) | SM120/121 | MERGED | B12x W4A16 fused MoE；`#3336` 用 packed-route 设计替换早期 static/dynamic/micro kernel split。 |
| [SGLang #35120](https://github.com/sgl-project/sglang/pull/35120) | SM100/103 | OPEN、REVIEW_REQUIRED；多项 checks 失败 | 接入 FlashInfer CuTeDSL NVFP4 W4A16 dense + MoE，并保持 activation/output 为 BF16。当前仍依赖 FlashInfer 版本更新和 CI 收口。 |

## 2. Hopper 上更成熟的 MXFP4 路线

下表经常被宽泛归入“FP4A8/FP4A16”，但权重格式是 MXFP4 group-32/E8M0，并非严格 NVFP4。

| PR | 状态 | 精度与作用 |
|---|---|---|
| [FlashInfer #3084](https://github.com/flashinfer-ai/flashinfer/pull/3084) | MERGED | SM90 CUTLASS mixed-input MoE。核心路径为 MXFP4×BF16/W4A16，同时优化 INT4×FP8；加入 Hopper weight/scale interleave helpers。 |
| [SGLang #24816](https://github.com/sgl-project/sglang/pull/24816) | MERGED | 将 FlashInfer `#3084` 的 MXFP4 W4A16 MoE 接入 GPT-OSS 与 DeepSeek-V4。 |
| [FlashInfer #3738](https://github.com/flashinfer-ai/flashinfer/pull/3738) | MERGED | 当前 FlashInfer 主线的 Hopper MXFP4×FP8 MoE：将 weight scale 变换为 E8M0 offset，在 MMA 前融合到 FP8 operand，执行 FP8 WGMMA。 |
| [Humming #35](https://github.com/inclusionAI/humming/pull/35) → [#37](https://github.com/inclusionAI/humming/pull/37) | MERGED | 为 fused-E8M0 MXFP4 W4A8 GEMM 增加 per-token-group FP8 activation scale，并支持 DeepEP group-128 输入布局。 |
| [SGLang #34967](https://github.com/sgl-project/sglang/pull/34967) | OPEN、REVIEW_REQUIRED；多项 checks 失败 | 将 FlashInfer `#3738` 作为显式 `fp8` precision 接入。旧候选 [#27806](https://github.com/sgl-project/sglang/pull/27806) 仍开放，但当前应优先跟踪 `#34967`。 |
| [FlashInfer #3349](https://github.com/flashinfer-ai/flashinfer/pull/3349) | OPEN、REVIEW_REQUIRED；1 项 check 失败 | 早期 SM90 CUTLASS MXFP4×FP8 Fused MoE 候选。 |
| [FlashInfer #3516](https://github.com/flashinfer-ai/flashinfer/pull/3516) | OPEN、REVIEW_REQUIRED | CuTeDSL W4A8 MXFP4 grouped GEMM/MoE；FP4 在 kernel 内 decode 为 FP8，包含 fused gather/scatter/SwiGLU。主线已有 `#3738` 的另一套实现。 |
| [DeepGEMM #411](https://github.com/deepseek-ai/DeepGEMM/pull/411) | OPEN；当前 `DIRTY`/有冲突 | SM90 FP8×MXFP4 persistent MegaMoE。packed MXFP4 常驻 HBM，在线展开成 FP8；融合 dispatch、L1/L2、shared expert 和 combine。 |
| [sgl-project/DeepGEMM #53](https://github.com/sgl-project/DeepGEMM/pull/53) | OPEN、`CLEAN`；无公开 checks/approval | SM90 FP8 activation × packed FP4 weight MegaMoE，带 small-batch swapAB。 |

## 3. Blackwell 的 MXFP4 / 泛 FP4 A8、A16 路线

| PR | 状态 | 精度与作用 |
|---|---|---|
| [DeepGEMM #304](https://github.com/deepseek-ai/DeepGEMM/pull/304) | MERGED | Blackwell FP8×FP4 GEMM 与 MegaMoE 初始公开实现，融合 dispatch、FC1、SwiGLU、FC2 和 combine。PR 本身使用泛化的“FP8×FP4”表述。 |
| [FlashInfer #4159](https://github.com/flashinfer-ai/flashinfer/pull/4159) | MERGED | 通过 unified MoE API 暴露 SM100/103 MXFP4×MXFP8 W4A8，以及 SM100 MXFP4×BF16 W4A16 TRTLLM-Gen backend。 |
| [FlashInfer #4361](https://github.com/flashinfer-ai/flashinfer/pull/4361) | OPEN、BLOCKED、REVIEW_REQUIRED；1 项 check 失败 | Blackwell PrimsTS MoE backend，支持 MXFP4×MXFP8 和 MXFP4×BF16；B200 qualification 尚未完成。 |
| [FlashInfer #4632](https://github.com/flashinfer-ai/flashinfer/pull/4632) | OPEN、REVIEW_REQUIRED；stacked on `#4387` | SM120 CuTeDSL W4A8 MegaMoE：MXFP4 weight × MXFP8 activation，保留通信/计算重叠和 CUDA Graph 支持。 |

## 4. 主链关系

```text
Hopper W4A16:
  FlashInfer #3084 -> SGLang #24816                       已合入

Hopper W4A8 MXFP4:
  Humming #35/#37                                         已合入
  FlashInfer #3738 -> SGLang #34967                       内核已合入，SGLang 接入开放
  FlashInfer #3349/#3516                                  早期/并行开放候选

Hopper W4A8 MegaMoE:
  DeepGEMM #411 / sgl-project DeepGEMM #53                开放
  FlashInfer #4589                                        严格 NVFP4 checkpoint 候选，开放

Hopper NVFP4 W4A16 compatibility:
  vLLM #35733 -> #40033 -> #44667                         emulation 链已合入
  vLLM #41769 -> #42566                                   Marlin W4A16 链已合入
  TensorRT-LLM #14009                                     Draft fallback

Blackwell NVFP4 W4A8:
  TensorRT-LLM #6809 -> #7968                             dense + Fused MoE 已合入

Blackwell NVFP4 W4A16:
  FlashInfer #4466 -> #4686                               SM100/103 dense 已合入
  FlashInfer #4048                                        SM100 MoE 已合入
  FlashInfer #3271 -> #3336                               SM120/121 MoE 已合入
  SGLang #35120                                           serving 接入开放
```

## 5. 未计入的相近 PR

以下 PR 虽然包含 NVFP4，但不属于本文要求的 A8/A16 主范围：

- [TensorRT-LLM #14608](https://github.com/NVIDIA/TensorRT-LLM/pull/14608)：SM100/103 `MegaMoECuteDsl` NVFP4 backend，主要是 NVFP4×NVFP4/W4A4 MegaMoE。
- [FlashInfer #4340](https://github.com/flashinfer-ai/flashinfer/pull/4340)：SM100/103 AlphaMoE，activation 和 weight 都是 packed E2M1，属于 W4A4。
- NVFP4 KV cache、attention、quantization-only、测试、文档、单纯 bugfix PR：除非它们直接改变 A8/A16 GEMM/MoE 能力，否则不纳入主表。
- INT4×FP8 Hopper kernel：与 NVFP4/MXFP4 的 scale 和 checkpoint contract 不同，不应仅凭“W4A8”名称混为一条路线。

## 6. 使用这些 PR 时的判断标准

1. 先确认 checkpoint 是 NVFP4 group-16/E4M3 scale，还是 MXFP4 group-32/E8M0 scale。
2. 区分“输入接口为 BF16、kernel 内再量化”和“真正以 FP8 activation/scales 进入 GEMM”。
3. Hopper 上看到 FP4 kernel 时，要确认实际是 FP4→FP8/BF16 decode、Marlin，还是完整权重预展开。
4. `MERGED` 只说明代码进入对应目标分支；框架是否可用仍取决于 wheel/version bump、backend selection、模型权重布局和 CUDA/CuTeDSL 版本。
5. 对开放 PR，应同时检查 merge state、review decision、CI、正确性测试和具体硬件 benchmark，不能仅依据 PR 标题判断可用性。

