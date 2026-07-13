# Marlin、Machete 与 Swordfish 参考资料及代码 PR

> 整理日期：2026-07-13<br>
> 主线资料：[Swordfish 设计长文](https://blog.alpindale.net/posts/swordfish/) · [dphnAI/sonar #1707](https://github.com/dphnAI/sonar/pull/1707)

## 核心结论

`Marlin -> Machete -> Swordfish` 是 weight-only 量化 GEMM 随 NVIDIA GPU 架构演进的设计谱系，不是三个相同代码库的简单 fork。

| Kernel | 目标硬件 | Tensor Core / 实现路径 | 与前代关系 | 主要覆盖 |
| --- | --- | --- | --- | --- |
| Marlin | Ampere / Ada 起步；downstream 版本也作为后续架构 fallback | 手写 CUDA/PTX、`mma.sync`、固定的预排权重布局 | 原始基线 | 原始仓库以 FP16 x INT4 为主；vLLM 后续扩展 BF16、INT8、AWQ、act-order、MoE |
| Machete | Hopper `sm90` | CUTLASS/CuTe、TMA、`wgmma`、可配置的预排布局 | Marlin 的 Hopper “spiritual successor”；不是原始 Marlin kernel 的直接换指令版本 | FP16/BF16 activation，WNA16；后续增加 zero point、group size 64 等 |
| Swordfish | Blackwell `sm100`、Thor `sm110`；不支持 `sm120` | 小/中 M decode：`mma.sync`；中 M prefill：`tcgen05`；大 M：dequant once + cuBLAS | Decode/ABI 直接复用 Marlin；prefill 借鉴 Machete fork CUTLASS collective 的方法，但没有运行时依赖 Machete | GPTQ INT4/INT8、AWQ+ZP、act-order、FP16/BF16、group 32/64/128、channelwise、fused MoE |

## 参考资料与主体代码

| Kernel | 类型 | 资料 | 用途 |
| --- | --- | --- | --- |
| Marlin | 论文 | [MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models](https://arxiv.org/abs/2408.11743) | 算法动机、batch 16-32 仍保持接近理想量化加速的设计 |
| Marlin | 原始仓库 | [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin) | 原始实现、测试和 benchmark |
| Marlin | 核心文件 | [marlin_cuda_kernel.cu](https://github.com/IST-DASLab/marlin/blob/master/marlin/marlin_cuda_kernel.cu) | 原始单文件 CUDA kernel |
| Machete | 设计与初始实现 | [vLLM #7174](https://github.com/vllm-project/vllm/pull/7174) | 解释为什么 Hopper 需要 `wgmma`，加入 kernel、测试和 benchmark |
| Machete | README | [Machete Readme](https://github.com/vllm-project/vllm/blob/5288c06aa03b100eab4f873452b65da941a1a232/csrc/quantization/machete/Readme.md) | Hopper/CUTLASS 设计定位与支持范围 |
| Swordfish | 作者长文 | [Swordfish, a Weight-Quantized GEMM Family for NVIDIA Blackwell](https://blog.alpindale.net/posts/swordfish/) | 完整设计说明、调度分段、benchmark 和数值行为 |
| Swordfish | 主体 PR | [dphnAI/sonar #1707](https://github.com/dphnAI/sonar/pull/1707) | 唯一主体代码 PR；31 commits、30 files、`+7078/-12` |
| Swordfish | 合并提交 | [a231864e](https://github.com/dphnAI/sonar/commit/a231864e9670fa758997c105c614e055aed01e06) | 固定到最终合并代码快照 |

## Marlin 关键 PR

博客中的 Marlin benchmark 指 Sonar/vLLM downstream 扩展版，不能把 INT8、AWQ、BF16、MoE 等能力归到原始 Marlin 仓库。

| PR | 状态 / 时间 | 作用 |
| --- | --- | --- |
| [vLLM #2497](https://github.com/vllm-project/vllm/pull/2497) | Merged · 2024-03-01 | 首次将 Marlin INT4 GPTQ kernel 集成到 vLLM |
| [vLLM #3922](https://github.com/vllm-project/vllm/pull/3922) | Merged · 2024-04-29 | AutoGPTQ 直接加载、act-order、更多 group size、运行时 repack |
| [vLLM #4533](https://github.com/vllm-project/vllm/pull/4533) | Merged · 2024-05-02 | GPTQ INT8 |
| [vLLM #4788](https://github.com/vllm-project/vllm/pull/4788) | Merged · 2024-05-16 | BF16 activation |
| [vLLM #6612](https://github.com/vllm-project/vllm/pull/6612) | Merged · 2024-07-21 | AWQ / zero point |
| [vLLM #7766](https://github.com/vllm-project/vllm/pull/7766) | Merged · 2024-08-27 | Fused Marlin MoE kernel；这是被 revert 后重新落地的版本 |
| [vLLM #8217](https://github.com/vllm-project/vllm/pull/8217) | Merged · 2024-09-10 | GPTQ frontend 接入 fused Marlin MoE |
| [vLLM #8973](https://github.com/vllm-project/vllm/pull/8973) | Merged · 2024-10-04 | Marlin MoE zero point 与 AWQ fused MoE |
| [Sonar #547](https://github.com/dphnAI/sonar/pull/547) | Merged · 2024-07-23 | Sonar/Aphrodite 集成 Marlin GPTQ kernel |

## Machete 关键 PR

Machete 未见官方独立论文或独立仓库；主体代码和设计讨论都在 vLLM。

| PR | 状态 / 时间 | 作用 |
| --- | --- | --- |
| [vLLM #7174](https://github.com/vllm-project/vllm/pull/7174) | Merged · 2024-08-20 | Hopper 优化的核心 Machete kernel；CUTLASS/CuTe + `wgmma` |
| [vLLM #7701](https://github.com/vllm-project/vllm/pull/7701) | Merged · 2024-09-23 | 接入 `CompressedTensorsWNA16` 与 `GPTQMarlin`，加入 dynamic `g_idx` / act-order 路径 |
| [vLLM #9855](https://github.com/vllm-project/vllm/pull/9855) | Merged · 2024-11-18 | W4A8/QQQ 底层扩展与 dispatch/codegen 重构；PR 当时仍把 E2E 接线列为后续工作 |
| [vLLM #20268](https://github.com/vllm-project/vllm/pull/20268) | Merged · 2025-07-01 | Zero point 支持 |
| [vLLM #20290](https://github.com/vllm-project/vllm/pull/20290) | Merged · 2025-07-02 | Group size 64 |
| [vLLM #20830](https://github.com/vllm-project/vllm/pull/20830) | Merged · 2025-07-12 | 明确限制 Machete 只运行于 Hopper |
| [Sonar #842](https://github.com/dphnAI/sonar/pull/842) | Merged · 2024-11-27 | Sonar 集成 Machete；PR 报告 H100 GPTQ serving throughput 相对 Marlin 提升 46% |

## Swordfish #1707 代码地图

| 模块 | 代码 | 重点 |
| --- | --- | --- |
| Backend gate 与权重加载 | [`swordfish.py`](https://github.com/dphnAI/sonar/blob/a231864e9670fa758997c105c614e055aed01e06/aphrodite/model_executor/kernels/linear/mixed_precision/swordfish.py#L27-L156) | `sm100/sm110` gate、act-order、channelwise scale replication、统一 `swordfish_mm` 入口 |
| Packed ABI | [`swordfish_abi.cuh`](https://github.com/dphnAI/sonar/blob/a231864e9670fa758997c105c614e055aed01e06/csrc/libtorch_stable/quantization/swordfish/swordfish_abi.cuh#L1-L41) | 保留 Marlin 16x64 tile 内置换，改成 Swordfish block-linear 外层布局 |
| Prepack | [`swordfish_prepack.cu`](https://github.com/dphnAI/sonar/blob/a231864e9670fa758997c105c614e055aed01e06/csrc/libtorch_stable/quantization/swordfish/swordfish_prepack.cu#L1-L70) | 先做 `gptq_marlin_repack`，再重排为 Swordfish ABI |
| Decode | [`swordfish_decode.cuh`](https://github.com/dphnAI/sonar/blob/a231864e9670fa758997c105c614e055aed01e06/csrc/libtorch_stable/quantization/swordfish/swordfish_decode.cuh#L1-L30) | 直接 include Marlin dtype、dequant、MMA、`cp.async` helper |
| Runtime dispatch | [`swordfish_mm.cu`](https://github.com/dphnAI/sonar/blob/a231864e9670fa758997c105c614e055aed01e06/csrc/libtorch_stable/quantization/swordfish/swordfish_mm.cu#L38-L105) | 按真实 M、shape 和 SM 数在 decode / `tcgen05` prefill / dense tier 间切换 |
| `tcgen05` prefill | [`swordfish_prefill_mainloop.cuh`](https://github.com/dphnAI/sonar/blob/a231864e9670fa758997c105c614e055aed01e06/csrc/libtorch_stable/quantization/swordfish/swordfish_prefill_mainloop.cuh#L1-L36) | Fork CUTLASS 4.4.2 SM100 mixed-input collective；Marlin tile order解量化后喂给 `tcgen05` |
| Dense tier | [`swordfish_dense_tier.cu`](https://github.com/dphnAI/sonar/blob/a231864e9670fa758997c105c614e055aed01e06/csrc/libtorch_stable/quantization/swordfish/swordfish_dense_tier.cu) | 大 M 时一次解量化为 FP16/BF16 scratch，再调用 cuBLAS |
| Fused MoE | [`swordfish_moe.cu`](https://github.com/dphnAI/sonar/blob/a231864e9670fa758997c105c614e055aed01e06/csrc/libtorch_stable/quantization/swordfish/swordfish_moe.cu) | Token-sorted Stream-K MoE 路径 |
| MoE 上层调度 | [`swordfish_moe.py`](https://github.com/dphnAI/sonar/blob/a231864e9670fa758997c105c614e055aed01e06/aphrodite/model_executor/layers/fused_moe/experts/swordfish_moe.py#L217-L329) | Fused、per-expert `tcgen05` 与 dense fused-experts 三档切换 |

## Swordfish #1707 关键提交

| 能力 / 优化 | Commit |
| --- | --- |
| 初始 W4A16 Blackwell kernel | [bf8e9ed](https://github.com/dphnAI/sonar/commit/bf8e9ed4af8cdf1523859a3138560ed0face09b6) |
| Stream-K decode window | [6c8d7dc](https://github.com/dphnAI/sonar/commit/6c8d7dca9b7cce8977f74a09601689fd0519d36c) |
| AWQ / zero point | [52e76f3](https://github.com/dphnAI/sonar/commit/52e76f31f0a28655406aad93a301aef81fb7efc0) |
| GPTQ INT8 | [9b4c4df](https://github.com/dphnAI/sonar/commit/9b4c4df6a25debefdab8cdb10f998292eadd266c) |
| Act-order | [4789c52](https://github.com/dphnAI/sonar/commit/4789c5201fc54eaaaf9bb95b07917b681b9f431e) |
| Fused MoE | [89d5206](https://github.com/dphnAI/sonar/commit/89d5206188c788a82c11342904feeacef0305755) |
| 大 M dense tier | [282200f](https://github.com/dphnAI/sonar/commit/282200f9f9cb1ea822e30bcd24ef1d403ee88c21) |
| Channelwise scale replication | [53a9ec6](https://github.com/dphnAI/sonar/commit/53a9ec616b172bcdcc6fd03e2438e2ccfa87bb02) |
| Many-SM fused atomic decode band | [311d86c](https://github.com/dphnAI/sonar/commit/311d86c23fbcb951cef2b0c14ee2a512c755acf6) |
| Few-SM column-quad scheduling | [de4b1a0](https://github.com/dphnAI/sonar/commit/de4b1a08d770cf051d18b137ca8d56099e1d2cd4) |

## Benchmark 解读边界

| 项目 | 结论 |
| --- | --- |
| Machete 对比 | #1707 没有在 B200/Thor 上运行 Machete；PR 明确说明它是 Hopper-only，因此不存在 Swordfish-vs-Machete 实测数字 |
| Marlin 对比 | Marlin 是 GPTQ4、AWQ、GPTQ8、act-order、MoE、channelwise 表格的共同 baseline |
| Prefill | Swordfish 在 B200/Thor 上通常显著领先 Marlin，主要来自 `tcgen05` prefill 和大 M dense tier |
| Decode | 多数区间接近 Marlin；Thor batch 32 的 M=17-48 区间仍存在最多约 3% 的 Marlin 优势 |
| 自动 fallback | 仅 `linear_backend=auto` 时，unsupported corner 才会继续选择 Marlin；显式强制 `--linear-backend swordfish` 不会自动回退 |

## 补充资料

| 主题 | 资料 |
| --- | --- |
| GPTQ | [论文](https://arxiv.org/abs/2210.17323) · [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq) |
| AWQ | [论文](https://arxiv.org/abs/2306.00978) · [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq) |
| Hopper WGMMA | [NVIDIA PTX ISA：WGMMA](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma) |
| Blackwell `tcgen05` | [NVIDIA PTX ISA：tcgen05.mma](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions-tcgen05-mma) |
| CUTLASS collective | [CUTLASS 4.4.2 SM100 mixed-input collective](https://github.com/NVIDIA/cutlass/blob/v4.4.2/include/cutlass/gemm/collective/sm100_mma_warpspecialized_mixed_input.hpp) |
| QuTLASS | [IST-DASLab/qutlass](https://github.com/IST-DASLab/qutlass)；博客称其 SM100 配置帮助定位了 Swordfish 的 instruction-width ceiling，它不是 Swordfish 代码库 |

## 建议阅读顺序

| 顺序 | 资料 |
| --- | --- |
| 1 | Marlin 论文与原始仓库，理解 weight-only GEMM、预排布局和 `mma.sync` decode |
| 2 | vLLM #2497、#3922，理解 downstream Marlin 与原始 Marlin 的差别 |
| 3 | Machete #7174、#7701，理解 Hopper 上为何切换到 CUTLASS/CuTe + `wgmma` |
| 4 | Swordfish 长文，理解 Blackwell 下 decode / prefill / dense 三档设计 |
| 5 | Sonar #1707，按 ABI -> decode -> runtime dispatch -> prefill -> dense -> MoE 顺序阅读 |
