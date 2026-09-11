# mHC PR 全索引（2026-09-11）

> 266 个 PR，28 个仓库：143 MERGED、67 OPEN、56 CLOSED 未合入。状态来自逐 PR REST 详情；日期为 UTC。OPEN/CLOSED 不代表当前 main 一定缺少等价实现。

[主报告：kernel、关键进展与性能证据](mhc-pr-kernel-progress-20260911.md) · [结构化快照](evidence/20260911/snapshot.json)

## 仓库统计

| 仓库 | 合入 | 开放 | 关闭未合入 | 合计 |
| --- | ---: | ---: | ---: | ---: |
| NVIDIA-NeMo/Automodel | 0 | 1 | 0 | 1 |
| NVIDIA/Megatron-LM | 17 | 3 | 16 | 36 |
| NVIDIA/TensorRT-LLM | 16 | 0 | 8 | 24 |
| NVIDIA/TileGym | 2 | 1 | 1 | 4 |
| NVIDIA/TransformerEngine | 3 | 0 | 1 | 4 |
| ROCm/ATOM | 2 | 1 | 1 | 4 |
| ROCm/aiter | 29 | 5 | 5 | 39 |
| ROCm/rocm-libraries | 0 | 0 | 1 | 1 |
| ai-dynamo/aiconfigurator | 3 | 0 | 1 | 4 |
| alibaba/rtp-llm | 0 | 1 | 0 | 1 |
| deepseek-ai/DeepGEMM | 2 | 1 | 1 | 4 |
| deepseek-ai/TileKernels | 0 | 2 | 1 | 3 |
| flashinfer-ai/flashinfer | 1 | 2 | 0 | 3 |
| lightseekorg/tokenspeed | 1 | 0 | 0 | 1 |
| sgl-project/DeepGEMM | 1 | 1 | 0 | 2 |
| sgl-project/sgl-kernel-npu | 0 | 2 | 0 | 2 |
| sgl-project/sgl-kernel-xpu | 3 | 0 | 0 | 3 |
| sgl-project/sglang | 28 | 12 | 9 | 49 |
| sgl-project/sglang-jax | 1 | 0 | 1 | 2 |
| tile-ai/TileOPs | 7 | 0 | 0 | 7 |
| tile-ai/tilelang | 2 | 1 | 0 | 3 |
| tile-ai/tilelang-ascend | 3 | 13 | 1 | 17 |
| tile-ai/tilelang-mlir-ascend | 1 | 0 | 0 | 1 |
| vllm-project/tpu-inference | 2 | 0 | 0 | 2 |
| vllm-project/vllm | 18 | 14 | 7 | 39 |
| vllm-project/vllm-ascend | 0 | 4 | 1 | 5 |
| vllm-project/vllm-omni | 0 | 2 | 0 | 2 |
| vllm-project/vllm-xpu-kernels | 1 | 1 | 1 | 3 |

## 全部 OPEN PR

下列 merge/review/check 信息只对重点候选做专项采样。`BLOCKED` 是合入门禁状态，`FAILURE` 不一定是数值测试失败；没有 checks 也不等于已验证。

| PR | 状态 / 目标 | 标题 | 最近更新 UTC | 重点采样 |
| --- | --- | --- | --- | --- |
| [ROCm/aiter#5412](https://github.com/ROCm/aiter/pull/5412) | OPEN → `main` | \[HIP\] \[JIT\] feat: add packed BF16 mHC computation and gfx1250 tuning | 2026-09-11 | MERGEABLE; BLOCKED; FAILURE=3, QUEUED=2, SKIPPED=28, SUCCESS=6 |
| [sgl-project/DeepGEMM#86](https://github.com/sgl-project/DeepGEMM/pull/86) | OPEN · draft → `dev` | Merge upstream main (Public Release 26/09) into dev | 2026-09-11 | MERGEABLE; CLEAN; 无可见 checks |
| [vllm-project/vllm#50178](https://github.com/vllm-project/vllm/pull/50178) | OPEN → `main` | \[9/N\]\[warmup\]\[DSv4\] Migrate MHC TileLang kernels | 2026-09-11 | CONFLICTING; DIRTY; APPROVED; FAILURE=1, SUCCESS=7 |
| [vllm-project/vllm#55573](https://github.com/vllm-project/vllm/pull/55573) | OPEN → `main` | \[Performance\]\[Model\] DeepSeek-V4: gather logits rows before hc_head on prefill steps | 2026-09-11 | 未专项采样 |
| [NVIDIA-NeMo/Automodel#3855](https://github.com/NVIDIA-NeMo/Automodel/pull/3855) | OPEN → `main` | feat: add DeepSeek-V4.1-Flash model training support | 2026-09-11 | 未专项采样 |
| [sgl-project/sglang#38798](https://github.com/sgl-project/sglang/pull/38798) | OPEN → `main` | \[Model\] Add DeepSeek V4.1 support | 2026-09-11 | 未专项采样 |
| [vllm-project/vllm-ascend#16230](https://github.com/vllm-project/vllm-ascend/pull/16230) | OPEN → `main` | \[Performance\]\[Model\] Elide DeepSeek-V4 mHC residual clones | 2026-09-11 | MERGEABLE; BLOCKED; REVIEW_REQUIRED; FAILURE=1, SKIPPED=7, SUCCESS=5 |
| [vllm-project/vllm#56342](https://github.com/vllm-project/vllm/pull/56342) | OPEN → `main` | \[ROCm\]\[DSv4.1\] Fix TileLang mHC pre on 64-wide wavefronts and use the fused kernels on the AMD path | 2026-09-11 | MERGEABLE; BLOCKED; REVIEW_REQUIRED; FAILURE=1, SKIPPED=5, SUCCESS=5 |
| [ROCm/ATOM#2195](https://github.com/ROCm/ATOM/pull/2195) | OPEN → `main` | feat: enable BF16 mHC computation and fix tracing copies | 2026-09-11 | 未专项采样 |
| [alibaba/rtp-llm#1371](https://github.com/alibaba/rtp-llm/pull/1371) | OPEN → `feat/dsv4_on_dev` | feat(dsv4): integrate MegaKernel decode | 2026-09-11 | 未专项采样 |
| [vllm-project/vllm-ascend#16321](https://github.com/vllm-project/vllm-ascend/pull/16321) | OPEN → `main` | \[Performance\]\[Model\] Reuse fused mHC operators for GLM-5.3-Flash | 2026-09-11 | MERGEABLE; BLOCKED; REVIEW_REQUIRED; CANCELLED=4, SKIPPED=14, SUCCESS=8 |
| [vllm-project/vllm-omni#7261](https://github.com/vllm-project/vllm-omni/pull/7261) | OPEN · draft → `main` | feat(magi2): add opt-in mHC post-processing CustomOps | 2026-09-11 | 未专项采样 |
| [NVIDIA/Megatron-LM#7054](https://github.com/NVIDIA/Megatron-LM/pull/7054) | OPEN → `main` | GLM5.3 Flash (KDA + mHC + KPool DSA) support + FP8 | 2026-09-10 | 未专项采样 |
| [vllm-project/vllm#51802](https://github.com/vllm-project/vllm/pull/51802) | OPEN → `main` | \[Bugfix\] Fix NVIDIA DeepSeek V4 mHC warmup | 2026-09-10 | MERGEABLE; BLOCKED; REVIEW_REQUIRED; ACTION_REQUIRED=1, FAILURE=1, SKIPPED=2, SUCCESS=4 |
| [vllm-project/vllm#56255](https://github.com/vllm-project/vllm/pull/56255) | OPEN → `dsv41-optimized` | \[DSv4.1\] Integrate Mega-mHC from DeepGEMM | 2026-09-10 | MERGEABLE; UNSTABLE; FAILURE=1, SKIPPED=7, SUCCESS=5 |
| [sgl-project/sglang#38545](https://github.com/sgl-project/sglang/pull/38545) | OPEN → `main` | \[AMD\] \[GLM-5.3-Flash Day 0\] Route mHC through AITER on gfx950 | 2026-09-10 | MERGEABLE; BLOCKED; REVIEW_REQUIRED; CANCELLED=6, FAILURE=22, SKIPPED=55, SUCCESS=87 |
| [vllm-project/vllm#51244](https://github.com/vllm-project/vllm/pull/51244) | OPEN → `main` | feat: optimize MHC post + HC head + RMSNorm fusions for DeepSeek V4 | 2026-09-09 | MERGEABLE; BLOCKED; REVIEW_REQUIRED; FAILURE=1, SKIPPED=2, SUCCESS=5 |
| [vllm-project/vllm#52941](https://github.com/vllm-project/vllm/pull/52941) | OPEN → `main` | \[Bugfix\]\[Model\]\[NVIDIA\] Fix DeepSeek V4 mHC TileLang warmup for nvidi… | 2026-09-09 | MERGEABLE; BLOCKED; REVIEW_REQUIRED; FAILURE=2, SKIPPED=4, SUCCESS=3 |
| [vllm-project/vllm-omni#7288](https://github.com/vllm-project/vllm-omni/pull/7288) | OPEN · draft → `main` | \[Refactor\] Express MAGI-2 mHC pre application as a batch matmul | 2026-09-08 | 未专项采样 |
| [sgl-project/sglang#37168](https://github.com/sgl-project/sglang/pull/37168) | OPEN → `main` | Prevent full CUDA-graph serving from corrupting MHC and DSA tensors | 2026-09-08 | CONFLICTING; DIRTY; REVIEW_REQUIRED; FAILURE=16, SKIPPED=88, SUCCESS=17 |
| [vllm-project/vllm#53055](https://github.com/vllm-project/vllm/pull/53055) | OPEN → `main` | \[Bugfix\]\[Kernels\] Fallback mhc_pre_broadcast to TileLang when DeepGEMM unsupported | 2026-09-08 | MERGEABLE; BLOCKED; REVIEW_REQUIRED; FAILURE=1, SKIPPED=2, SUCCESS=5 |
| [ROCm/aiter#5296](https://github.com/ROCm/aiter/pull/5296) | OPEN → `main` | \[HIP\] perf(mhc): cap the split-k partial-reduction width in mhc_pre_big_fuse | 2026-09-08 | MERGEABLE; BLOCKED; SKIPPED=13, SUCCESS=36 |
| [sgl-project/sglang#38475](https://github.com/sgl-project/sglang/pull/38475) | OPEN → `main` | \[DSV4\] Use mhc_pre's split-K pre-norm GEMM in the mhc_fused_post_pre fallback | 2026-09-08 | CONFLICTING; DIRTY; REVIEW_REQUIRED; FAILURE=16, SKIPPED=88, SUCCESS=17 |
| [sgl-project/sglang#38442](https://github.com/sgl-project/sglang/pull/38442) | OPEN → `main` | Fix DeepGEMM import for independently enabled MHC prenorm | 2026-09-08 | MERGEABLE; BLOCKED; REVIEW_REQUIRED; FAILURE=16, SKIPPED=88, SUCCESS=17 |
| [tile-ai/tilelang-ascend#1621](https://github.com/tile-ai/tilelang-ascend/pull/1621) | OPEN → `ascendc_pto` | \[Example\] Add mhc_pre operator for Ascend NPU | 2026-09-07 | 未专项采样 |
| [tile-ai/tilelang-ascend#1712](https://github.com/tile-ai/tilelang-ascend/pull/1712) | OPEN → `ascendc_pto` | 新增examples/cann-bench/mhc_sinkhorn算子 | 2026-09-04 | 未专项采样 |
| [vllm-project/vllm#53972](https://github.com/vllm-project/vllm/pull/53972) | OPEN → `main` | \[Perf\]\[DSV4\] Use broadcast mHC pre for DeepSeek V4 DSpark | 2026-09-03 | CONFLICTING; DIRTY; REVIEW_REQUIRED; CANCELLED=4, FAILURE=2, SKIPPED=7, SUCCESS=8 |
| [tile-ai/tilelang#3156](https://github.com/tile-ai/tilelang/pull/3156) | OPEN → `main` | \[ROCm\] Support DeepSeek mHC pre and post kernels | 2026-09-03 | 未专项采样 |
| [vllm-project/vllm-ascend#15429](https://github.com/vllm-project/vllm-ascend/pull/15429) | OPEN → `main` | \[Perf\]\[Ops\] Fused mHC via AscendC npu_hc_pre_v2/npu_hc_post | 2026-09-03 | MERGEABLE; BLOCKED; REVIEW_REQUIRED; FAILURE=2, SKIPPED=6, SUCCESS=3 |
| [NVIDIA/Megatron-LM#7005](https://github.com/NVIDIA/Megatron-LM/pull/7005) | OPEN · draft → `dev` | \[dev\](feat): Support mHC recompute attention split capture range with THD format. | 2026-09-01 | 未专项采样 |
| [vllm-project/vllm-xpu-kernels#533](https://github.com/vllm-project/vllm-xpu-kernels/pull/533) | OPEN → `main` | fuse rmsnorm into mhc_pre & mhc_post_pre | 2026-09-01 | 未专项采样 |
| [tile-ai/tilelang-ascend#1643](https://github.com/tile-ai/tilelang-ascend/pull/1643) | OPEN → `ascendc_pto` | Add mhc_post example | 2026-08-31 | 未专项采样 |
| [sgl-project/sglang#36803](https://github.com/sgl-project/sglang/pull/36803) | OPEN → `main` | \[Intel\] \[XPU\] Restore XPU MHC dispatch | 2026-08-31 | CONFLICTING; DIRTY; REVIEW_REQUIRED; FAILURE=20, SKIPPED=109, SUCCESS=21 |
| [vllm-project/vllm#42735](https://github.com/vllm-project/vllm/pull/42735) | OPEN → `main` | \[Perf\]\[Kernel\] Use bf16 shared staging in mHC pre TileLang kernel | 2026-08-28 | CONFLICTING; DIRTY; REVIEW_REQUIRED; FAILURE=2, SKIPPED=1, SUCCESS=3 |
| [tile-ai/tilelang-ascend#1629](https://github.com/tile-ai/tilelang-ascend/pull/1629) | OPEN → `ascendc_pto` | Add mhc_pre example | 2026-08-24 | 未专项采样 |
| [vllm-project/vllm-ascend#14212](https://github.com/vllm-project/vllm-ascend/pull/14212) | OPEN · draft → `main` | Adapt telechat4 model and add mhc related operators | 2026-08-24 | 未专项采样 |
| [vllm-project/vllm#50645](https://github.com/vllm-project/vllm/pull/50645) | OPEN → `main` | \[Bugfix\] Guard mhc_pre_broadcast_tilelang on DeepGEMM support | 2026-08-23 | 未专项采样 |
| [NVIDIA/Megatron-LM#6699](https://github.com/NVIDIA/Megatron-LM/pull/6699) | OPEN → `dev` | \[Dev\] Fix DS4 lite train/inference consistency: mHC combine transpose and hash-route weights | 2026-08-21 | 未专项采样 |
| [vllm-project/vllm#52725](https://github.com/vllm-project/vllm/pull/52725) | OPEN · draft → `main` | fuse norm into mhc | 2026-08-20 | MERGEABLE; BLOCKED; REVIEW_REQUIRED; CANCELLED=2, FAILURE=2, SKIPPED=5, SUCCESS=6 |
| [vllm-project/vllm#49707](https://github.com/vllm-project/vllm/pull/49707) | OPEN → `main` | Fix NVIDIA DeepSeek V4 MHC warmup coverage | 2026-08-19 | 未专项采样 |
| [tile-ai/tilelang-ascend#1633](https://github.com/tile-ai/tilelang-ascend/pull/1633) | OPEN → `ascendc_pto` | \[Example\] Add mhc_bwd (Sinkhorn implicit CG) operator for Ascend NPU | 2026-08-19 | 未专项采样 |
| [vllm-project/vllm#48619](https://github.com/vllm-project/vllm/pull/48619) | OPEN → `main` | Port DeepGemm MHC Kernel to CuTeDSL | 2026-08-18 | CONFLICTING; DIRTY; REVIEW_REQUIRED; ACTION_REQUIRED=1, SKIPPED=1, SUCCESS=3 |
| [sgl-project/sgl-kernel-npu#698](https://github.com/sgl-project/sgl-kernel-npu/pull/698) | OPEN → `main` | Add mHC pre and post kernels for TeleChat4 | 2026-08-14 | 未专项采样 |
| [sgl-project/sglang#29740](https://github.com/sgl-project/sglang/pull/29740) | OPEN → `main` | Bugfix in tf32_hc_prenorm_gemm when SGLANG_ENABLE_JIT_DEEPGEMM=0 | 2026-08-12 | 未专项采样 |
| [sgl-project/sglang#34021](https://github.com/sgl-project/sglang/pull/34021) | OPEN → `main` | Fix DeepSeek-V4 mHC combine allocating transients in the NCCL symmetric pool | 2026-08-07 | CONFLICTING; DIRTY; REVIEW_REQUIRED; FAILURE=16, SKIPPED=85, SUCCESS=17 |
| [sgl-project/sgl-kernel-npu#670](https://github.com/sgl-project/sgl-kernel-npu/pull/670) | OPEN · draft → `main` | \[NPU\] Add AscendC mHC operators for TeleChat4 | 2026-08-07 | 未专项采样 |
| [tile-ai/tilelang-ascend#1425](https://github.com/tile-ai/tilelang-ascend/pull/1425) | OPEN → `ascendc_pto` | \[Add\]Add the mhc post kernel | 2026-08-04 | 未专项采样 |
| [sgl-project/sglang#32220](https://github.com/sgl-project/sglang/pull/32220) | OPEN → `main` | \[Spec\]\[DSV4\] perf: Compile draft hc_head during CUDA graph capture | 2026-08-03 | 未专项采样 |
| [sgl-project/sglang#32497](https://github.com/sgl-project/sglang/pull/32497) | OPEN → `main` | \[Spec\]\[DSV4\] perf: Compile draft-extend hc_head during CUDA graph capture | 2026-08-03 | 未专项采样 |
| [tile-ai/tilelang-ascend#1434](https://github.com/tile-ai/tilelang-ascend/pull/1434) | OPEN → `ascendc_pto` | \[Add\]Add the mhc expand kernel | 2026-08-03 | 未专项采样 |
| [tile-ai/tilelang-ascend#1432](https://github.com/tile-ai/tilelang-ascend/pull/1432) | OPEN → `ascendc_pto` | \[Add\]Add the mhc pre_big_fuse kernel | 2026-08-03 | 未专项采样 |
| [tile-ai/tilelang-ascend#1433](https://github.com/tile-ai/tilelang-ascend/pull/1433) | OPEN → `ascendc_pto` | \[Add\]Add the mhc multilayer_recompute kernel | 2026-08-03 | 未专项采样 |
| [ROCm/aiter#3613](https://github.com/ROCm/aiter/pull/3613) | OPEN · draft → `main` | \[Triton\] \[Gluon\] \[GFX12\] mHC_post_pre kernel | 2026-08-02 | CONFLICTING; DIRTY; FAILURE=1, SKIPPED=30, SUCCESS=2 |
| [ROCm/aiter#4279](https://github.com/ROCm/aiter/pull/4279) | OPEN · draft → `main` | mhc bf16 compute optimize on gfx12xx | 2026-08-02 | CONFLICTING; DIRTY; SKIPPED=10, SUCCESS=30 |
| [ROCm/aiter#4432](https://github.com/ROCm/aiter/pull/4432) | OPEN → `main` | fix: use residual.stride(1) for MHC HC-slice indexing | 2026-08-02 | CONFLICTING; DIRTY; SKIPPED=10, SUCCESS=30 |
| [tile-ai/tilelang-ascend#1412](https://github.com/tile-ai/tilelang-ascend/pull/1412) | OPEN → `ascendc_pto` | \[Add\]Add the mhc sinkhorn kernel | 2026-08-02 | 未专项采样 |
| [tile-ai/tilelang-ascend#1393](https://github.com/tile-ai/tilelang-ascend/pull/1393) | OPEN → `ascendc_pto` | \[Add\]Add the mhc pre_split_mixes kernel | 2026-07-30 | 未专项采样 |
| [deepseek-ai/TileKernels#23](https://github.com/deepseek-ai/TileKernels/pull/23) | OPEN → `main` | \[BugFix\]\[mhc\] Apply RMSNorm weight fusion in the no-grad mhc_pre path | 2026-07-28 | 未专项采样 |
| [tile-ai/tilelang-ascend#1409](https://github.com/tile-ai/tilelang-ascend/pull/1409) | OPEN → `ascendc_pto` | \[Add\]Add the mhc pre_apply_mix kernel | 2026-07-28 | 未专项采样 |
| [tile-ai/tilelang-ascend#1420](https://github.com/tile-ai/tilelang-ascend/pull/1420) | OPEN → `ascendc_pto` | \[Add\]Add the mhc norm_fn kernel | 2026-07-21 | 未专项采样 |
| [flashinfer-ai/flashinfer#4071](https://github.com/flashinfer-ai/flashinfer/pull/4071) | OPEN → `main` | feat(experimental): add flashinfer.experimental.sm12x | 2026-07-20 | 未专项采样 |
| [deepseek-ai/DeepGEMM#376](https://github.com/deepseek-ai/DeepGEMM/pull/376) | OPEN → `main` | Fix NVRTC compilation for HyperConnection kernels | 2026-07-11 | MERGEABLE; CLEAN; 无可见 checks |
| [sgl-project/sglang#30732](https://github.com/sgl-project/sglang/pull/30732) | OPEN → `main` | mhc: portable Triton hc_split_sinkhorn  fallback for non-TileLang bac… | 2026-07-10 | CONFLICTING; DIRTY; REVIEW_REQUIRED; FAILURE=15, SKIPPED=78, SUCCESS=16 |
| [flashinfer-ai/flashinfer#3460](https://github.com/flashinfer-ai/flashinfer/pull/3460) | OPEN → `main` | perf: improved performance of mhc_pre_big_fuse kernel on h100 | 2026-06-19 | MERGEABLE; BLOCKED; REVIEW_REQUIRED; SUCCESS=2 |
| [sgl-project/sglang#26787](https://github.com/sgl-project/sglang/pull/26787) | OPEN → `main` | Remove dead prewarm_mhc_token_count_buckets | 2026-05-30 | 未专项采样 |
| [deepseek-ai/TileKernels#10](https://github.com/deepseek-ai/TileKernels/pull/10) | OPEN → `main` | fix(mhc): MHCPreNormFn.backward returns wrong number of gradients | 2026-04-25 | 未专项采样 |
| [NVIDIA/TileGym#47](https://github.com/NVIDIA/TileGym/pull/47) | OPEN → `main` | \[Update\] A better sinkhorn implementation. | 2026-02-06 | 未专项采样 |

## NVIDIA-NeMo/Automodel

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#3855](https://github.com/NVIDIA-NeMo/Automodel/pull/3855) | OPEN | feat: add DeepSeek-V4.1-Flash model training support | `main` | — | 2026-09-11 |

## NVIDIA/Megatron-LM

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#7054](https://github.com/NVIDIA/Megatron-LM/pull/7054) | OPEN | GLM5.3 Flash (KDA + mHC + KPool DSA) support + FP8 | `main` | — | 2026-09-10 |
| [#7014](https://github.com/NVIDIA/Megatron-LM/pull/7014) | MERGED | \[Dev\] fix: Preserve offload events in Hybrid mHC CUDA graphs | `dev` | 2026-09-09 | 2026-09-09 |
| [#7005](https://github.com/NVIDIA/Megatron-LM/pull/7005) | OPEN · draft | \[dev\](feat): Support mHC recompute attention split capture range with THD format. | `dev` | — | 2026-09-01 |
| [#6699](https://github.com/NVIDIA/Megatron-LM/pull/6699) | OPEN | \[Dev\] Fix DS4 lite train/inference consistency: mHC combine transpose and hash-route weights | `dev` | — | 2026-08-21 |
| [#6661](https://github.com/NVIDIA/Megatron-LM/pull/6661) | MERGED | \[dev\](Fix): Make the mHC attention CUDA-graph split opt-in | `dev` | 2026-08-26 | 2026-08-26 |
| [#6401](https://github.com/NVIDIA/Megatron-LM/pull/6401) | MERGED | Add fused mHC support for HybridModel | `main` | 2026-08-27 | 2026-08-27 |
| [#6371](https://github.com/NVIDIA/Megatron-LM/pull/6371) | MERGED | Fix non-fused mHC projection for empty sequences | `dev` | 2026-08-16 | 2026-08-16 |
| [#6172](https://github.com/NVIDIA/Megatron-LM/pull/6172) | MERGED | \[Dev\] Keep the mHC mapping computation in fp32 on the fused cuTile path | `dev` | 2026-08-02 | 2026-08-02 |
| [#5994](https://github.com/NVIDIA/Megatron-LM/pull/5994) | MERGED | fix(mhc): update CUDA graph module API | `pull-request/4531` | 2026-08-10 | 2026-08-10 |
| [#5943](https://github.com/NVIDIA/Megatron-LM/pull/5943) | CLOSED · draft | feat(pipeline): carry mHC residual streams between stages | `pull-request/5936` | — | 2026-07-23 |
| [#5841](https://github.com/NVIDIA/Megatron-LM/pull/5841) | MERGED | \[Dev\] Support mHC selective recompute with CUDA graphs under EP a2a overlap | `dev` | 2026-08-18 | 2026-08-18 |
| [#5471](https://github.com/NVIDIA/Megatron-LM/pull/5471) | MERGED | \[dev\]: Fix mHC boundaries in EP overlap schedule | `dev` | 2026-07-22 | 2026-07-22 |
| [#5192](https://github.com/NVIDIA/Megatron-LM/pull/5192) | CLOSED · draft | fix(mHC): make HybridStack mHC wrapper CUDA-graph capturable | `dev` | — | 2026-06-29 |
| [#4949](https://github.com/NVIDIA/Megatron-LM/pull/4949) | MERGED | Add mHC support for HybridModel on dev | `dev` | 2026-05-27 | 2026-05-27 |
| [#4948](https://github.com/NVIDIA/Megatron-LM/pull/4948) | CLOSED · draft | \[Split 5/N of #3430\] feat(mHC): functional-test recipe | `main` | — | 2026-09-10 |
| [#4947](https://github.com/NVIDIA/Megatron-LM/pull/4947) | CLOSED · draft | \[Split 4/N of #3430\] feat(mHC): fused cuTile kernels | `main` | — | 2026-09-10 |
| [#4946](https://github.com/NVIDIA/Megatron-LM/pull/4946) | CLOSED · draft | \[Split 3/N of #3430\] feat(mHC): pipeline-parallel support | `main` | — | 2026-09-10 |
| [#4945](https://github.com/NVIDIA/Megatron-LM/pull/4945) | CLOSED · draft | \[Split 2/N of #3430\] feat(mHC): GPT model wiring | `main` | — | 2026-09-10 |
| [#4898](https://github.com/NVIDIA/Megatron-LM/pull/4898) | CLOSED | test: skip cuTile fused mHC tests on Hopper | `dev` | — | 2026-05-22 |
| [#4869](https://github.com/NVIDIA/Megatron-LM/pull/4869) | CLOSED · draft | \[main\] \[DeepSeek-v4\] MTP support with mHC and new mHC contract | `main` | — | 2026-07-30 |
| [#4817](https://github.com/NVIDIA/Megatron-LM/pull/4817) | MERGED | \[Dev\]\[opt\] Optimize e_proj and h_proj TP communication for MTP with mHC | `dev` | 2026-05-18 | 2026-05-18 |
| [#4624](https://github.com/NVIDIA/Megatron-LM/pull/4624) | MERGED | \[dev\]: faster implementation of mHC fused kernels | `dev` | 2026-06-12 | 2026-08-06 |
| [#4568](https://github.com/NVIDIA/Megatron-LM/pull/4568) | CLOSED | Add mHC support for HybridModel on dsv4 | `dsv4` | — | 2026-05-01 |
| [#4567](https://github.com/NVIDIA/Megatron-LM/pull/4567) | CLOSED | Add mHC support for HybridModel on dsv4 | `dsv4` | — | 2026-05-01 |
| [#4531](https://github.com/NVIDIA/Megatron-LM/pull/4531) | MERGED | \[Split 1/N of #3430\] feat(mHC): basic pytorch implementation of manifold hyper connection | `main` | 2026-08-18 | 2026-08-18 |
| [#4530](https://github.com/NVIDIA/Megatron-LM/pull/4530) | CLOSED · draft | Add mHC functional CI coverage (stacks on #4483) | `dsv4` | — | 2026-04-29 |
| [#4529](https://github.com/NVIDIA/Megatron-LM/pull/4529) | CLOSED · draft | Add mHC support for HybridModel on dsv4 (stacks on #4483) | `dsv4` | — | 2026-04-29 |
| [#4528](https://github.com/NVIDIA/Megatron-LM/pull/4528) | CLOSED · draft | Add pipeline-parallel mHC compatibility (stacks on #4483) | `dsv4` | — | 2026-04-29 |
| [#4527](https://github.com/NVIDIA/Megatron-LM/pull/4527) | CLOSED · draft | Add fused mHC cuTile kernels (stacks on #4483) | `dsv4` | — | 2026-04-29 |
| [#4518](https://github.com/NVIDIA/Megatron-LM/pull/4518) | MERGED | \[dev\] \[DeepSeek-v4\] Part 3: MTP support with mHC and new mHC contract | `dev` | 2026-05-14 | 2026-05-14 |
| [#4483](https://github.com/NVIDIA/Megatron-LM/pull/4483) | CLOSED | Add mHC transformer reference implementation | `dsv4` | — | 2026-05-01 |
| [#4469](https://github.com/NVIDIA/Megatron-LM/pull/4469) | MERGED | Add mHC support for HybridModel on dsv4 | `dsv4` | 2026-05-04 | 2026-05-04 |
| [#4190](https://github.com/NVIDIA/Megatron-LM/pull/4190) | MERGED | \[dev\] fix: Support mHC with cuda graph and activation offloading | `dev` | 2026-04-10 | 2026-04-10 |
| [#3828](https://github.com/NVIDIA/Megatron-LM/pull/3828) | MERGED | \[dev\] mHC kernel fusion | `dev` | 2026-03-25 | 2026-03-25 |
| [#3430](https://github.com/NVIDIA/Megatron-LM/pull/3430) | CLOSED | Feature: Support of Manifold Hyper Connection(mHC) | `main` | — | 2026-05-12 |
| [#2943](https://github.com/NVIDIA/Megatron-LM/pull/2943) | MERGED | \[dev\] feat(mHC): Add basic pytorch implementation of manifold hyper connection(mHC). | `dev` | 2026-03-06 | 2026-03-06 |

## NVIDIA/TensorRT-LLM

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#18331](https://github.com/NVIDIA/TensorRT-LLM/pull/18331) | MERGED | \[TRTLLM-15316\]\[feat\] Fix fused mHC Phase-4 coherence and support uneven split-K | `main` | 2026-09-02 | 2026-09-02 |
| [#17997](https://github.com/NVIDIA/TensorRT-LLM/pull/17997) | CLOSED · draft | \[None\]\[feat\] add TeleChat4 model support with mHC, reasoning and tool parsers | `main` | — | 2026-09-01 |
| [#17801](https://github.com/NVIDIA/TensorRT-LLM/pull/17801) | MERGED | \[https://nvbugs/6567554\]\[fix\] Make DeepSeek-V4 layer-wise benchmarks run, and derive module perf cases from the trace | `main` | 2026-08-19 | 2026-08-19 |
| [#16799](https://github.com/NVIDIA/TensorRT-LLM/pull/16799) | MERGED | \[None\]\[perf\] Optimize Blackwell fused MHC half-MMA kernel | `main` | 2026-07-27 | 2026-08-27 |
| [#16774](https://github.com/NVIDIA/TensorRT-LLM/pull/16774) | MERGED | \[None\]\[feat\] Support DeepSeek-V4 in layer_wise_benchmarks | `main` | 2026-07-26 | 2026-07-26 |
| [#16221](https://github.com/NVIDIA/TensorRT-LLM/pull/16221) | MERGED | \[None\]\[fix\] Fix fused mHC output reuse and extend compressor next_n | `main` | 2026-07-13 | 2026-07-13 |
| [#16176](https://github.com/NVIDIA/TensorRT-LLM/pull/16176) | CLOSED | \[https://nvbugs/6430358\]\[fix\] mhc: make _FusedHcWorkspaceCache CUDA-graph-safe | `main` | — | 2026-07-12 |
| [#16175](https://github.com/NVIDIA/TensorRT-LLM/pull/16175) | CLOSED | \[None\]\[fix\] mhc: make _FusedHcWorkspaceCache CUDA-graph-safe | `feat/deepseek_v4` | — | 2026-07-09 |
| [#15626](https://github.com/NVIDIA/TensorRT-LLM/pull/15626) | MERGED | \[None\]\[perf\] DSv4 follow-up: autotuner updates | `main` | 2026-06-29 | 2026-06-29 |
| [#15414](https://github.com/NVIDIA/TensorRT-LLM/pull/15414) | MERGED | \[None\]\[feat\] DSv4: model, tokenizer, and integration coverage | `main` | 2026-06-28 | 2026-06-30 |
| [#15379](https://github.com/NVIDIA/TensorRT-LLM/pull/15379) | MERGED | \[None\]\[feat\] DSv4 prep: compressor and mHC primitives | `main` | 2026-06-24 | 2026-06-24 |
| [#14675](https://github.com/NVIDIA/TensorRT-LLM/pull/14675) | MERGED | \[None\]\[fix\] mhc: disable PARALLEL distributed tuning for MhcFusedHc | `feat/deepseek_v4` | 2026-05-28 | 2026-05-28 |
| [#14458](https://github.com/NVIDIA/TensorRT-LLM/pull/14458) | MERGED | \[None\]\[perf\] Autotuner: distributed tuning for mhc + single-(runner,tactic) shortcut | `feat/deepseek_v4` | 2026-05-28 | 2026-05-28 |
| [#14329](https://github.com/NVIDIA/TensorRT-LLM/pull/14329) | MERGED | \[None\]\[chore\] Reduce mhc autotune tactics | `feat/deepseek_v4` | 2026-05-22 | 2026-05-22 |
| [#14310](https://github.com/NVIDIA/TensorRT-LLM/pull/14310) | CLOSED · draft | \[None\]\[perf\] Enable PDL for MHC BigFuse kernels | `feat/deepseek_v4` | — | 2026-07-09 |
| [#14305](https://github.com/NVIDIA/TensorRT-LLM/pull/14305) | CLOSED | \[None\]\[chore\] Route MHC fused-HC workspace through global Buffers pool | `feat/deepseek_v4` | — | 2026-07-09 |
| [#14053](https://github.com/NVIDIA/TensorRT-LLM/pull/14053) | CLOSED | \[None\]\[perf\] Add TRTLLM_DSV4_MEM_OPTS master switch for DSv4 memory optimizations | `feat/deepseek_v4` | — | 2026-07-21 |
| [#13961](https://github.com/NVIDIA/TensorRT-LLM/pull/13961) | CLOSED · draft | \[None\]\[perf\] Workaround: route DSv4-Pro hidden=7168 eager-cache-miss fallback to FMA to fit the bench within wall budget | `feat/deepseek_v4` | — | 2026-05-11 |
| [#13892](https://github.com/NVIDIA/TensorRT-LLM/pull/13892) | MERGED | \[None\]\[perf\] mHC fused_hc kernel optimizations + DS-V4 entry-boundary RMSNorm fold-in | `feat/deepseek_v4` | 2026-05-12 | 2026-05-12 |
| [#13771](https://github.com/NVIDIA/TensorRT-LLM/pull/13771) | MERGED | \[None\]\[fix\] Fix fused MHC for DeepSeek-V4-Pro hidden size | `feat/deepseek_v4` | 2026-05-05 | 2026-05-07 |
| [#13710](https://github.com/NVIDIA/TensorRT-LLM/pull/13710) | CLOSED | \[None\]\[fix\] Fix fused MHC for DeepSeek-V4-Pro hidden size | `feat/deepseek_v4` | — | 2026-05-05 |
| [#13660](https://github.com/NVIDIA/TensorRT-LLM/pull/13660) | MERGED | \[TRTLLM-12383\]\[fix\] limit MHC TF32 pmap GEMM to SM100 | `feat/deepseek_v4` | 2026-04-30 | 2026-04-30 |
| [#13611](https://github.com/NVIDIA/TensorRT-LLM/pull/13611) | MERGED | \[None\]\[fix\]  Remove MHC fused hidden-size guard | `feat/deepseek_v4` | 2026-04-29 | 2026-04-29 |
| [#13587](https://github.com/NVIDIA/TensorRT-LLM/pull/13587) | MERGED | \[None\]\[fix\] Fix fused mHC RMS normalization | `feat/deepseek_v4` | 2026-04-29 | 2026-04-29 |

## NVIDIA/TileGym

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#87](https://github.com/NVIDIA/TileGym/pull/87) | CLOSED | Fix random mhc test and benchmark failures & Add unsloth geglu and grouped_gemm kernels (#84) | `main` | — | 2026-03-28 |
| [#84](https://github.com/NVIDIA/TileGym/pull/84) | MERGED | Fix random mhc test and benchmark failures & Add unsloth geglu and grouped_gemm kernels | `main` | 2026-03-26 | 2026-03-26 |
| [#47](https://github.com/NVIDIA/TileGym/pull/47) | OPEN | \[Update\] A better sinkhorn implementation. | `main` | — | 2026-02-06 |
| [#38](https://github.com/NVIDIA/TileGym/pull/38) | MERGED | Add mHC fused kernels and tests | `main` | 2026-01-26 | 2026-01-26 |

## NVIDIA/TransformerEngine

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#3442](https://github.com/NVIDIA/TransformerEngine/pull/3442) | MERGED | \[Common\] Reduce default cfg's SMEM usage for mhc triton kernel | `main` | 2026-09-01 | 2026-09-01 |
| [#2978](https://github.com/NVIDIA/TransformerEngine/pull/2978) | MERGED | \[Common, PyTorch\] Improve mHC to match DeepSeek's implementation | `main` | 2026-07-29 | 2026-07-29 |
| [#2953](https://github.com/NVIDIA/TransformerEngine/pull/2953) | CLOSED · draft | \[Common, PyTorch\] Improve mHC to match DeepSeek's implementation | `main` | — | 2026-05-12 |
| [#2790](https://github.com/NVIDIA/TransformerEngine/pull/2790) | MERGED | \[Common, PyTorch\] Add triton mHC kernels & pytorch APIs | `main` | 2026-04-28 | 2026-04-28 |

## ROCm/ATOM

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#2195](https://github.com/ROCm/ATOM/pull/2195) | OPEN | feat: enable BF16 mHC computation and fix tracing copies | `main` | — | 2026-09-11 |
| [#2194](https://github.com/ROCm/ATOM/pull/2194) | CLOSED | feat: enable BF16 mHC computation and fix tracing copies | `main` | — | 2026-09-10 |
| [#1140](https://github.com/ROCm/ATOM/pull/1140) | MERGED | perf(v4): Use hc_state to transfer hidden_state and residual between layers for using mhc_post/pre fused | `main` | 2026-06-10 | 2026-06-10 |
| [#1102](https://github.com/ROCm/ATOM/pull/1102) | MERGED | fix mhc fallback | `main` | 2026-06-05 | 2026-06-05 |

## ROCm/aiter

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#5412](https://github.com/ROCm/aiter/pull/5412) | OPEN | \[HIP\] \[JIT\] feat: add packed BF16 mHC computation and gfx1250 tuning | `main` | — | 2026-09-11 |
| [#5408](https://github.com/ROCm/aiter/pull/5408) | CLOSED | \[HIP\] \[JIT\] feat: add packed BF16 mHC computation and gfx1250 tuning | `main` | — | 2026-09-10 |
| [#5296](https://github.com/ROCm/aiter/pull/5296) | OPEN | \[HIP\] perf(mhc): cap the split-k partial-reduction width in mhc_pre_big_fuse | `main` | — | 2026-09-08 |
| [#5245](https://github.com/ROCm/aiter/pull/5245) | MERGED | \[HIP\] Add data-init to test_mhc | `gfx1250/microbench` | 2026-09-03 | 2026-09-03 |
| [#5021](https://github.com/ROCm/aiter/pull/5021) | MERGED | \[Triton/Gluon\] Migrate the MHC tuned configs to the nested layout | `main` | 2026-08-27 | 2026-08-27 |
| [#4832](https://github.com/ROCm/aiter/pull/4832) | MERGED | \[HIP\] \[module_fused_ar_mhc\] detorch fused_ar_mhc_post + pybind | `main` | 2026-08-19 | 2026-08-19 |
| [#4432](https://github.com/ROCm/aiter/pull/4432) | OPEN | fix: use residual.stride(1) for MHC HC-slice indexing | `main` | — | 2026-08-02 |
| [#4279](https://github.com/ROCm/aiter/pull/4279) | OPEN · draft | mhc bf16 compute optimize on gfx12xx | `main` | — | 2026-08-02 |
| [#4115](https://github.com/ROCm/aiter/pull/4115) | MERGED | Optimize mhc on gfx12xx | `main` | 2026-07-07 | 2026-07-07 |
| [#3796](https://github.com/ROCm/aiter/pull/3796) | MERGED | hip mhc : fix async waitcnt and  tune gfx12xx config | `main` | 2026-06-18 | 2026-06-18 |
| [#3791](https://github.com/ROCm/aiter/pull/3791) | MERGED | Add fused custom AllReduce + MHC post for TP paths | `main` | 2026-06-18 | 2026-06-18 |
| [#3781](https://github.com/ROCm/aiter/pull/3781) | CLOSED | perf: use vectorized LDS loads for mhc_pre_gemm_sqrsum on gfx942 | `main` | — | 2026-08-11 |
| [#3768](https://github.com/ROCm/aiter/pull/3768) | MERGED | simple support hip mhc on wave32 platform | `main` | 2026-06-17 | 2026-06-17 |
| [#3726](https://github.com/ROCm/aiter/pull/3726) | CLOSED | Add fused custom AllReduce + MHC post/pre + RMSNorm for TP paths | `main` | — | 2026-06-18 |
| [#3714](https://github.com/ROCm/aiter/pull/3714) | MERGED | fix(mhc): pass i_os=0 in the async_load callsite #3648 missed | `main` | 2026-06-14 | 2026-06-14 |
| [#3651](https://github.com/ROCm/aiter/pull/3651) | MERGED | Mhc large m | `main` | 2026-06-12 | 2026-06-12 |
| [#3623](https://github.com/ROCm/aiter/pull/3623) | MERGED | Add hip mhc_fused_post_pre | `main` | 2026-06-10 | 2026-06-10 |
| [#3613](https://github.com/ROCm/aiter/pull/3613) | OPEN · draft | \[Triton\] \[Gluon\] \[GFX12\] mHC_post_pre kernel | `main` | — | 2026-08-02 |
| [#3536](https://github.com/ROCm/aiter/pull/3536) | CLOSED | Add HIP fused mHC post-pre path | `main` | — | 2026-06-12 |
| [#3417](https://github.com/ROCm/aiter/pull/3417) | MERGED | Fix mhc_pre_big_fuse accuracy issue in rocm7.2.3 | `main` | 2026-05-29 | 2026-05-29 |
| [#3411](https://github.com/ROCm/aiter/pull/3411) | MERGED | \[Triton\] mhc fix | `main` | 2026-05-29 | 2026-05-29 |
| [#3396](https://github.com/ROCm/aiter/pull/3396) | MERGED | Add add mhc_pre fused rmsnorm | `main` | 2026-05-28 | 2026-05-28 |
| [#3305](https://github.com/ROCm/aiter/pull/3305) | MERGED | \[Triton\] update mHC_pre_post | `main` | 2026-05-21 | 2026-05-21 |
| [#3278](https://github.com/ROCm/aiter/pull/3278) | MERGED | \[bugfix\] fix mhc split-k acc_sq mask | `main` | 2026-05-21 | 2026-05-21 |
| [#3237](https://github.com/ROCm/aiter/pull/3237) | MERGED | Optimize mHC hip kernel performce(Adjust fp32 gemm pipeline and add NT cache policy to memory bound case) | `main` | 2026-05-18 | 2026-05-18 |
| [#3059](https://github.com/ROCm/aiter/pull/3059) | MERGED | Add mhc_pre to custom_op | `main` | 2026-05-07 | 2026-05-07 |
| [#3044](https://github.com/ROCm/aiter/pull/3044) | MERGED | Update mhc_pre hip kernel support hc_head | `main` | 2026-05-06 | 2026-05-06 |
| [#3033](https://github.com/ROCm/aiter/pull/3033) | MERGED | Fix sqrsum store race condition in mhc_pre_gemm_sqrsum_kernel | `main` | 2026-05-06 | 2026-05-06 |
| [#2978](https://github.com/ROCm/aiter/pull/2978) | MERGED | Fix mhc_pre_big_fuse's accuracy because of loading synchronization | `main` | 2026-05-01 | 2026-05-01 |
| [#2967](https://github.com/ROCm/aiter/pull/2967) | MERGED | \[TRITON\] mHC-post: Apply post-stream and res-stream mixing | `main` | 2026-05-13 | 2026-05-13 |
| [#2963](https://github.com/ROCm/aiter/pull/2963) | MERGED | Fix mhc_post accuracy and optimize perfmance on small M | `main` | 2026-04-30 | 2026-04-30 |
| [#2916](https://github.com/ROCm/aiter/pull/2916) | MERGED | fix mhc device | `main` | 2026-04-28 | 2026-04-28 |
| [#2915](https://github.com/ROCm/aiter/pull/2915) | MERGED | mHC: Optimize mhc_pre performance in small M | `main` | 2026-04-25 | 2026-04-25 |
| [#2646](https://github.com/ROCm/aiter/pull/2646) | MERGED | \[TRITON\] mHC-pre: Manifold-constrained Hyper Connection | `main` | 2026-05-11 | 2026-05-11 |
| [#2479](https://github.com/ROCm/aiter/pull/2479) | MERGED | mhc：add mhc_post hip kernel | `main` | 2026-03-30 | 2026-03-30 |
| [#2168](https://github.com/ROCm/aiter/pull/2168) | MERGED | fix  mhc build | `main` | 2026-03-04 | 2026-03-04 |
| [#2136](https://github.com/ROCm/aiter/pull/2136) | MERGED | add mhc_pre hip kernel (mhc_pre_gemm_sqrsum, mhc_pre_big_fuse) | `main` | 2026-03-03 | 2026-03-03 |
| [#1877](https://github.com/ROCm/aiter/pull/1877) | MERGED | Refactor mHC kernel and wrapper to implement equations 14-18 as fused kernel | `feat/mhc-deepseek` | 2026-01-20 | 2026-01-20 |
| [#1859](https://github.com/ROCm/aiter/pull/1859) | CLOSED | \[TRITON\] mHC/mHC-lite: Manifold-constrained Hyper Connection | `main` | — | 2026-04-07 |

## ROCm/rocm-libraries

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#5744](https://github.com/ROCm/rocm-libraries/pull/5744) | CLOSED · draft | Users/damien lejeune/ck/mhc core | `develop` | — | 2026-05-09 |

## ai-dynamo/aiconfigurator

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#1508](https://github.com/ai-dynamo/aiconfigurator/pull/1508) | MERGED | fix(aic-core): DSV4 CSA CP parity — mHC seq_split, reuse-aware top_last, mixed-pass filters (Phase 2 PR-2.5) | `main` | 2026-08-13 | 2026-08-13 |
| [#1486](https://github.com/ai-dynamo/aiconfigurator/pull/1486) | MERGED | feat: TRT-LLM DeepSeek-V4 collectors — mHC + CSA/HCA attention modules (#1480) | `main` | 2026-08-20 | 2026-08-20 |
| [#1349](https://github.com/ai-dynamo/aiconfigurator/pull/1349) | CLOSED · draft | Add GB300 vLLM mHC perf data | `main` | — | 2026-08-04 |
| [#942](https://github.com/ai-dynamo/aiconfigurator/pull/942) | MERGED | feat: deepseekv4 mhc collect and query | `main` | 2026-04-30 | 2026-04-30 |

## alibaba/rtp-llm

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#1371](https://github.com/alibaba/rtp-llm/pull/1371) | OPEN | feat(dsv4): integrate MegaKernel decode | `feat/dsv4_on_dev` | — | 2026-09-11 |

## deepseek-ai/DeepGEMM

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#432](https://github.com/deepseek-ai/DeepGEMM/pull/432) | MERGED | Public Release 26/09 | `main` | 2026-09-10 | 2026-09-10 |
| [#384](https://github.com/deepseek-ai/DeepGEMM/pull/384) | MERGED | Sync nv_dev with upstream #377 | `nv_dev` | 2026-07-20 | 2026-07-20 |
| [#376](https://github.com/deepseek-ai/DeepGEMM/pull/376) | OPEN | Fix NVRTC compilation for HyperConnection kernels | `main` | — | 2026-07-11 |
| [#319](https://github.com/deepseek-ai/DeepGEMM/pull/319) | CLOSED | Implement SM120 kernel for tf32_hc_prenorm_gemm | `main` | — | 2026-04-29 |

## deepseek-ai/TileKernels

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#23](https://github.com/deepseek-ai/TileKernels/pull/23) | OPEN | \[BugFix\]\[mhc\] Apply RMSNorm weight fusion in the no-grad mhc_pre path | `main` | — | 2026-07-28 |
| [#18](https://github.com/deepseek-ai/TileKernels/pull/18) | CLOSED | Add CUDA MHC inference kernels for SM 90 (Hopper) | `main` | — | 2026-06-02 |
| [#10](https://github.com/deepseek-ai/TileKernels/pull/10) | OPEN | fix(mhc): MHCPreNormFn.backward returns wrong number of gradients | `main` | — | 2026-04-25 |

## flashinfer-ai/flashinfer

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#4071](https://github.com/flashinfer-ai/flashinfer/pull/4071) | OPEN | feat(experimental): add flashinfer.experimental.sm12x | `main` | — | 2026-07-20 |
| [#3460](https://github.com/flashinfer-ai/flashinfer/pull/3460) | OPEN | perf: improved performance of mhc_pre_big_fuse kernel on h100 | `main` | — | 2026-06-19 |
| [#3285](https://github.com/flashinfer-ai/flashinfer/pull/3285) | MERGED | Add mHC post mapping and pre big-fuse kernels | `main` | 2026-05-29 | 2026-05-29 |

## lightseekorg/tokenspeed

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#30](https://github.com/lightseekorg/tokenspeed/pull/30) | MERGED | feat(deepseek-v4): add mega_moe and compressed KV perf path | `main` | 2026-05-09 | 2026-05-09 |

## sgl-project/DeepGEMM

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#86](https://github.com/sgl-project/DeepGEMM/pull/86) | OPEN · draft | Merge upstream main (Public Release 26/09) into dev | `dev` | — | 2026-09-11 |
| [#61](https://github.com/sgl-project/DeepGEMM/pull/61) | MERGED | Cover SM120 GPUs in CI test | `dev` | 2026-07-20 | 2026-07-20 |

## sgl-project/sgl-kernel-npu

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#698](https://github.com/sgl-project/sgl-kernel-npu/pull/698) | OPEN | Add mHC pre and post kernels for TeleChat4 | `main` | — | 2026-08-14 |
| [#670](https://github.com/sgl-project/sgl-kernel-npu/pull/670) | OPEN · draft | \[NPU\] Add AscendC mHC operators for TeleChat4 | `main` | — | 2026-08-07 |

## sgl-project/sgl-kernel-xpu

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#302](https://github.com/sgl-project/sgl-kernel-xpu/pull/302) | MERGED | Implement mhc_fused_post_pre with SYCL | `main` | 2026-08-19 | 2026-08-19 |
| [#199](https://github.com/sgl-project/sgl-kernel-xpu/pull/199) | MERGED | MHC Pre Big Fuse Kernel | `main` | 2026-06-11 | 2026-06-11 |
| [#186](https://github.com/sgl-project/sgl-kernel-xpu/pull/186) | MERGED | HC Split Sinkhorn Kernel | `main` | 2026-05-13 | 2026-05-13 |

## sgl-project/sglang

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#38952](https://github.com/sgl-project/sglang/pull/38952) | MERGED | Guard hc_split_sinkhorn against DP attention's empty idle batch | `dsv4.1` | 2026-09-10 | 2026-09-10 |
| [#38798](https://github.com/sgl-project/sglang/pull/38798) | OPEN | \[Model\] Add DeepSeek V4.1 support | `main` | — | 2026-09-11 |
| [#38545](https://github.com/sgl-project/sglang/pull/38545) | OPEN | \[AMD\] \[GLM-5.3-Flash Day 0\] Route mHC through AITER on gfx950 | `main` | — | 2026-09-10 |
| [#38475](https://github.com/sgl-project/sglang/pull/38475) | OPEN | \[DSV4\] Use mhc_pre's split-K pre-norm GEMM in the mhc_fused_post_pre fallback | `main` | — | 2026-09-08 |
| [#38442](https://github.com/sgl-project/sglang/pull/38442) | OPEN | Fix DeepGEMM import for independently enabled MHC prenorm | `main` | — | 2026-09-08 |
| [#37626](https://github.com/sgl-project/sglang/pull/37626) | CLOSED | \[AMD\] \[GLM-5.3-Flash Day 0\] Route mHC through AITER on gfx950 | `xinyuan/glm-5.3-flash-support` | — | 2026-09-06 |
| [#37375](https://github.com/sgl-project/sglang/pull/37375) | CLOSED | fix(model): align GLM5 mHC pipeline proxy contract | `xinyuan/glm-5.3-flash-support` | — | 2026-09-07 |
| [#37168](https://github.com/sgl-project/sglang/pull/37168) | OPEN | Prevent full CUDA-graph serving from corrupting MHC and DSA tensors | `main` | — | 2026-09-08 |
| [#37056](https://github.com/sgl-project/sglang/pull/37056) | CLOSED · draft | \[AMD\] Enable AITER mHC on gfx942 | `xinyuan/glm-5.3-flash-support` | — | 2026-08-29 |
| [#36884](https://github.com/sgl-project/sglang/pull/36884) | MERGED | fix(glm): mirror should_use_dp_reduce_scatterv() into the MHC communicator | `xinyuan/glm-5.3-flash-support` | 2026-08-28 | 2026-08-28 |
| [#36817](https://github.com/sgl-project/sglang/pull/36817) | CLOSED | mhc tilelang bugfix | `main` | — | 2026-08-28 |
| [#36803](https://github.com/sgl-project/sglang/pull/36803) | OPEN | \[Intel\] \[XPU\] Restore XPU MHC dispatch | `main` | — | 2026-08-31 |
| [#36755](https://github.com/sgl-project/sglang/pull/36755) | MERGED | Fix DFLASH aux hidden-state capture on mHC models | `xinyuan/glm-5.3-flash-support` | 2026-08-28 | 2026-08-28 |
| [#36507](https://github.com/sgl-project/sglang/pull/36507) | MERGED | GLM-5.3-Flash support | `main` | 2026-09-06 | 2026-09-07 |
| [#35214](https://github.com/sgl-project/sglang/pull/35214) | MERGED | \[DSV4\] Turn on mhc post pre fusion by default | `main` | 2026-08-18 | 2026-08-18 |
| [#35118](https://github.com/sgl-project/sglang/pull/35118) | MERGED | \[DSV4\] hc-prenorm: fuse the combine step into a Triton kernel | `main` | 2026-09-01 | 2026-09-01 |
| [#34021](https://github.com/sgl-project/sglang/pull/34021) | OPEN | Fix DeepSeek-V4 mHC combine allocating transients in the NCCL symmetric pool | `main` | — | 2026-08-07 |
| [#34019](https://github.com/sgl-project/sglang/pull/34019) | MERGED | \[SM12x\] Default the fused MHC post+pre path on | `main` | 2026-08-14 | 2026-08-14 |
| [#33972](https://github.com/sgl-project/sglang/pull/33972) | CLOSED · draft | \[NPU\] Add TeleChat4 support with fused mHC backends | `main` | — | 2026-08-10 |
| [#33616](https://github.com/sgl-project/sglang/pull/33616) | MERGED | feat: Add flashinfer mHC fusion for DSV4 | `main` | 2026-08-07 | 2026-08-07 |
| [#32577](https://github.com/sgl-project/sglang/pull/32577) | MERGED | \[AMD\] DeepSeek-V4: add aiter fused mHC post+pre with cross-layer boundary dispatch | `main` | 2026-08-22 | 2026-08-22 |
| [#32497](https://github.com/sgl-project/sglang/pull/32497) | OPEN | \[Spec\]\[DSV4\] perf: Compile draft-extend hc_head during CUDA graph capture | `main` | — | 2026-08-03 |
| [#32220](https://github.com/sgl-project/sglang/pull/32220) | OPEN | \[Spec\]\[DSV4\] perf: Compile draft hc_head during CUDA graph capture | `main` | — | 2026-08-03 |
| [#32166](https://github.com/sgl-project/sglang/pull/32166) | MERGED | \[XPU\] Use SYCL kernels for DeepSeek V4 MHC on XPU | `main` | 2026-08-25 | 2026-08-25 |
| [#30954](https://github.com/sgl-project/sglang/pull/30954) | MERGED | \[SM120\] Allow fused MHC opt-in with standalone TileLang pre disabled | `main` | 2026-07-26 | 2026-07-26 |
| [#30741](https://github.com/sgl-project/sglang/pull/30741) | MERGED | Prewarm DSV4 MHC post kernel at model load | `main` | 2026-08-04 | 2026-08-04 |
| [#30732](https://github.com/sgl-project/sglang/pull/30732) | OPEN | mhc: portable Triton hc_split_sinkhorn  fallback for non-TileLang bac… | `main` | — | 2026-07-10 |
| [#30580](https://github.com/sgl-project/sglang/pull/30580) | MERGED | Lazy load TileLang MHC kernels | `main` | 2026-07-12 | 2026-07-12 |
| [#29988](https://github.com/sgl-project/sglang/pull/29988) | MERGED | \[dsv4\] Trigger MHC prenorm prewarm at weight-load time with rank sync | `main` | 2026-07-03 | 2026-08-25 |
| [#29927](https://github.com/sgl-project/sglang/pull/29927) | MERGED | \[SM120\] DeepSeek-V4: DeepGEMM paged-MQA indexer +FP4 MoE+ page-split | `main` | 2026-09-02 | 2026-09-02 |
| [#29740](https://github.com/sgl-project/sglang/pull/29740) | OPEN | Bugfix in tf32_hc_prenorm_gemm when SGLANG_ENABLE_JIT_DEEPGEMM=0 | `main` | — | 2026-08-12 |
| [#27986](https://github.com/sgl-project/sglang/pull/27986) | MERGED | \[dsv4\] Prewarm MHC prenorm kernel at startup | `main` | 2026-06-15 | 2026-08-25 |
| [#27783](https://github.com/sgl-project/sglang/pull/27783) | MERGED | \[Intel GPU\] DeepSeek V4 3/N: Support hc_split_sinkhorn on XPU using sgl_kernel | `main` | 2026-06-26 | 2026-07-13 |
| [#27781](https://github.com/sgl-project/sglang/pull/27781) | MERGED | \[CI\] Move misplaced mhc kernel test into test/registered/kernels | `main` | 2026-06-10 | 2026-06-10 |
| [#27729](https://github.com/sgl-project/sglang/pull/27729) | CLOSED · draft | Warm DeepGEMM MHC prenorm split buckets | `main` | — | 2026-08-25 |
| [#27566](https://github.com/sgl-project/sglang/pull/27566) | CLOSED | \[DeepSeek V4\] optimize hc_head() for DeepSeekV4ModelNextN | `main` | — | 2026-07-25 |
| [#26843](https://github.com/sgl-project/sglang/pull/26843) | CLOSED · draft | Fix DeepSeek V4 NextN fused mHC guard | `main` | — | 2026-05-31 |
| [#26787](https://github.com/sgl-project/sglang/pull/26787) | OPEN | Remove dead prewarm_mhc_token_count_buckets | `main` | — | 2026-05-30 |
| [#26238](https://github.com/sgl-project/sglang/pull/26238) | MERGED | refactor(dsv4): route MHC prenorm through DeepGEMM wrapper | `main` | 2026-05-28 | 2026-08-25 |
| [#26071](https://github.com/sgl-project/sglang/pull/26071) | MERGED | \[Cherry-pick to release/v0.5.12\] perf(dsv4): add MHC token-count prewarm (#25810) | `release/v0.5.12` | 2026-05-22 | 2026-05-22 |
| [#26014](https://github.com/sgl-project/sglang/pull/26014) | MERGED | amd/deepseek_v4 31/N enable Triton fused mhc_post_pre for low concurrency | `amd/deepseek_v4` | 2026-05-22 | 2026-05-22 |
| [#25976](https://github.com/sgl-project/sglang/pull/25976) | MERGED | \[DeepSeek-V4\] Add mhc_fused_post_pre kernel | `main` | 2026-05-30 | 2026-05-30 |
| [#25810](https://github.com/sgl-project/sglang/pull/25810) | MERGED | perf(dsv4): add MHC token-count prewarm | `main` | 2026-05-21 | 2026-08-25 |
| [#24775](https://github.com/sgl-project/sglang/pull/24775) | MERGED | Optimize MHC pipeline: DeepGemm, fused norm, fused hc_head | `main` | 2026-05-10 | 2026-05-10 |
| [#24665](https://github.com/sgl-project/sglang/pull/24665) | MERGED | \[AMD\] Cherry-pick aiter commit for mhc_pre fix | `main` | 2026-05-08 | 2026-05-08 |
| [#24438](https://github.com/sgl-project/sglang/pull/24438) | MERGED | use deepgemm tf32_hc_prenorm_gemm for mhc_pre | `deepseek_v4_dev` | 2026-05-05 | 2026-05-05 |
| [#24355](https://github.com/sgl-project/sglang/pull/24355) | MERGED | amd/deepseek_v4 integration 10/N optimize mhc performance | `amd/deepseek_v4` | 2026-05-04 | 2026-05-04 |
| [#23882](https://github.com/sgl-project/sglang/pull/23882) | MERGED | Deepseek V4 | `main` | 2026-05-08 | 2026-05-08 |
| [#18352](https://github.com/sgl-project/sglang/pull/18352) | CLOSED | feat: add mhc support for deepseek_v2 | `main` | — | 2026-08-23 |

## sgl-project/sglang-jax

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#1603](https://github.com/sgl-project/sglang-jax/pull/1603) | CLOSED | \[DSv4 M1.0\] Bring mHC hyper-connection kernels onto epic/dsv4 (cherry-pick #1547) | `epic/dsv4` | — | 2026-09-07 |
| [#1547](https://github.com/sgl-project/sglang-jax/pull/1547) | MERGED | feat(mhc): add DeepSeek-V4 hyper-connection kernels | `main` | 2026-09-08 | 2026-09-08 |

## tile-ai/TileOPs

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#1649](https://github.com/tile-ai/TileOPs/pull/1649) | MERGED | \[Fix\]\[MHC\] Fix post shared-buffer stride | `main` | 2026-07-06 | 2026-07-06 |
| [#859](https://github.com/tile-ai/TileOPs/pull/859) | MERGED | \[Refactor\]\[MHC\] Rename mhc family classes to MHC* per naming convention | `main` | 2026-04-09 | 2026-04-09 |
| [#715](https://github.com/tile-ai/TileOPs/pull/715) | MERGED | \[Fix\]\[Op\] MHC-Pre kernel ZeroDivisionError on medium config | `main` | 2026-03-31 | 2026-03-31 |
| [#520](https://github.com/tile-ai/TileOPs/pull/520) | MERGED | \[Fix\]\[MHC\] Fix MHC pre auto-tuning: sigmoid serialization and scalar params | `main` | 2026-03-15 | 2026-03-19 |
| [#205](https://github.com/tile-ai/TileOPs/pull/205) | MERGED | \[BugFix\]\[mHC\] Fix register_fake signature mismatch in mHC kernels | `main` | 2026-02-26 | 2026-02-26 |
| [#172](https://github.com/tile-ai/TileOPs/pull/172) | MERGED | \[BugFix\]\[mHC\] Address numerical instability in the mHC kernel | `main` | 2026-02-25 | 2026-02-25 |
| [#161](https://github.com/tile-ai/TileOPs/pull/161) | MERGED | \[Feat\]\[mHC\] Implement mHC kernels | `main` | 2026-01-28 | 2026-02-28 |

## tile-ai/tilelang

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#3156](https://github.com/tile-ai/tilelang/pull/3156) | OPEN | \[ROCm\] Support DeepSeek mHC pre and post kernels | `main` | — | 2026-09-03 |
| [#1758](https://github.com/tile-ai/tilelang/pull/1758) | MERGED | Add an example: mHC residual projection backward | `main` | 2026-02-16 | 2026-02-16 |
| [#1684](https://github.com/tile-ai/tilelang/pull/1684) | MERGED | \[Example\] Add example for mHC inference kernels. | `main` | 2026-01-16 | 2026-01-17 |

## tile-ai/tilelang-ascend

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#1712](https://github.com/tile-ai/tilelang-ascend/pull/1712) | OPEN | 新增examples/cann-bench/mhc_sinkhorn算子 | `ascendc_pto` | — | 2026-09-04 |
| [#1643](https://github.com/tile-ai/tilelang-ascend/pull/1643) | OPEN | Add mhc_post example | `ascendc_pto` | — | 2026-08-31 |
| [#1633](https://github.com/tile-ai/tilelang-ascend/pull/1633) | OPEN | \[Example\] Add mhc_bwd (Sinkhorn implicit CG) operator for Ascend NPU | `ascendc_pto` | — | 2026-08-19 |
| [#1629](https://github.com/tile-ai/tilelang-ascend/pull/1629) | OPEN | Add mhc_pre example | `ascendc_pto` | — | 2026-08-24 |
| [#1621](https://github.com/tile-ai/tilelang-ascend/pull/1621) | OPEN | \[Example\] Add mhc_pre operator for Ascend NPU | `ascendc_pto` | — | 2026-09-07 |
| [#1561](https://github.com/tile-ai/tilelang-ascend/pull/1561) | MERGED | \[Example\] Add mhc_post operator for Ascend NPU | `ascendc_pto` | 2026-09-10 | 2026-09-10 |
| [#1434](https://github.com/tile-ai/tilelang-ascend/pull/1434) | OPEN | \[Add\]Add the mhc expand kernel | `ascendc_pto` | — | 2026-08-03 |
| [#1433](https://github.com/tile-ai/tilelang-ascend/pull/1433) | OPEN | \[Add\]Add the mhc multilayer_recompute kernel | `ascendc_pto` | — | 2026-08-03 |
| [#1432](https://github.com/tile-ai/tilelang-ascend/pull/1432) | OPEN | \[Add\]Add the mhc pre_big_fuse kernel | `ascendc_pto` | — | 2026-08-03 |
| [#1425](https://github.com/tile-ai/tilelang-ascend/pull/1425) | OPEN | \[Add\]Add the mhc post kernel | `ascendc_pto` | — | 2026-08-04 |
| [#1420](https://github.com/tile-ai/tilelang-ascend/pull/1420) | OPEN | \[Add\]Add the mhc norm_fn kernel | `ascendc_pto` | — | 2026-07-21 |
| [#1412](https://github.com/tile-ai/tilelang-ascend/pull/1412) | OPEN | \[Add\]Add the mhc sinkhorn kernel | `ascendc_pto` | — | 2026-08-02 |
| [#1409](https://github.com/tile-ai/tilelang-ascend/pull/1409) | OPEN | \[Add\]Add the mhc pre_apply_mix kernel | `ascendc_pto` | — | 2026-07-28 |
| [#1393](https://github.com/tile-ai/tilelang-ascend/pull/1393) | OPEN | \[Add\]Add the mhc pre_split_mixes kernel | `ascendc_pto` | — | 2026-07-30 |
| [#1392](https://github.com/tile-ai/tilelang-ascend/pull/1392) | CLOSED | \[Add\]Add the mhc pre_split_mixes kernel | `ascendc_pto` | — | 2026-07-16 |
| [#1287](https://github.com/tile-ai/tilelang-ascend/pull/1287) | MERGED | \[Add\]Add the mhc head_compute_mix kernel | `ascendc_pto` | 2026-07-14 | 2026-07-14 |
| [#953](https://github.com/tile-ai/tilelang-ascend/pull/953) | MERGED | \[Add\] optim hc_split_sinkhorn performance | `ascendc_pto` | 2026-04-30 | 2026-04-30 |

## tile-ai/tilelang-mlir-ascend

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#94](https://github.com/tile-ai/tilelang-mlir-ascend/pull/94) | MERGED | \[AscendNPU-IR\]\[A5\]Support dsv4 sparse and mHC | `main` | 2026-06-04 | 2026-06-04 |

## vllm-project/tpu-inference

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#3392](https://github.com/vllm-project/tpu-inference/pull/3392) | MERGED | \[DSv4\] Port mhc kernels from g3 to tpu-inference | `main` | 2026-08-14 | 2026-08-14 |
| [#2950](https://github.com/vllm-project/tpu-inference/pull/2950) | MERGED | \[Deedseek V4\] add  HCHeadOp  support to mhc.py | `main` | 2026-06-23 | 2026-06-23 |

## vllm-project/vllm

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#56342](https://github.com/vllm-project/vllm/pull/56342) | OPEN | \[ROCm\]\[DSv4.1\] Fix TileLang mHC pre on 64-wide wavefronts and use the fused kernels on the AMD path | `main` | — | 2026-09-11 |
| [#56255](https://github.com/vllm-project/vllm/pull/56255) | OPEN | \[DSv4.1\] Integrate Mega-mHC from DeepGEMM | `dsv41-optimized` | — | 2026-09-10 |
| [#55573](https://github.com/vllm-project/vllm/pull/55573) | OPEN | \[Performance\]\[Model\] DeepSeek-V4: gather logits rows before hc_head on prefill steps | `main` | — | 2026-09-11 |
| [#53972](https://github.com/vllm-project/vllm/pull/53972) | OPEN | \[Perf\]\[DSV4\] Use broadcast mHC pre for DeepSeek V4 DSpark | `main` | — | 2026-09-03 |
| [#53055](https://github.com/vllm-project/vllm/pull/53055) | OPEN | \[Bugfix\]\[Kernels\] Fallback mhc_pre_broadcast to TileLang when DeepGEMM unsupported | `main` | — | 2026-09-08 |
| [#52941](https://github.com/vllm-project/vllm/pull/52941) | OPEN | \[Bugfix\]\[Model\]\[NVIDIA\] Fix DeepSeek V4 mHC TileLang warmup for nvidi… | `main` | — | 2026-09-09 |
| [#52740](https://github.com/vllm-project/vllm/pull/52740) | CLOSED · draft | \[Perf\]\[DSv4\] Warm up mHC and prefill metadata kernels at startup to avoid runtime JIT | `main` | — | 2026-08-18 |
| [#52737](https://github.com/vllm-project/vllm/pull/52737) | MERGED | \[ROCm\]\[Perf\] Fuse DeepSeek-V4 mHC post/pre and RMSNorm with AITER | `main` | 2026-08-20 | 2026-08-20 |
| [#52725](https://github.com/vllm-project/vllm/pull/52725) | OPEN · draft | fuse norm into mhc | `main` | — | 2026-08-20 |
| [#52626](https://github.com/vllm-project/vllm/pull/52626) | MERGED | \[Bugfix\] Fix DeepSeek V4 mHC broadcast buffer for weight sync | `main` | 2026-08-18 | 2026-08-18 |
| [#51959](https://github.com/vllm-project/vllm/pull/51959) | CLOSED | \[Build\] DeepGEMM pin has no SM120 kernels: family-12 Blackwell cannot run hyperconnections | `main` | — | 2026-08-13 |
| [#51802](https://github.com/vllm-project/vllm/pull/51802) | OPEN | \[Bugfix\] Fix NVIDIA DeepSeek V4 mHC warmup | `main` | — | 2026-09-10 |
| [#51368](https://github.com/vllm-project/vllm/pull/51368) | MERGED | \[Bugfix\] Fix DeepSeek V4 mHC broadcast buffer for dummy load | `main` | 2026-08-19 | 2026-08-19 |
| [#51244](https://github.com/vllm-project/vllm/pull/51244) | OPEN | feat: optimize MHC post + HC head + RMSNorm fusions for DeepSeek V4 | `main` | — | 2026-09-09 |
| [#50645](https://github.com/vllm-project/vllm/pull/50645) | OPEN | \[Bugfix\] Guard mhc_pre_broadcast_tilelang on DeepGEMM support | `main` | — | 2026-08-23 |
| [#50178](https://github.com/vllm-project/vllm/pull/50178) | OPEN | \[9/N\]\[warmup\]\[DSv4\] Migrate MHC TileLang kernels | `main` | — | 2026-09-11 |
| [#49707](https://github.com/vllm-project/vllm/pull/49707) | OPEN | Fix NVIDIA DeepSeek V4 MHC warmup coverage | `main` | — | 2026-08-19 |
| [#49429](https://github.com/vllm-project/vllm/pull/49429) | MERGED | \[Bugfix\] Fix mHC block-M prenorm GEMM cross-row reduction carry-over | `main` | 2026-07-27 | 2026-07-27 |
| [#49315](https://github.com/vllm-project/vllm/pull/49315) | MERGED | \[2/N\]\[Feat\]\[Perf\] Add new warmup infrastructure for JITs. Add predicate filtering for JIT warmup, and migrate Inkling FA4 | `main` | 2026-08-11 | 2026-08-11 |
| [#48619](https://github.com/vllm-project/vllm/pull/48619) | OPEN | Port DeepGemm MHC Kernel to CuTeDSL | `main` | — | 2026-08-18 |
| [#47807](https://github.com/vllm-project/vllm/pull/47807) | CLOSED | refactor: streamline DeepSeek V4 mHC warmup and remove token-size cap | `main` | — | 2026-09-10 |
| [#47518](https://github.com/vllm-project/vllm/pull/47518) | CLOSED · draft | \[ROCm\]\[DSV4\] Enable fused AITER mHC post+pre kernel for decode | `main` | — | 2026-08-02 |
| [#47419](https://github.com/vllm-project/vllm/pull/47419) | MERGED | \[ROCm\] Enable DeepSeek-V4 DSpark speculative decoding on AMD (MI350X / MI355X, gfx950) | `main` | 2026-07-10 | 2026-07-23 |
| [#47245](https://github.com/vllm-project/vllm/pull/47245) | MERGED | \[XPU\]add sycl path for Mhc | `main` | 2026-07-20 | 2026-07-20 |
| [#46704](https://github.com/vllm-project/vllm/pull/46704) | CLOSED · draft | Add sycl kernel mhc path for dsv4 | `main` | — | 2026-06-30 |
| [#45931](https://github.com/vllm-project/vllm/pull/45931) | MERGED | \[ROCm\]\[DSV4\] Disable TileLang MHC dispatch on gfx942 | `main` | 2026-06-22 | 2026-06-22 |
| [#44692](https://github.com/vllm-project/vllm/pull/44692) | MERGED | \[Bugfix\]\[Kernel\] Fix mHC fused-RMSNorm big-fuse miscompile for hidden_size != 4096 | `main` | 2026-06-06 | 2026-06-06 |
| [#44144](https://github.com/vllm-project/vllm/pull/44144) | MERGED | \[DSV4\]\[XPU\] Add MHC fused_post_pre support | `main` | 2026-06-09 | 2026-06-09 |
| [#43950](https://github.com/vllm-project/vllm/pull/43950) | MERGED | \[ROCm\]\[DSV4\] Use aiter mHC pre/post as the default ROCm path | `main` | 2026-07-01 | 2026-07-01 |
| [#43905](https://github.com/vllm-project/vllm/pull/43905) | MERGED | \[DSv4\] Move mHC tilelang kernels & Don't use CustomOP in dsv4/nvidia | `main` | 2026-05-29 | 2026-05-29 |
| [#43679](https://github.com/vllm-project/vllm/pull/43679) | MERGED | \[ROCm\]\[DSV4\] Enable Tilelang MHC replacing torch/triton mhc | `main` | 2026-05-28 | 2026-05-28 |
| [#43474](https://github.com/vllm-project/vllm/pull/43474) | MERGED | \[Kernel\] Add mhc_pre_big_fuse_with_norm_tilelang | `main` | 2026-05-25 | 2026-05-25 |
| [#43437](https://github.com/vllm-project/vllm/pull/43437) | MERGED | mhc_post - remove sts & add vectorized copies | `main` | 2026-05-22 | 2026-05-22 |
| [#42930](https://github.com/vllm-project/vllm/pull/42930) | MERGED | \[Bugfix\] Fix DSV4 MTP after ROCm mHC integration | `main` | 2026-05-18 | 2026-05-18 |
| [#42735](https://github.com/vllm-project/vllm/pull/42735) | OPEN | \[Perf\]\[Kernel\] Use bf16 shared staging in mHC pre TileLang kernel | `main` | — | 2026-08-28 |
| [#42733](https://github.com/vllm-project/vllm/pull/42733) | CLOSED | \[Kernel\] Use bf16 shared staging in mHC pre TileLang kernel | `main` | — | 2026-05-15 |
| [#41946](https://github.com/vllm-project/vllm/pull/41946) | MERGED | \[Bugfix\] \[ROCm\] \[DSV4\] \[Perf\] Add aiter mhc support | `main` | 2026-05-13 | 2026-05-13 |
| [#41536](https://github.com/vllm-project/vllm/pull/41536) | MERGED | add fused mhc_post_pre kernel | `main` | 2026-05-11 | 2026-05-11 |
| [#41441](https://github.com/vllm-project/vllm/pull/41441) | CLOSED · draft | \[DSV4\] AR+mhc_post fusion | `main` | — | 2026-08-03 |

## vllm-project/vllm-ascend

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#16321](https://github.com/vllm-project/vllm-ascend/pull/16321) | OPEN | \[Performance\]\[Model\] Reuse fused mHC operators for GLM-5.3-Flash | `main` | — | 2026-09-11 |
| [#16230](https://github.com/vllm-project/vllm-ascend/pull/16230) | OPEN | \[Performance\]\[Model\] Elide DeepSeek-V4 mHC residual clones | `main` | — | 2026-09-11 |
| [#15429](https://github.com/vllm-project/vllm-ascend/pull/15429) | OPEN | \[Perf\]\[Ops\] Fused mHC via AscendC npu_hc_pre_v2/npu_hc_post | `main` | — | 2026-09-03 |
| [#15398](https://github.com/vllm-project/vllm-ascend/pull/15398) | CLOSED | \[Perf\]\[Ops\] Fused mHC and AscendC causal_conv1d for GLM/KDA hybrid models on A2 | `main` | — | 2026-08-31 |
| [#14212](https://github.com/vllm-project/vllm-ascend/pull/14212) | OPEN · draft | Adapt telechat4 model and add mhc related operators | `main` | — | 2026-08-24 |

## vllm-project/vllm-omni

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#7288](https://github.com/vllm-project/vllm-omni/pull/7288) | OPEN · draft | \[Refactor\] Express MAGI-2 mHC pre application as a batch matmul | `main` | — | 2026-09-08 |
| [#7261](https://github.com/vllm-project/vllm-omni/pull/7261) | OPEN · draft | feat(magi2): add opt-in mHC post-processing CustomOps | `main` | — | 2026-09-11 |

## vllm-project/vllm-xpu-kernels

| PR | 状态 | 原始标题 | 目标分支 | 合入 UTC | 最近更新 UTC |
| --- | --- | --- | --- | --- | --- |
| [#533](https://github.com/vllm-project/vllm-xpu-kernels/pull/533) | OPEN | fuse rmsnorm into mhc_pre & mhc_post_pre | `main` | — | 2026-09-01 |
| [#425](https://github.com/vllm-project/vllm-xpu-kernels/pull/425) | MERGED | Add mHC kernels for DeepSeek-V4 | `main` | 2026-07-03 | 2026-07-03 |
| [#375](https://github.com/vllm-project/vllm-xpu-kernels/pull/375) | CLOSED | add implementation for dsv4-mhc | `main` | — | 2026-06-17 |

## 检索边界

包含 mHC 直接实现及部分相关大 PR/依赖/替代方案，标题也可能只有模型名或 release 名。全局同名搜索含大量无关结果且受到 1000 条上限限制；核心仓库另行 scoped 搜索。该目录不声称枚举全互联网或私有 PR。PR 有合入记录也不代表所有内容已出现在稳定版 wheel，部署前仍需匹配具体版本和分支。
