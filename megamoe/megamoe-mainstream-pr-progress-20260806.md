# MegaMoE 主流实现与接入进展（2026-08-13）

> 状态快照：2026-08-13，Asia/Shanghai。本文中的 `OPEN`、`MERGED`、`MERGEABLE`、`CONFLICTING`、`APPROVED` 和 checks 状态以 2026-08-13 的 GitHub 为准。

## 结论

截至 2026-08-13，MegaMoE 生态的主线比 8 月 6 日更清晰了：

1. **SM100 主链继续稳定，Kimi-K3 SiTU 也已打通到上游实现层**：DeepGEMM `#396` 已在 2026-08-11 合入，TensorRT-LLM `#17063` 已在 2026-08-13 合入。
2. **FlashInfer 的 Hopper 路线已经从“候选”变成“主线已合入”**：`#4113` pull-style 于 2026-08-08 合入，`#4069` push-style 于 2026-08-12 合入；但 DeepGEMM 的 Hopper `#323/#360/#383` 仍开放且冲突，生态并未收敛为单一上游实现。
3. **框架消费层仍未完全收口**：vLLM 的 FlashInfer backend `#49636` 仍开放；SGLang 的 `#31470` 仍是开放候选，zero-copy 版 `#33571` 仍是 draft 且已冲突；FlashInfer `#4341` 的 workspace output view 也尚未合入。
4. **非 NVIDIA 路线继续推进，但成熟度不均衡**：ROCm 的 AITER `#4439` 已在 2026-08-10 合入，FlyDSL `#972` 仍开放；昇腾 A5 的 `#13655` 仍开放、可合并，旧的 `#11137` 仍冲突。
当前最值得跟踪的关系为：

```text
DeepGEMM SM100:   #304 -> #316 -> #364 -> #377 -> #396      已合入
FlashInfer SM100: #3686 -> #3852 -> #3980 -> #4079 -> #4101 已合入

Hopper implementations:
  DeepGEMM:   #323 / #360 / #383                            开放、冲突
  FlashInfer: #4113 pull / #4069 push                       已合入

Serving integrations:
  TensorRT-LLM: #13384 / #14608 / #16190 / #17063           已合入
  vLLM native:  #40860 / #43339 / #43632 / #51146          已合入
  vLLM FlashInfer: #49636                                   开放
  SGLang native: #23882 / #25052 / #29016                  已合入
  SGLang FlashInfer: #31470 / #33571                       开放（后者 draft、冲突）

Other hardware:
  ROCm:   FlyDSL #876 / #972, AITER #4439                  部分已合入
  Ascend: #11701 / #11137 / #13655                         部分已合入
```

## 相比 2026-08-06 的主要变化

| 路线 | PR | 2026-08-13 状态 | 本轮变化或新信息 |
| --- | --- | --- | --- |
| FlashInfer Hopper pull | [flashinfer#4113](https://github.com/flashinfer-ai/flashinfer/pull/4113) | MERGED | 2026-08-08 合入。Hopper pull-style 不再只是候选；PR 中补的 multi-rank pure-torch oracle 也随之进入主线。 |
| FlashInfer Hopper push | [flashinfer#4069](https://github.com/flashinfer-ai/flashinfer/pull/4069) | MERGED | 2026-08-12 合入。说明 FlashInfer 选择同时保留 push / pull 两条 SM90 路线，而不是只收敛到单 backend。 |
| FlashInfer zero-copy | [flashinfer#4341](https://github.com/flashinfer-ai/flashinfer/pull/4341) | OPEN、CONFLICTING、REVIEW_REQUIRED | 仍未推进；自 2026-08-06 后没有新的可见更新，说明 zero-copy API 链路暂时没有跟上 Hopper backend 的合入节奏。 |
| SGLang FlashInfer 候选 | [sglang#31470](https://github.com/sgl-project/sglang/pull/31470) | OPEN、MERGEABLE、REVIEW_REQUIRED | 2026-08-13 仍有更新；相比 `#33571`，它现在反而是更活跃的 FlashInfer 接入候选。 |
| SGLang zero-copy | [sglang#33571](https://github.com/sgl-project/sglang/pull/33571) | OPEN draft、CONFLICTING | 仍依赖 `flashinfer#4341`，而且当前已冲突；短期内更像停在实验分支，而不是即将合入的消费路径。 |
| DeepGEMM Kimi-K3 | [DeepGEMM#396](https://github.com/deepseek-ai/DeepGEMM/pull/396) | MERGED | 2026-08-11 合入。SiTU 不再只是 SM100 MegaMoE 的候选功能，而是已进入上游实现。 |
| TensorRT-LLM Kimi-K3 | [TensorRT-LLM#17063](https://github.com/NVIDIA/TensorRT-LLM/pull/17063) | MERGED | 2026-08-13 合入。Kimi-K3 SiTU MegaMoE 已进入 TensorRT-LLM 主线。 |
| AMD engine integration | [AITER#4439](https://github.com/ROCm/aiter/pull/4439) | MERGED | 2026-08-10 合入。ROCm 路线从“底层 kernel 已有、engine 还在接”推进到 engine 集成也进主线。 |
| AMD A4W4 | [FlyDSL#972](https://github.com/ROCm/FlyDSL/pull/972) | OPEN、MERGEABLE、REVIEW_REQUIRED | 仍在推进 MI355X MegaMoE V2 A4W4 与 FP8 blockwise P2P Stage2 transport。 |
| DeepGEMM Hopper | [DeepGEMM#383](https://github.com/deepseek-ai/DeepGEMM/pull/383) | OPEN、CONFLICTING | 2026-08-13 仍有更新，但目标分支还是 `nv_dev`，冲突依旧存在。FlashInfer 已合入并不等于 DeepGEMM Hopper upstream 已收敛。 |

## 1. DeepGEMM：原生 MegaMoE

### SM100 已合入主线

| PR | 状态 | 定位 |
| --- | --- | --- |
| [#304](https://github.com/deepseek-ai/DeepGEMM/pull/304) | MERGED | 首次公开 MegaMoE：SM100、FP8 activation × FP4 weight，融合 dispatch、L1、SwiGLU、L2 和 combine。 |
| [#316](https://github.com/deepseek-ai/DeepGEMM/pull/316) | MERGED | 第一轮正式优化与 DeepSeek-V4 EP8 benchmark。 |
| [#364](https://github.com/deepseek-ai/DeepGEMM/pull/364) | MERGED | 增加 BF16 MegaMoE 等能力。 |
| [#377](https://github.com/deepseek-ai/DeepGEMM/pull/377) | MERGED | 新 scheduler、融合 shared experts；PR 将路径概括为 11 kernels 合为 1 个 persistent kernel。 |
| [#396](https://github.com/deepseek-ai/DeepGEMM/pull/396) | MERGED | 为 SM100 FP8×FP4 MegaMoE 增加 SiTU JIT specialization。 |

### SM90 仍是并行候选，而非统一 upstream

- [#323](https://github.com/deepseek-ai/DeepGEMM/pull/323)：SM90 FP8 fused path，OPEN、CONFLICTING。
- [#360](https://github.com/deepseek-ai/DeepGEMM/pull/360)：cooperative 单 kernel 方案，OPEN、CONFLICTING。
- [#383](https://github.com/deepseek-ai/DeepGEMM/pull/383)：destination-rank pull、L1/L2 两个大 kernel，OPEN、CONFLICTING，且 2026-08-13 仍有更新。

这意味着截至 2026-08-13，FlashInfer 已经把 Hopper backend 合到主线，但 DeepGEMM 自己的 Hopper upstream 还没有收口成单一可用版本。

### Kimi-K3 SiTU 已合入 DeepGEMM

[DeepGEMM #396](https://github.com/deepseek-ai/DeepGEMM/pull/396) 已在 2026-08-11 合入。它将 `activation`、`situ_beta` 和 `situ_linear_beta` 穿透 Python/C++/JIT/kernel contract，并在 JIT specialization 时选择 SiTU 或 SwiGLU。当前已知边界仍是：

- 仅适用于 SM100 FP8×FP4 MegaMoE；
- 不在 kernel inner loop 增加运行时 activation 分支；
- BF16 MegaMoE 仍只支持 SwiGLU。

### 本地新增代码审计与 roofline 证据

本仓库本轮新增了两份与 DeepGEMM Hopper 路线直接相关的本地材料：

- [sm90-sm100-megamoe-component-comparison-20260810.md](./sm90-sm100-megamoe-component-comparison-20260810.md)：结论是 SM90 与 SM100 共享 symmetric-memory 机制和公共 layout，但 **ABI/type guard 不兼容**，不能把 `SM90SymmBuffer` 简单等价成 `SymmBuffer`。
- [deepgemm-pr383-sm90-megamoe-roofline-20260811.md](./deepgemm-pr383-sm90-megamoe-roofline-20260811.md)：对 PR383 的 H20 数据做 roofline 建模，结论是 PR383 的 22 个 H20 点都高于文中 296 TFLOP/s 名义估计，适合作为一致性检查，但不是严格物理下界证明。

## 2. FlashInfer `moe_ep`

SM100 主链已经闭合：

| PR | 状态 | 定位 |
| --- | --- | --- |
| [#3686](https://github.com/flashinfer-ai/flashinfer/pull/3686) | MERGED | `MoEEpLayer` 和 NCCL-EP/NIXL-EP split 基础。 |
| [#3852](https://github.com/flashinfer-ai/flashinfer/pull/3852) | MERGED | 接入 `deep_gemm_mega`、CuTeDSL NVFP4、CuTeDSL MXFP8。 |
| [#3980](https://github.com/flashinfer-ai/flashinfer/pull/3980) | MERGED | 重组 CuTeDSL kernel drop，补 tuning、FC2 reduce 和量化 combine。 |
| [#4079](https://github.com/flashinfer-ai/flashinfer/pull/4079) | MERGED | CUDA Graph、fused quant+stage、预量化权重、workspace pool、持久 knob cache。 |
| [#4101](https://github.com/flashinfer-ai/flashinfer/pull/4101) | MERGED | 修复 CuTeDSL 4.5.2 mainloop 性能回退。 |

### Hopper push 与 pull 都已合入

| PR | 通信/执行模型 | 当前状态 |
| --- | --- | --- |
| [#4069](https://github.com/flashinfer-ai/flashinfer/pull/4069) | source-rank push；deduplicated dispatch、owner-side grouped combine、round/ack 生命周期 | MERGED，2026-08-12 合入。 |
| [#4113](https://github.com/flashinfer-ai/flashinfer/pull/4113) | NVSHMEM pull-style；dispatch + FC1 + SwiGLU + FC2 + combine 单 launch | MERGED，2026-08-08 合入。 |

这两个 PR 先后合入，意味着 FlashInfer 当前不是在 push / pull 之间二选一，而是把两条 Hopper 路线都留在主线里继续演化。

### Zero-copy 输出链仍未闭合

[FlashInfer #4341](https://github.com/flashinfer-ai/flashinfer/pull/4341) 仍是 OPEN、CONFLICTING、REVIEW_REQUIRED。它定义的核心路径仍然是：

```text
旧路径：workspace output --copy--> caller-owned tensor
新路径：workspace output view -----> framework downstream
```

默认行为不变；只有 backend 声明 `supports_output_view` 且调用方显式开启时才返回 view。当前的问题不是设计是否存在，而是这条 API 线还没有进入主线，因此依赖它的消费方也一起停住了。

## 3. TensorRT-LLM

TensorRT-LLM 现在已经形成更完整的 engine 级 MegaMoE 集成链：

- [#13384](https://github.com/NVIDIA/TensorRT-LLM/pull/13384)：`MegaMoEDeepGemmFusedMoE`，MERGED。
- [#14608](https://github.com/NVIDIA/TensorRT-LLM/pull/14608)：SM100/SM103 NVFP4 `MegaMoECuteDsl`，MERGED。
- [#16190](https://github.com/NVIDIA/TensorRT-LLM/pull/16190)：CuTeDSL kernel、tactic、workspace/CUDA Graph 和 combine 更新，MERGED。
- [#17063](https://github.com/NVIDIA/TensorRT-LLM/pull/17063)：Kimi-K3 SiTU MegaMoE，MERGED，2026-08-13 合入。

因此，相比 8 月 6 日，当时还只是“候选”的 Kimi-K3 SiTU 路线现在已经把 DeepGEMM 和 TensorRT-LLM 两层都推进到主线。需要继续观察的是 release、默认 backend 选择和后续硬件验证，而不是它是否还能 upstream。

## 4. vLLM 与 SGLang

### vLLM

已合入的 native DeepGEMM 路径：

- [#40860](https://github.com/vllm-project/vllm/pull/40860)：DeepSeek-V4 和 `deep_gemm_mega_moe` 主接入；
- [#43339](https://github.com/vllm-project/vllm/pull/43339)：MegaMoE EPLB；
- [#43632](https://github.com/vllm-project/vllm/pull/43632)：input-prep kernel 迁移到 `nvidia/ops`；
- [#51146](https://github.com/vllm-project/vllm/pull/51146)：修正 Kimi-K3 MegaMoE path 的额外 add，已于 2026-08-06 合入。

正在推进的 FlashInfer 路径是 [#49636](https://github.com/vllm-project/vllm/pull/49636)，增加：

- `flashinfer_moe_ep_mega_deep_gemm`；
- `flashinfer_moe_ep_mega_cutedsl`。

它仍然是 OPEN、MERGEABLE、REVIEW_REQUIRED，`pre-run-check` 当前失败。它保持 opt-in，并明确拒绝 EPLB，因此就算合入，也不会等价于“FlashInfer 路线完全替代 native DeepGEMM”。

Kimi-K3 邻近变化仍需注意：[vLLM recipes #752](https://github.com/vllm-project/recipes/pull/752) 已在 2026-08-06 合入，把非 GB 平台的 Kimi-K3 recipe 移除 MegaMoE。因此不能把“vLLM 主线里有 MegaMoE 代码”泛化成“所有 Blackwell 平台都默认推荐 Kimi-K3 MegaMoE”。

### SGLang

已合入 native DeepGEMM 路径：

- [#23882](https://github.com/sgl-project/sglang/pull/23882)：DeepSeek-V4 主接入；
- [#25052](https://github.com/sgl-project/sglang/pull/25052)：W4A4 MegaMoE；
- [#29016](https://github.com/sgl-project/sglang/pull/29016)：SM90 FP8 MegaMoE。

FlashInfer 消费方仍未合入，但候选形势比 8 月 6 日更分化：

- [#31470](https://github.com/sgl-project/sglang/pull/31470)：早期 `flashinfer_megamoe` runner/A2A backend，OPEN、MERGEABLE、REVIEW_REQUIRED，且 2026-08-13 仍有更新；
- [#33571](https://github.com/sgl-project/sglang/pull/33571)：zero-copy adapter，仍是 OPEN draft、CONFLICTING，且公开可见更新停在 2026-08-05。

因此当前 SGLang 的生产主线仍是原生 DeepGEMM 路径；如果只看“谁更像下一条会继续推进的 FlashInfer 接入”，现在反而应优先盯 `#31470`，而不是依赖 `#4341` 的 `#33571`。

## 5. AMD、昇腾及其他硬件生态

### AMD

- [FlyDSL #876](https://github.com/ROCm/FlyDSL/pull/876)：MI355X/gfx950 A8W4 MegaMoEV2，MERGED。
- [FlyDSL #972](https://github.com/ROCm/FlyDSL/pull/972)：A4W4 + FP8 blockwise P2P，OPEN、MERGEABLE、REVIEW_REQUIRED。
- [AITER #4439](https://github.com/ROCm/aiter/pull/4439)：A8W4 MegaMoEV2 engine integration，MERGED，2026-08-10 合入。

相比 8 月 6 日，AMD 路线的最大变化不是又多了一个候选 kernel，而是 AITER engine integration 也已经进主线。这让 ROCm 路线从“算子存在”更接近“框架/engine 可消费”。

### Ascend

- [vllm-ascend #11701](https://github.com/vllm-project/vllm-ascend/pull/11701)：A3 CANN `mega_moe`，MERGED。
- [vllm-ascend #11137](https://github.com/vllm-project/vllm-ascend/pull/11137)：A5 MegaMoe 单 C kernel，OPEN、CONFLICTING。
- [vllm-ascend #13655](https://github.com/vllm-project/vllm-ascend/pull/13655)：A5 Kimi-K3 W4A8 MXFP + SiTU，OPEN、MERGEABLE。

昇腾路线本轮没有像 ROCm 那样出现“关键 PR 刚合入”的拐点，但 `#13655` 仍是值得跟踪的 A5 增量。

### 其他

- [FastDeploy #7943](https://github.com/PaddlePaddle/FastDeploy/pull/7943) / [#8038](https://github.com/PaddlePaddle/FastDeploy/pull/8038)：WFP4A8 MegaMoE，MERGED。
- [cuDNN Frontend #448](https://github.com/NVIDIA/cudnn-frontend/pull/448)：统一 `MoeEp` Python API + CuTeDSL backend，OPEN draft、CONFLICTING。

## 建议跟踪顺序

1. **DeepGEMM Hopper 是否继续 upstream**：FlashInfer 已经完成 SM90 合入，接下来要看 DeepGEMM `#383` 能否脱离 `nv_dev` 冲突，或者 `#323/#360/#383` 是否继续分叉。
2. **Serving zero-copy 链路是否恢复推进**：`flashinfer#4341` 长时间停滞后，SGLang `#33571` 也一起变成 draft+conflicting；这是当前最明显的“设计存在，但主线没走通”的环节。
3. **vLLM FlashInfer backend**：`#49636` 仍是 vLLM 接入 FlashInfer `moe_ep` 的关键入口，尤其要看 `pre-run-check`、review 和 opt-in 语义是否变化。
4. **SGLang 会选哪条 FlashInfer 路线**：`#31470` 近期还有更新，而 `#33571` 已冲突；短期值得优先盯前者。
5. **ROCm / Ascend 扩展**：FlyDSL `#972` 与 vLLM Ascend `#13655` 仍是非 NVIDIA 路线里最值得继续跟踪的两个开放增量。

## 证据边界

- GitHub 的 `MERGEABLE` 只说明当前 diff 可以合并，不代表 review、CI、硬件验证或发布条件已经完成。
- `MERGED` 只说明代码进入主分支，不自动等同于默认启用、正式 release、端到端推荐配置或所有硬件都已验证。
- PR benchmark 与本地 bench 使用的 GPU、EP 拓扑、模型 geometry、token bucket、quantization、计时边界和 baseline 并不统一，不能直接横向排名。
- 本文混合了两类证据：`GitHub PR 状态` 和 `本地新增实验`。前者回答“是否进主线”，后者回答“当前可见的描述性表现如何”；两者不能相互替代。
