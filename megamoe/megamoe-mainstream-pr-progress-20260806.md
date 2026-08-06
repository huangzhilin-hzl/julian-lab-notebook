# MegaMoE 主流实现与接入进展（2026-08-06）

> 状态快照：2026-08-06，Asia/Shanghai。本文中的 `OPEN`、`MERGEABLE`、`APPROVED`、`CONFLICTING` 和 checks 状态来自当日 GitHub；作者给出的性能、精度和硬件验证结果均标明为 PR 证据，不等同于本地复现。

## 结论

截至 2026-08-06，MegaMoE 生态可以分成三层：

1. **SM100 核心实现已经稳定落地主线**：DeepGEMM 的 FP8×FP4 persistent MegaMoE、FlashInfer `moe_ep` 的 DeepGEMM/CuTeDSL backends，以及 TensorRT-LLM 的两套 engine backend 均已合入。
2. **Hopper 仍未收敛为单一 upstream 实现**：DeepGEMM `#323/#360/#383` 仍开放且与目标分支冲突；FlashInfer `#4069` push-style 和 `#4113` pull-style 均可合并，但仍未落入 main。
3. **近期增量已从“能接入”转向“减少框架适配开销和扩展模型/硬件”**：FlashInfer/SGLang 在推进 zero-copy workspace output；DeepGEMM/TensorRT-LLM 在补 Kimi-K3 SiTU；AMD 在补 A4W4 和 FP8 P2P；昇腾在补 A5 W4A8 MXFP。

当前最值得跟踪的关系为：

```text
DeepGEMM SM100:  #304 -> #316 -> #364 -> #377        已合入
FlashInfer SM100: #3686 -> #3852 -> #3980 -> #4079 -> #4101

Hopper candidates:
  DeepGEMM: #323 / #360 / #383                       开放、冲突
  FlashInfer: #4069 push-style / #4113 pull-style    开放、可合并

Serving integrations:
  TensorRT-LLM: #13384 / #14608 / #16190             已合入
  vLLM native: #40860 / #43339 / #43632              已合入
  vLLM FlashInfer: #49636                             开放
  SGLang native: #23882 / #25052 / #29016            已合入
  SGLang FlashInfer: #31470 / #33571                  draft
```

## 相比 2026-08-03 的主要变化

| 路线 | PR | 2026-08-06 状态 | 本轮变化或新信息 |
| --- | --- | --- | --- |
| FlashInfer Hopper pull | [flashinfer#4113](https://github.com/flashinfer-ai/flashinfer/pull/4113) | OPEN、APPROVED、MERGEABLE | 已获得批准；新增 SM90 multi-rank torch oracle，同时覆盖三条既有 SM100 mega backend。当前 `AOT Build Import` 和 aggregate `Test Results Summary` 仍有失败项。 |
| FlashInfer zero-copy | [flashinfer#4341](https://github.com/flashinfer-ai/flashinfer/pull/4341) | OPEN、MERGEABLE、REVIEW_REQUIRED | 新增 opt-in `return_workspace_view`，允许支持的 backend 直接返回 workspace-backed output，默认仍复制到 caller-owned tensor。 |
| SGLang zero-copy | [sglang#33571](https://github.com/sgl-project/sglang/pull/33571) | OPEN draft、MERGEABLE | 与 `flashinfer#4341` 配对，保留 int32 router IDs，并移除 MegaMoE 输出 materialization；PR 报告 TP4/DP4 serving 和 CUDA Graph 验证。 |
| vLLM FlashInfer | [vllm#49636](https://github.com/vllm-project/vllm/pull/49636) | OPEN、MERGEABLE、REVIEW_REQUIRED | 继续推进 opt-in `flashinfer_moe_ep_mega_deep_gemm` 和 `flashinfer_moe_ep_mega_cutedsl`；仍拒绝 EPLB。 |
| DeepGEMM Kimi-K3 | [DeepGEMM#396](https://github.com/deepseek-ai/DeepGEMM/pull/396) | OPEN、MERGEABLE | 新增 SM100 FP8×FP4 MegaMoE 的 SiTU JIT specialization；BF16 MegaMoE 仍保持 SwiGLU-only。 |
| TensorRT-LLM Kimi-K3 | [TensorRT-LLM#17063](https://github.com/NVIDIA/TensorRT-LLM/pull/17063) | OPEN、APPROVED、MERGEABLE | 已从 draft 转为 ready，并获批准；仍以 TRTLLM-Gen 为默认，显式选择才走 MegaMoE。最新 DeepGEMM 2.6.1 port 的 fresh B200/B300 runtime 验证仍待补。 |
| AMD A4W4 | [FlyDSL#972](https://github.com/ROCm/FlyDSL/pull/972) | OPEN、MERGEABLE | 新增 MI355X MegaMoE V2 A4W4，以及 FP8 blockwise P2P Stage2 transport。 |
| AMD engine integration | [AITER#4439](https://github.com/ROCm/aiter/pull/4439) | OPEN、MERGEABLE | MegaMoEV2 接入 AITER，覆盖 fixed-slot/compact dispatch 和静态 MTPR 配置。 |
| Ascend A5 | [vllm-ascend#13655](https://github.com/vllm-project/vllm-ascend/pull/13655) | OPEN、MERGEABLE | 新增 Kimi-K3 W4A8 MXFP + SiTU 的 CANN MegaMoe；PR 记录了 4 节点/32 NPU serving 验证。 |
| DeepGEMM Hopper | [DeepGEMM#383](https://github.com/deepseek-ai/DeepGEMM/pull/383) | OPEN、CONFLICTING | 8 月 6 日仍有更新，但目标是 `nv_dev`，尚未解决冲突，也尚未成为 upstream SM90 主线。 |

## 1. DeepGEMM：原生 MegaMoE

### SM100 已合入主线

| PR | 状态 | 定位 |
| --- | --- | --- |
| [#304](https://github.com/deepseek-ai/DeepGEMM/pull/304) | MERGED | 首次公开 MegaMoE：SM100、FP8 activation × FP4 weight，融合 dispatch、L1、SwiGLU、L2 和 combine。 |
| [#316](https://github.com/deepseek-ai/DeepGEMM/pull/316) | MERGED | 第一轮正式优化与 DeepSeek-V4 EP8 benchmark。 |
| [#364](https://github.com/deepseek-ai/DeepGEMM/pull/364) | MERGED | 增加 BF16 MegaMoE 等能力。 |
| [#377](https://github.com/deepseek-ai/DeepGEMM/pull/377) | MERGED | 新 scheduler、融合 shared experts；PR 将路径概括为 11 kernels 合为 1 个 persistent kernel。 |

### SM90 仍是并行候选

- [#323](https://github.com/deepseek-ai/DeepGEMM/pull/323)：SM90 FP8 fused path，OPEN、CONFLICTING。
- [#360](https://github.com/deepseek-ai/DeepGEMM/pull/360)：cooperative 单 kernel 方案，OPEN、CONFLICTING。
- [#383](https://github.com/deepseek-ai/DeepGEMM/pull/383)：destination-rank pull、L1/L2 两个大 kernel，OPEN、CONFLICTING。

`#383` 的 PR benchmark 覆盖 H200/H20，但它仍是作者分支数据。本轮未在 Hopper 上复现，不能把表中 kernel 对比直接当成 serving 端到端结果。

### Kimi-K3 SiTU

[DeepGEMM #396](https://github.com/deepseek-ai/DeepGEMM/pull/396) 将 `activation`、`situ_beta` 和 `situ_linear_beta` 穿透 Python/C++/JIT/kernel contract，并在 JIT specialization 时选择 SiTU 或 SwiGLU。当前约束是：

- 仅适用于 SM100 FP8×FP4 MegaMoE；
- 不在 kernel inner loop 增加运行时 activation 分支；
- BF16 MegaMoE 仍只支持 SwiGLU；
- PR 已有 deterministic regression 和 PTXAS 静态检查，但仍未合入。

## 2. FlashInfer `moe_ep`

SM100 主链已经闭合：

| PR | 状态 | 定位 |
| --- | --- | --- |
| [#3686](https://github.com/flashinfer-ai/flashinfer/pull/3686) | MERGED | `MoEEpLayer` 和 NCCL-EP/NIXL-EP split 基础。 |
| [#3852](https://github.com/flashinfer-ai/flashinfer/pull/3852) | MERGED | 接入 `deep_gemm_mega`、CuTeDSL NVFP4、CuTeDSL MXFP8。 |
| [#3980](https://github.com/flashinfer-ai/flashinfer/pull/3980) | MERGED | 重组 CuTeDSL kernel drop，补 tuning、FC2 reduce 和量化 combine。 |
| [#4079](https://github.com/flashinfer-ai/flashinfer/pull/4079) | MERGED | CUDA Graph、fused quant+stage、预量化权重、workspace pool、持久 knob cache。 |
| [#4101](https://github.com/flashinfer-ai/flashinfer/pull/4101) | MERGED | 修复 CuTeDSL 4.5.2 mainloop 性能回退。 |

### Hopper push 与 pull

| PR | 通信/执行模型 | 当前状态 |
| --- | --- | --- |
| [#4069](https://github.com/flashinfer-ai/flashinfer/pull/4069) | source-rank push；deduplicated dispatch、owner-side grouped combine、round/ack 生命周期 | OPEN、MERGEABLE、REVIEW_REQUIRED；aggregate `Test Results Summary` 有失败项。 |
| [#4113](https://github.com/flashinfer-ai/flashinfer/pull/4113) | NVSHMEM pull-style；dispatch + FC1 + SwiGLU + FC2 + combine 单 launch | OPEN、APPROVED、MERGEABLE；`AOT Build Import` 和 aggregate summary 有失败项。 |

`#4113` 的重要进展不只是新增 Hopper backend：它还给 SM90 和三条 SM100 mega backend 增加真实跨 rank 的 pure-torch oracle，避免“同一 fused kernel 自己和自己比”掩盖通信与计算同时出错。

### Zero-copy 输出链

[FlashInfer #4341](https://github.com/flashinfer-ai/flashinfer/pull/4341) 新增可选的 workspace output view：

```text
旧路径：workspace output --copy--> caller-owned tensor
新路径：workspace output view -----> framework downstream
```

默认行为不变；只有 backend 声明 `supports_output_view` 且调用方显式开启时才返回 view。它的直接消费方是 [SGLang #33571](https://github.com/sgl-project/sglang/pull/33571)。后者报告约 2.6%–2.9% output-throughput 改善，但数据来自 PR 指定的 TP4/DP4、短输入长输出实验，本轮没有独立复现。

## 3. TensorRT-LLM

TensorRT-LLM 仍是当前最完整的 engine 级双 backend 集成：

- [#13384](https://github.com/NVIDIA/TensorRT-LLM/pull/13384)：`MegaMoEDeepGemmFusedMoE`，MERGED。
- [#14608](https://github.com/NVIDIA/TensorRT-LLM/pull/14608)：SM100/SM103 NVFP4 `MegaMoECuteDsl`，MERGED。
- [#16190](https://github.com/NVIDIA/TensorRT-LLM/pull/16190)：CuTeDSL kernel、tactic、workspace/CUDA Graph 和 combine 更新，MERGED。
- [#17063](https://github.com/NVIDIA/TensorRT-LLM/pull/17063)：Kimi-K3 SiTU MegaMoE，OPEN、APPROVED、MERGEABLE。

`#17063` 当前仍保留 TRTLLM-Gen 默认 backend；只有显式选择 `MEGAMOE_DEEPGEMM` 才切换。PR 仍依赖 DeepGEMM 开发 fork，且最新 2.6.1 port 的 fresh native build 与 B200/B300 runtime 验证尚未完成，因此还不能把它视为已稳定发布的 Kimi-K3 backend。

## 4. vLLM 与 SGLang

### vLLM

已合入的 native DeepGEMM 路径：

- [#40860](https://github.com/vllm-project/vllm/pull/40860)：DeepSeek-V4 和 `deep_gemm_mega_moe` 主接入；
- [#43339](https://github.com/vllm-project/vllm/pull/43339)：MegaMoE EPLB；
- [#43632](https://github.com/vllm-project/vllm/pull/43632)：input-prep kernel 迁移到 `nvidia/ops`。

正在推进的 FlashInfer 路径是 [#49636](https://github.com/vllm-project/vllm/pull/49636)，增加：

- `flashinfer_moe_ep_mega_deep_gemm`；
- `flashinfer_moe_ep_mega_cutedsl`。

它保持 opt-in，并明确拒绝 EPLB。当前为 OPEN、MERGEABLE、REVIEW_REQUIRED，`pre-run-check` 有失败项。

Kimi-K3 邻近变化：[#51146](https://github.com/vllm-project/vllm/pull/51146) 已修正 MegaMoE path 的额外 add；但 [vLLM recipes #752](https://github.com/vllm-project/recipes/pull/752) 已将非 GB 平台的 Kimi-K3 recipe 移除 MegaMoE。因此不能把“vLLM 支持 Kimi-K3 MegaMoE”泛化为所有 Blackwell 型号的默认推荐配置。

### SGLang

已合入 native DeepGEMM 路径：

- [#23882](https://github.com/sgl-project/sglang/pull/23882)：DeepSeek-V4 主接入；
- [#25052](https://github.com/sgl-project/sglang/pull/25052)：W4A4 MegaMoE；
- [#29016](https://github.com/sgl-project/sglang/pull/29016)：SM90 FP8 MegaMoE。

FlashInfer 消费方仍未合入：

- [#31470](https://github.com/sgl-project/sglang/pull/31470)：早期 `flashinfer_megamoe` runner/A2A backend，OPEN draft；
- [#33571](https://github.com/sgl-project/sglang/pull/33571)：迁移到 current main 的 zero-copy adapter，依赖尚未发布的 `flashinfer#4341` API，同样是 OPEN draft。

因此当前 SGLang 的生产主线仍是原生 DeepGEMM 路径，FlashInfer `moe_ep` 仍属于候选接入。

## 5. AMD、昇腾及其他硬件生态

### AMD

- [FlyDSL #876](https://github.com/ROCm/FlyDSL/pull/876)：MI355X/gfx950 A8W4 MegaMoEV2，MERGED。
- [FlyDSL #972](https://github.com/ROCm/FlyDSL/pull/972)：A4W4 + FP8 blockwise P2P，OPEN、MERGEABLE。
- [AITER #4439](https://github.com/ROCm/aiter/pull/4439)：A8W4 MegaMoEV2 engine integration，OPEN、MERGEABLE。

### Ascend

- [vllm-ascend #11701](https://github.com/vllm-project/vllm-ascend/pull/11701)：A3 CANN `mega_moe`，MERGED。
- [vllm-ascend #11137](https://github.com/vllm-project/vllm-ascend/pull/11137)：A5 MegaMoe 单 C kernel，OPEN、CONFLICTING。
- [vllm-ascend #13655](https://github.com/vllm-project/vllm-ascend/pull/13655)：A5 Kimi-K3 W4A8 MXFP + SiTU，OPEN、MERGEABLE。

### 其他

- [FastDeploy #7943](https://github.com/PaddlePaddle/FastDeploy/pull/7943) / [#8038](https://github.com/PaddlePaddle/FastDeploy/pull/8038)：WFP4A8 MegaMoE，MERGED。
- [cuDNN Frontend #448](https://github.com/NVIDIA/cudnn-frontend/pull/448)：统一 `MoeEp` Python API + CuTeDSL backend，OPEN、CONFLICTING。

## 建议跟踪顺序

1. **Hopper 收敛**：观察 FlashInfer `#4113` 的 AOT/check 修复与最终合入；同时看 `#4069` 是否继续作为独立 push backend，DeepGEMM `#383` 是否解决 `nv_dev` 冲突。
2. **Serving zero-copy**：先看 FlashInfer `#4341` API 是否合入，再判断 SGLang `#33571` 是否取代或吸收 `#31470`。
3. **vLLM FlashInfer backend**：关注 `#49636` 的 pre-run failure、review 和 EPLB 缺口。
4. **Kimi-K3 SiTU**：DeepGEMM `#396` upstream 后，再看 TensorRT-LLM `#17063` 是否去除开发 fork，并补 fresh B200/B300 runtime 验证。
5. **AMD/Ascend 扩展**：FlyDSL `#972`、AITER `#4439`、vLLM Ascend `#13655` 是非 NVIDIA 路线中最值得继续跟踪的三个增量。

## 证据边界

- GitHub 的 `MERGEABLE` 只说明当前 diff 可以合并，不代表 review、CI、硬件验证或发布条件已经完成。
- aggregate check 失败不能直接等同于 kernel 逻辑失败；本文没有逐一追踪所有 leaf job 的根因。
- PR benchmark 使用不同 GPU、EP 拓扑、模型 geometry、token bucket、quantization 和 baseline，不能横向直接排名。
- 本文是 PR/source 状态审计，没有在 B200/B300/H200/H20/MI355X/Ascend A5 上重新执行这些实现。
