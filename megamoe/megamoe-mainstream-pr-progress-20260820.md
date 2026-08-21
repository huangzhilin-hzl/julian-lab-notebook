# MegaMoE 主流实现与接入进展（2026-08-20）

> 状态快照：2026-08-20，Asia/Shanghai。本文中的 `OPEN`、`MERGED`、`MERGEABLE`、`CONFLICTING`、`APPROVED`、`BLOCKED` 和 checks 状态以本次 GitHub 查询为准。

## 结论

截至 2026-08-20，MegaMoE 生态出现了四个比 8 月 13 日更明确的变化：

1. **FlashInfer 的 serving 基础链进一步闭合**：workspace output view `#4341` 已于 8 月 18 日合入，SM100 BF16 backend `#4386` 和 zero-token livelock 修复 `#4531` 也已合入。FlashInfer 的主线已不只覆盖量化 MegaMoE kernel，也开始补齐零拷贝、BF16 baseline 和真实不均匀路由下的稳定性。
2. **消费框架开始接近可合入，但还没有完全收口**：vLLM 的 FlashInfer backend `#49636` 已获批准且当前 diff 可合并，但 GitHub merge state 仍是 `BLOCKED`；SGLang `#31470` 持续跟进 `main`，但仍待 review 且 checks 有多项失败。旧 zero-copy 草案 `#33571` 的 FlashInfer 依赖虽已合入，自身仍停在 8 月 5 日、处于 draft + conflicting。
3. **优化重心转向 shared expert、新精度和新架构**：DeepGEMM `#409`、vLLM `#53040` 都在推进 fused shared expert；FlashInfer 已合入 SM100 BF16，并新增 SM90 NVFP4、SM120 MXFP8/W4A8 和 SM107 Rubin 候选。多数新架构 PR 仍有冲突、stacked dependency 或未完成性能验证，不能视为生产可用。
4. **非 NVIDIA 路线的 engine 接入继续向上层推进**：FlyDSL `#972` 仍在活跃开发；vLLM `#51918` 与 SGLang `#35619` 开始消费 AITER/FlyDSL MegaMoEV2；Ascend 则同时推进 EPLB、A5 fused backend 和 Kimi-K3 enablement，但分散在 `main`、release 和 RFC 分支，成熟度不一致。

当前主链可以简化为：

```text
DeepGEMM SM100:
  #304 -> #316 -> #364 -> #377 -> #396                  已合入
  #404 cache policy / #409 fused shared expert          开放

FlashInfer:
  SM100 base: #3686 -> #3852 -> #3980 -> #4079 -> #4101 已合入
  serving:    #4341 zero-copy / #4531 zero-token fix     已合入
  precision:  #4386 BF16                                 已合入
  Hopper:     #4113 pull / #4069 push                    已合入
              #4589 NVFP4 W4A8 push                      开放、冲突
  SM120:      #4387 MXFP8 -> #4632 W4A8                  开放、冲突
  SM107:      #4601 Rubin                                开放、冲突

Serving integrations:
  TensorRT-LLM: Kimi-K3 main cherry-pick #17624          已合入
  vLLM native:  #40860 / #43339 / #43632 / #52445       已合入
  vLLM FlashInfer: #49636                                APPROVED、MERGEABLE、BLOCKED
  SGLang native: #23882 / #25052 / #29016 / #34883      已合入
  SGLang FlashInfer: #31470 / #33571                     开放

Other hardware:
  ROCm: FlyDSL #972, vLLM #51918, SGLang #35619          开放
  Ascend: #13994 / #14449 / #14495 / #14664             开放
```

## 相比 2026-08-13 的主要变化

| 路线 | PR | 2026-08-20 状态 | 本轮变化或新信息 |
| --- | --- | --- | --- |
| FlashInfer zero-copy | [flashinfer#4341](https://github.com/flashinfer-ai/flashinfer/pull/4341) | MERGED，2026-08-18 | `return_workspace_view` 和 backend capability 已进入主线；SGLang 的依赖阻塞已从“上游 API 未合入”变成“下游分支尚未刷新”。 |
| FlashInfer SM100 BF16 | [flashinfer#4386](https://github.com/flashinfer-ai/flashinfer/pull/4386) | MERGED，2026-08-19 | 增加无量化 BF16 fused MegaMoE backend，可作为 BF16 checkpoint 路径和量化 backend 的精度/性能基线。 |
| FlashInfer zero-token 稳定性 | [flashinfer#4531](https://github.com/flashinfer-ai/flashinfer/pull/4531) | MERGED，2026-08-19 | 修复 MXFP8/NVFP4 in-kernel FC2 reduce 在某 rank 收到 0 token 时的 livelock；该问题来自真实 SGLang 路由分布。 |
| FlashInfer SM120 | [flashinfer#4387](https://github.com/flashinfer-ai/flashinfer/pull/4387) | OPEN、APPROVED、CONFLICTING | MXFP8 backend 已做功能正确性验证，但仍在调优且当前与 `main` 冲突；aggregate summary 失败，stacked W4A8 `#4632` 也处于冲突状态。 |
| FlashInfer SM90 W4A8 | [flashinfer#4589](https://github.com/flashinfer-ai/flashinfer/pull/4589) | OPEN、CONFLICTING、REVIEW_REQUIRED | 新增 NVFP4 checkpoint 的 packed/folded/hot/dual load-time policy，但尚未进入主线，aggregate summary 当前失败。 |
| FlashInfer Rubin | [flashinfer#4601](https://github.com/flashinfer-ai/flashinfer/pull/4601) | OPEN、CONFLICTING、REVIEW_REQUIRED | 新增 SM107 NVFP4/MXFP8 backend；性能尚未给出，依赖尚未公开发布的 Rubin CuTe DSL codegen，aggregate summary 当前失败。 |
| DeepGEMM shared expert | [DeepGEMM#409](https://github.com/deepseek-ai/DeepGEMM/pull/409) | OPEN draft、MERGEABLE | 在 `nv_dev` 上增加 SM100 NVFP4 MegaMoE + fused BF16 shared expert；PR 作者报告 fusion gain 约 9.8%–19.8%，仍需 upstream review。 |
| vLLM FlashInfer | [vllm#49636](https://github.com/vllm-project/vllm/pull/49636) | OPEN、APPROVED、MERGEABLE、BLOCKED | 8 月 19 日完成跟随 `main`、DCO/pre-commit 修复并获批准；当前可见 checks 没有失败项，但 merge state 仍不是可直接合入。 |
| vLLM Kimi-K3 | [vllm#52445](https://github.com/vllm-project/vllm/pull/52445) | MERGED，2026-08-15 | 修正 `situ_beta` / `situ_linear_beta` 参数名，消除 Kimi-K3 首次 MegaMoE forward 的调用时 `TypeError`。 |
| SGLang FlashInfer | [sglang#31470](https://github.com/sgl-project/sglang/pull/31470) | OPEN、MERGEABLE、BLOCKED、REVIEW_REQUIRED | 8 月 19 日仍在合并 `main`；`#4341` 已合入，但该 PR 仍无批准且 aggregate checks 有多项失败。 |
| SGLang zero-copy | [sglang#33571](https://github.com/sgl-project/sglang/pull/33571) | OPEN draft、CONFLICTING | 依赖的 `flashinfer#4341` 已合入，但该分支自 8 月 5 日无提交更新，当前仍冲突。 |
| TensorRT-LLM Kimi-K3 | [TensorRT-LLM#17624](https://github.com/NVIDIA/TensorRT-LLM/pull/17624) | MERGED，2026-08-15 | 将 `#17063` 从 `feat/kimi_k3` cherry-pick 到 `main`；Kimi-K3 SiTU MegaMoE 这次才真正进入 TensorRT-LLM 主线。 |
| AMD A4W4 | [FlyDSL#972](https://github.com/ROCm/FlyDSL/pull/972) | OPEN、MERGEABLE、BLOCKED、REVIEW_REQUIRED | 8 月 20 日继续更新 hotspot tuning 和代码风格；当前无失败 check，但仍有 pending checks。 |
| Ascend A5 | [vllm-ascend#13655](https://github.com/vllm-project/vllm-ascend/pull/13655) | OPEN、CONFLICTING | 从 8 月 13 日的可合并退回冲突状态，目标为 `releases/v0.26.0rc`，且 pre-commit 失败。 |

## 1. DeepGEMM：原生 MegaMoE

### SM100 基础链已经合入

| PR | 状态 | 定位 |
| --- | --- | --- |
| [#304](https://github.com/deepseek-ai/DeepGEMM/pull/304) | MERGED | 首次公开 SM100 FP8 activation × FP4 weight MegaMoE。 |
| [#316](https://github.com/deepseek-ai/DeepGEMM/pull/316) | MERGED | 第一轮正式优化与 DeepSeek-V4 EP8 benchmark。 |
| [#364](https://github.com/deepseek-ai/DeepGEMM/pull/364) | MERGED | 增加 BF16 MegaMoE 等能力。 |
| [#377](https://github.com/deepseek-ai/DeepGEMM/pull/377) | MERGED | 新 scheduler、融合 shared experts；PR 将路径概括为 11 kernels 合为 1 个 persistent kernel。 |
| [#396](https://github.com/deepseek-ai/DeepGEMM/pull/396) | MERGED | 为 SM100 FP8×FP4 MegaMoE 增加 SiTU JIT specialization。 |

### 新增的 SM100 增量

- [#404](https://github.com/deepseek-ai/DeepGEMM/pull/404)：OPEN、MERGEABLE、`CLEAN`。它为 `num_tokens <= 512` 的 FP8×FP4 MegaMoE 增加 opt-in `EVICT_FIRST` weight-load hint；默认路径不变。作者在 8×SM100 上报告 8/16/32 token 每 rank 改善 3.2%–5.3%，这仍是 PR 指定环境下的微基准。
- [#409](https://github.com/deepseek-ai/DeepGEMM/pull/409)：OPEN draft、MERGEABLE、目标分支为 `nv_dev`。它将 BF16 shared expert 融合进 SM100 NVFP4 MegaMoE，并在最终 add 前应用 routed scaling。PR 给出的 GLM-5.2 层级回放和 8-GPU 数据属于作者证据，尚未进入 `main`。

### SM90 仍未形成统一 upstream

- [#323](https://github.com/deepseek-ai/DeepGEMM/pull/323)：SM90 FP8 fused path，OPEN、CONFLICTING；8 月 14 日有公开活动，但最后代码提交仍早于本轮。
- [#360](https://github.com/deepseek-ai/DeepGEMM/pull/360)：cooperative 单 kernel 方案，OPEN、CONFLICTING，8 月 3 日后无更新。
- [#383](https://github.com/deepseek-ai/DeepGEMM/pull/383)：destination-rank pull、L1/L2 两个大 kernel，OPEN、CONFLICTING，目标仍是 `nv_dev`。8 月 17 日新增了与 `sgl-project/DeepGEMM#36` 的 H20 对比讨论，但没有解决 upstream 冲突。

这意味着 FlashInfer 已经把两条 SM90 backend 合入主线，但 DeepGEMM 自己的 Hopper 路线仍是多个候选并行。

### 本地性能证据

[humming_mxfp4afp8_deepep_pr383_h20_3e_20260811.md](./sm90_humming_mxfp4afp8_megamoe/humming_mxfp4afp8_deepep_pr383_h20_3e_20260811.md) 提供了 H20/H20-3e 描述性比较。由于执行图和计时范围不同，这些结果不能当作可替换 backend 的端到端加速比。

## 2. FlashInfer `moe_ep`

### SM100 基础链与 serving 能力

| PR | 状态 | 定位 |
| --- | --- | --- |
| [#3686](https://github.com/flashinfer-ai/flashinfer/pull/3686) | MERGED | `MoEEpLayer` 和 NCCL-EP/NIXL-EP split 基础。 |
| [#3852](https://github.com/flashinfer-ai/flashinfer/pull/3852) | MERGED | 接入 `deep_gemm_mega`、CuTeDSL NVFP4、CuTeDSL MXFP8。 |
| [#3980](https://github.com/flashinfer-ai/flashinfer/pull/3980) | MERGED | 重组 CuTeDSL kernel drop，补 tuning、FC2 reduce 和量化 combine。 |
| [#4079](https://github.com/flashinfer-ai/flashinfer/pull/4079) | MERGED | CUDA Graph、fused quant+stage、预量化权重、workspace pool、持久 knob cache。 |
| [#4101](https://github.com/flashinfer-ai/flashinfer/pull/4101) | MERGED | 修复 CuTeDSL 4.5.2 mainloop 性能回退。 |
| [#4341](https://github.com/flashinfer-ai/flashinfer/pull/4341) | MERGED | opt-in workspace output view；默认仍返回 caller-owned tensor。 |
| [#4386](https://github.com/flashinfer-ai/flashinfer/pull/4386) | MERGED | SM100 BF16 activation/weight/output fused backend。 |
| [#4531](https://github.com/flashinfer-ai/flashinfer/pull/4531) | MERGED | 修复零 token rank 的 in-kernel FC2 reduce livelock。 |

[FlashInfer #4529](https://github.com/flashinfer-ai/flashinfer/pull/4529) 也已于 8 月 20 日合入，但它是 SM100 W4A8 **split path**：通过 MXFP8 packed dispatch 降低 NCCL-EP payload，不应误记成新的 whole-layer MegaMoE backend。

### Hopper：FP8 已合入，NVFP4 仍在候选阶段

| PR | 通信/执行模型 | 当前状态 |
| --- | --- | --- |
| [#4069](https://github.com/flashinfer-ai/flashinfer/pull/4069) | source-rank push；deduplicated dispatch、owner-side grouped combine | MERGED，2026-08-12。 |
| [#4113](https://github.com/flashinfer-ai/flashinfer/pull/4113) | NVSHMEM pull-style；dispatch + FC1 + SwiGLU + FC2 + combine 单 launch | MERGED，2026-08-08。 |
| [#4589](https://github.com/flashinfer-ai/flashinfer/pull/4589) | SM90 NVFP4 checkpoint；packed/folded/hot/dual residency policy | OPEN、CONFLICTING、REVIEW_REQUIRED。 |

`#4589` 的 folded 路径本质上是在 load time 将 NVFP4 权重转换为现有 FP8 push engine 可消费的格式；packed 路径才是 W4A8 in-kernel decode。两者的显存和性能权衡不同，不能只按同一个 “SM90 NVFP4 backend” 口径比较。

### SM120 与 Rubin：功能进入评审，生产条件尚未满足

- [#4387](https://github.com/flashinfer-ai/flashinfer/pull/4387)：SM120/SM121 MXFP8 swap-AB fused backend，APPROVED 但当前 CONFLICTING。PR 明确写明仍在性能调优，并对已知的 in-kernel reduce 和 cluster 配置问题做拒绝式 guard。
- [#4632](https://github.com/flashinfer-ai/flashinfer/pull/4632)：stacked 在 `#4387` 上的 SM120 W4A8 backend。标题含 `[Draft]`，但 GitHub `isDraft=false`；当前为 OPEN、CONFLICTING、REVIEW_REQUIRED，且尚未拆出依赖后的净 diff。
- [#4601](https://github.com/flashinfer-ai/flashinfer/pull/4601)：SM107 Rubin NVFP4/MXFP8 backends，OPEN、CONFLICTING。它依赖尚未公开发布的 Rubin CuTe DSL codegen，PR 也尚未给出性能数据。
- [#4604](https://github.com/flashinfer-ai/flashinfer/pull/4604)：MXFP8×BF16 integration，OPEN draft、MERGEABLE、BLOCKED；描述和验证信息仍不完整，`pre-commit` 与 aggregate summary 当前失败。

### Zero-copy 上游已合入，下游尚未完成迁移

`#4341` 已把以下 opt-in 契约合入 FlashInfer：

```text
默认：workspace output --copy--> caller-owned tensor
可选：workspace output view -----> framework downstream
```

因此当前阻塞点已不再是 FlashInfer API，而是消费方是否更新依赖、rebase 并完成 CI。SGLang `#31470` 在 8 月 19 日已经公开确认依赖合入并继续同步 `main`；`#33571` 则仍停在旧分支。

## 3. TensorRT-LLM

TensorRT-LLM 的已有 engine 级 MegaMoE 链仍然成立：

- [#13384](https://github.com/NVIDIA/TensorRT-LLM/pull/13384)：`MegaMoEDeepGemmFusedMoE`，MERGED。
- [#14608](https://github.com/NVIDIA/TensorRT-LLM/pull/14608)：SM100/SM103 NVFP4 `MegaMoECuteDsl`，MERGED。
- [#16190](https://github.com/NVIDIA/TensorRT-LLM/pull/16190)：CuTeDSL kernel、tactic、workspace/CUDA Graph 和 combine 更新，MERGED。
- [#17063](https://github.com/NVIDIA/TensorRT-LLM/pull/17063)：Kimi-K3 SiTU MegaMoE，8 月 13 日先合入 `feat/kimi_k3`。
- [#17624](https://github.com/NVIDIA/TensorRT-LLM/pull/17624)：将 `#17063` cherry-pick 到 `main`，8 月 15 日 MERGED；默认 backend 仍不是 MegaMoE。

8 月 13 日后的增量包括：

- [#17532](https://github.com/NVIDIA/TensorRT-LLM/pull/17532)：MERGED，将 MoE backend 选择改为可复现、可报告的 resolution contract。
- [#17865](https://github.com/NVIDIA/TensorRT-LLM/pull/17865)：Kimi-K3 NVFP4 的 CUTLASS + CuTeDSL MegaMoE SiTU，OPEN、APPROVED、MERGEABLE、BLOCKED；当前仍有 pending check。
- [#17907](https://github.com/NVIDIA/TensorRT-LLM/pull/17907)：MiniMax-M3 的 SwiGLUBias MegaMoE，已合入 `feat/m3_with_msa`，不是 `main`。
- [#17956](https://github.com/NVIDIA/TensorRT-LLM/pull/17956)：将 mixed-CGA、work-claiming 和 epilogue 优化移植到 TRT-LLM，OPEN draft，尚无性能表。

因此 Kimi-K3 DeepGEMM SiTU 路线已经进入 `main`，NVFP4 CuTeDSL 路线则仍在合入前阶段；MiniMax-M3 仍需区分 feature branch 和主线状态。

## 4. vLLM 与 SGLang

### vLLM

已合入的 native DeepGEMM 路径：

- [#40860](https://github.com/vllm-project/vllm/pull/40860)：DeepSeek-V4 和 `deep_gemm_mega_moe` 主接入；
- [#43339](https://github.com/vllm-project/vllm/pull/43339)：MegaMoE EPLB；
- [#43632](https://github.com/vllm-project/vllm/pull/43632)：input-prep kernel 迁移到 `nvidia/ops`；
- [#51146](https://github.com/vllm-project/vllm/pull/51146)：修正 Kimi-K3 MegaMoE path 的额外 add；
- [#52445](https://github.com/vllm-project/vllm/pull/52445)：修正 Kimi-K3 SiTU 参数名，避免首次 MegaMoE forward 直接报错。

[vLLM #49636](https://github.com/vllm-project/vllm/pull/49636) 仍是 FlashInfer `moe_ep` 的关键入口。它增加 `flashinfer_moe_ep_mega_deep_gemm` 和 `flashinfer_moe_ep_mega_cutedsl`，保持 opt-in，并明确限制为 DeepSeek-V4。当前已 APPROVED、MERGEABLE，但 merge state 是 BLOCKED；因此它比 8 月 13 日更接近合入，却还不能写成“随时可合”。

新的 native 路线重点是 shared expert：

- [#53040](https://github.com/vllm-project/vllm/pull/53040)：OPEN、MERGEABLE、REVIEW_REQUIRED，将 DeepSeek-V4 的 replicated FP8 shared expert 融入 DeepGEMM persistent MegaMoE；PR 作者报告 batch-size-1 输出吞吐约 +13.7%–15.0%，并发 64 workload 输出吞吐约 +9.4%。
- [#52705](https://github.com/vllm-project/vllm/pull/52705)：相邻 draft，增加 DeepGEMM NVFP4 + fused shared expert，当前 pre-commit 失败；评审时应避免把两条分支的证据混在一起。
- [#50647](https://github.com/vllm-project/vllm/pull/50647)：Kimi-K3 EPLB，OPEN draft、MERGEABLE、BLOCKED，仍在扩展 physical/redundant expert metadata。

ROCm 消费方 [#51918](https://github.com/vllm-project/vllm/pull/51918) 是 opt-in FlyDSL MegaMoEV2 backend，当前 OPEN draft、CONFLICTING，且 `pre-commit`/`pre-run-check` 失败，尚未形成可合入状态。

### SGLang

已合入的 native DeepGEMM 路径继续扩展：

- [#23882](https://github.com/sgl-project/sglang/pull/23882)：DeepSeek-V4 主接入；
- [#25052](https://github.com/sgl-project/sglang/pull/25052)：W4A4 MegaMoE；
- [#29016](https://github.com/sgl-project/sglang/pull/29016)：SM90 FP8 MegaMoE；
- [#34883](https://github.com/sgl-project/sglang/pull/34883)：Kimi-K3 显式使用 SiTU activation，8 月 15 日 MERGED；
- [#35372](https://github.com/sgl-project/sglang/pull/35372)：扩展 `mega_moe_pre_dispatch` 的 wide-row 支持，8 月 20 日 MERGED。

FlashInfer 消费方仍未合入：

- [#31470](https://github.com/sgl-project/sglang/pull/31470)：OPEN、MERGEABLE、BLOCKED、REVIEW_REQUIRED。它在 8 月 14/18/19 日持续合并 `main`，是当前更活跃的候选，但 GitHub aggregate checks 仍有多项失败。
- [#33571](https://github.com/sgl-project/sglang/pull/33571)：OPEN draft、CONFLICTING。它依赖的 `flashinfer#4341` 已合入，但自身最后提交仍是 8 月 5 日，尚未 rebase 或吸收到 `#31470`。

新的性能/精度候选尚处早期：

- [#35098](https://github.com/sgl-project/sglang/pull/35098)：direct TopK output 与 side-stream pre-dispatch overlap，OPEN、CONFLICTING、CHANGES_REQUESTED；PR 的 13.6%–19.9% 只针对前端 micro-chain，不是整机吞吐。
- [#35459](https://github.com/sgl-project/sglang/pull/35459)：MXFP8×BF16 MegaMoE，OPEN draft，描述、测试和 benchmark 尚未补齐。
- [#35619](https://github.com/sgl-project/sglang/pull/35619)：AITER/FlyDSL MegaMoEV2 for DeepSeek-V4，OPEN draft；PR 报告 8×MI355X 数据，但明确写明 rebase 到当前 `main` 后的 GPU e2e revalidation 仍待完成。

## 5. AMD、Ascend 及其他硬件生态

### AMD

- [FlyDSL #876](https://github.com/ROCm/FlyDSL/pull/876)：MI355X/gfx950 A8W4 MegaMoEV2，MERGED。
- [FlyDSL #972](https://github.com/ROCm/FlyDSL/pull/972)：A4W4 + FP8 blockwise P2P，OPEN、MERGEABLE、BLOCKED、REVIEW_REQUIRED；8 月 20 日仍在活跃更新。
- [AITER #4439](https://github.com/ROCm/aiter/pull/4439)：A8W4 MegaMoEV2 engine integration，MERGED。
- [AITER #4757](https://github.com/ROCm/aiter/pull/4757)：Mori HIP dispatch backend 已合入 `yanbo/mega_stage2_gfx1250` feature branch，不是 `main`。
- [AITER #4785](https://github.com/ROCm/aiter/pull/4785)：gfx1250 MegaMoE stage2 CI/feature，OPEN、CONFLICTING。

AMD 当前的关键问题已经从“是否有 kernel”转为“vLLM/SGLang 的 opt-in engine 接入能否 rebase、补齐 CI 并形成可复现的默认选择条件”。

### Ascend

- [vllm-ascend #11701](https://github.com/vllm-project/vllm-ascend/pull/11701)：A3 CANN `mega_moe`，MERGED。
- [#11137](https://github.com/vllm-project/vllm-ascend/pull/11137)：旧 A5 MegaMoe 单 C kernel，OPEN、CONFLICTING。
- [#13655](https://github.com/vllm-project/vllm-ascend/pull/13655)：A5 Kimi-K3 W4A8 MXFP + SiTU，OPEN、CONFLICTING，目标为 `releases/v0.26.0rc`。
- [#13994](https://github.com/vllm-project/vllm-ascend/pull/13994)：MegaMoE EPLB，OPEN、MERGEABLE、BLOCKED、REVIEW_REQUIRED，目标为 `main`。
- [#14358](https://github.com/vllm-project/vllm-ascend/pull/14358)：MegaMoE prefill buffer sizing 修复，8 月 20 日合入 `releases/v0.26.0rc`。
- [#14449](https://github.com/vllm-project/vllm-ascend/pull/14449)：A5 fused backend，OPEN、MERGEABLE、`CLEAN`，但目标是 `rfc/vllm_cann` 而非 `main`。
- [#14495](https://github.com/vllm-project/vllm-ascend/pull/14495)：Kimi-K3 enablement，OPEN、CONFLICTING，目标为 `releases/v0.26.0rc`。
- [#14664](https://github.com/vllm-project/vllm-ascend/pull/14664)：将 MegaMoE 设为唯一 FusedMC2 path 的重构，OPEN、MERGEABLE、BLOCKED，DCO/CI/pre-commit 当前失败。

这里必须按目标分支分别判断：release 分支已合入 bugfix，不等于新 backend 已进入 `main`；RFC 分支可合并，也不等于 production 路线已定。

### 其他

- [FastDeploy #7943](https://github.com/PaddlePaddle/FastDeploy/pull/7943) / [#8038](https://github.com/PaddlePaddle/FastDeploy/pull/8038)：WFP4A8 MegaMoE，MERGED。
- [cuDNN Frontend #448](https://github.com/NVIDIA/cudnn-frontend/pull/448)：统一 `MoeEp` Python API + CuTeDSL backend，OPEN draft、REVIEW_REQUIRED；8 月 10 日后无更新。

## 建议跟踪顺序

1. **vLLM `#49636` 的最后 gate**：它已经获批准且 diff 可合并，下一步要定位 `BLOCKED` 的精确门禁并确认合入后依赖的 FlashInfer 版本。
2. **SGLang FlashInfer 路线是否收敛**：优先看 `#31470` 能否清理 CI/review，并决定是否吸收 `#33571` 的 int32 router ID + output view 优化。
3. **fused shared expert 是否 upstream**：DeepGEMM `#409` 仍在 `nv_dev` draft，vLLM `#53040` 已给出 E2E 数据；需要先确认 kernel API/分支落点，再判断 framework patch 的可合入性。
4. **SM120/Rubin 依赖链**：`#4387 -> #4632` 需要先解决冲突和净 diff，`#4601` 需要等待可公开消费的 Rubin CuTe DSL 并补性能数据。
5. **DeepGEMM Hopper 是否继续 upstream**：`#323/#360/#383` 仍全部冲突；本地 H20 比较增加了性能证据，但没有改变它们的合入状态。
6. **ROCm/Ascend engine 接入**：ROCm 重点看 FlyDSL `#972` 与 SGLang `#35619`；Ascend 重点看 `#13994` 是否进入主线，以及 `#14449/#14495/#14664` 如何从 RFC/release 分支收口。

## 证据边界

- `MERGEABLE` 只说明当前 diff 没有文本冲突；`BLOCKED`、review、CI、目标分支策略和硬件验证仍可能阻止合入。
- `APPROVED` 不等于已经进入 merge queue；`MERGED` 也不等于默认启用、正式 release 或所有硬件都验证通过。
- PR benchmark 与本地 bench 的 GPU、EP/TP/DP 拓扑、模型 geometry、token bucket、量化格式、计时边界和 baseline 不统一，不能直接横向排名。
- 对非 `main` 目标分支的 merged PR，本文明确标出 feature/release branch；不能仅根据 GitHub 的 `MERGED` 状态写成“已进入主线”。
- 本文混合两类证据：GitHub PR 状态回答“是否进入指定分支”，本地实验回答“在指定环境下表现如何”；两者不能相互替代。
