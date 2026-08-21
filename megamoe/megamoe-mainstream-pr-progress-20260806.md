# MegaMoE 主流实现与接入进展（2026-08-21）

> 状态快照：2026-08-21，Asia/Shanghai。本文中的 `OPEN`、`MERGED`、`MERGEABLE`、`CONFLICTING`、`APPROVED`、`BLOCKED` 和 checks 状态以本次 GitHub 查询为准。

## 结论

截至 2026-08-21，相比昨天的快照有六项值得立即调整的判断：

1. **deepseek-ai/DeepGEMM Hopper 出现新的 FP8×MXFP4 候选**：`#411` 复用 SM100 ring-buffer/wave-interleaved 思路，在 SM90 上融合 dispatch、routed/shared L1/L2 和 combine。它当前 `MERGEABLE/CLEAN`，但 review bot 已指出集中路由可能产生错误输出、各 rank token 数不一致时可能死锁，因此不能把“无文本冲突”写成“可生产合入”。
2. **sgl-project/DeepGEMM 已形成独立的 SGLang 生产推进线**：SM90 FP8 基础 `#36`、decode swap-AB `#48`、连续 FP32 activation scale `#63` 已合入该 fork 的 `dev`，并由 SGLang `#29016` 接入 `main`。当前 `#53/#68/#69/#74` 均对 `dev` 可合并，但没有 recorded approval 或公开 status checks；其中 `#53` 的 SGLang FP4 集成尚未 upstream，`#69` 则在 8 月 20 日继续更新 fused shared-expert/scheduler 优化。
3. **vLLM shared-expert 融合已经进入主线**：`#53040` 于北京时间 8 月 21 日合入，将 replicated FP8 shared expert 融入 DeepGEMM persistent MegaMoE。vLLM 的 FlashInfer backend `#49636` 也已从 `BLOCKED` 推进到 `APPROVED/MERGEABLE/CLEAN`，但仍未合入。
4. **FlashInfer SM90 NVFP4 已解决文本冲突**：`#4589` 从 `CONFLICTING` 变为 `MERGEABLE`，但 merge state 仍为 `BLOCKED`，需要 review，aggregate summary 仍失败。
5. **DeepGEMM `#409` 撤回了旧性能表**：原表混用了 FlashInfer 通信和 CUDA Graph critical-path timing，与 DeepGEMM repository baseline 不对齐。GB300 EP8 的 Kineto/DeepEP 对齐复测尚未补齐，昨天记录的 fusion gain 不应继续引用。
6. **非 NVIDIA shared-expert 与 Ascend 主线候选增加**：vLLM `#53161` 新增 ROCm heterogeneous shared-expert draft，但明确标记 `DO NOT MERGE` 并依赖 AITER `#4891`；vLLM Ascend `#14658/#14664` 则开始把 MegaMoe 推向 `main` 的 A5/FusedMC2 唯一路径。

当前主链可以简化为：

```text
DeepGEMM SM100:
  #304 -> #316 -> #364 -> #377 -> #396                  已合入
  #404 cache policy / #409 fused shared expert          开放
DeepGEMM SM90:
  #323 / #360 / #383                                    开放、冲突
  #411 FP8 x MXFP4                                      开放、CLEAN、review 有正确性阻塞

sgl-project/DeepGEMM fork:
  SM90 FP8: #36 -> #48 -> #63                           已合入 dev
  SM90 FP4: #53 / correctness: #68 / perf: #69          开放、CLEAN
  nvcc13 compile fix: #74                               开放、CLEAN

FlashInfer:
  SM100 base: #3686 -> #3852 -> #3980 -> #4079 -> #4101 已合入
  serving:    #4341 zero-copy / #4531 zero-token fix     已合入
  precision:  #4386 BF16                                 已合入
  Hopper:     #4113 pull / #4069 push                    已合入
              #4589 NVFP4 W4A8 push                      开放、MERGEABLE、BLOCKED
  SM120:      #4387 MXFP8 -> #4632 W4A8                  开放、冲突
  SM107:      #4601 Rubin                                开放、冲突

Serving integrations:
  TensorRT-LLM: Kimi-K3 main cherry-pick #17624          已合入
  vLLM native:  #40860 / #43339 / #43632 / #52445 / #53040 已合入
  vLLM FlashInfer: #49636                                APPROVED、MERGEABLE、CLEAN
  SGLang native: #23882 / #25052 / #29016 / #34883      已合入
  SGLang FlashInfer: #31470 / #33571                     开放

Other hardware:
  ROCm: FlyDSL #972, vLLM #51918/#53161, SGLang #35619   开放
  Ascend: #13994 / #14449 / #14495 / #14658 / #14664    开放
```

## 相比 2026-08-20 的主要变化

| 路线 | PR | 2026-08-21 状态 | 本轮变化或新信息 |
| --- | --- | --- | --- |
| DeepGEMM SM90 MXFP4 | [DeepGEMM#411](https://github.com/deepseek-ai/DeepGEMM/pull/411) | OPEN、MERGEABLE、`CLEAN` | 新增 FP8 activation × MXFP4 weight persistent MegaMoE，目标为 `nv_dev`；review bot 已指出集中路由错误与不等 token 潜在死锁，尚无人工批准。 |
| DeepGEMM shared expert | [DeepGEMM#409](https://github.com/deepseek-ai/DeepGEMM/pull/409) | OPEN draft、MERGEABLE、`CLEAN` | 旧性能表因 baseline/计时不对齐被作者撤回；GB300 EP8 的 DeepEP/Kineto 对齐复测仍待补。 |
| SGL DeepGEMM fork | [sgl-DeepGEMM#53](https://github.com/sgl-project/DeepGEMM/pull/53) / [#68](https://github.com/sgl-project/DeepGEMM/pull/68) / [#69](https://github.com/sgl-project/DeepGEMM/pull/69) / [#74](https://github.com/sgl-project/DeepGEMM/pull/74) | 全部 OPEN、MERGEABLE、`CLEAN` | 本次补录 SGLang 自维护 fork：SM90 FP4 集成、pruned-weight 正确性修复、FP8 fused shared-expert/scheduler 优化、nvcc 13 编译修复；四条均目标 `dev`，无 recorded approval/公开 status checks。 |
| vLLM shared expert | [vllm#53040](https://github.com/vllm-project/vllm/pull/53040) | MERGED，2026-08-21（北京时间） | replicated FP8 shared expert 已融合进 DeepGEMM persistent MegaMoE；PR 报告 batch-size-1 输出吞吐约 +13.7%–15.0%，并发 64 workload 约 +9.4%。 |
| vLLM FlashInfer | [vllm#49636](https://github.com/vllm-project/vllm/pull/49636) | OPEN、APPROVED、MERGEABLE、`CLEAN` | 相比昨天已解除 `BLOCKED`；当前可见 checks 无失败，但仍未合入。 |
| FlashInfer SM90 W4A8 | [flashinfer#4589](https://github.com/flashinfer-ai/flashinfer/pull/4589) | OPEN、MERGEABLE、BLOCKED、REVIEW_REQUIRED | 已解决与 `main` 的文本冲突；aggregate `Test Results Summary` 仍失败。 |
| vLLM static EPLB | [vllm#53022](https://github.com/vllm-project/vllm/pull/53022) | OPEN draft、CONFLICTING、REVIEW_REQUIRED | 为 DeepSeek-V4/MegaMoE 增加按层静态 expert map 与 replicated slot，当前尚未 rebase。 |
| ROCm shared expert | [vLLM#53161](https://github.com/vllm-project/vllm/pull/53161) | OPEN draft、MERGEABLE、BLOCKED | 新增 AITER heterogeneous shared/routed expert 融合；PR 明确 `DO NOT MERGE`，等待 AITER `#4891` 和匹配 pin。 |
| Ascend A5 main | [vllm-ascend#14658](https://github.com/vllm-project/vllm-ascend/pull/14658) | OPEN、MERGEABLE、BLOCKED、REVIEW_REQUIRED | 面向 `main` 增加 A5 MegaMoe 路径；作者报告与旧 `dispatchFFNcombine` 基本持平，但 CI/pre-commit 当前失败。 |
| Ascend FusedMC2 | [vllm-ascend#14664](https://github.com/vllm-project/vllm-ascend/pull/14664) | OPEN、MERGEABLE、BLOCKED、REVIEW_REQUIRED | 计划让 MegaMoe 成为 `main` 唯一 FusedMC2 实现并删除 legacy fallback；DCO/CI/pre-commit 当前失败。 |

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
- [#409](https://github.com/deepseek-ai/DeepGEMM/pull/409)：OPEN draft、MERGEABLE、`CLEAN`，目标分支为 `nv_dev`。它将 BF16 shared expert 融合进 SM100 NVFP4 MegaMoE，并在最终 add 前应用 routed scaling。作者已删除原性能表，因为其 FlashInfer 通信 baseline 和 CUDA Graph critical-path timing 与 DeepGEMM repository baseline 不对齐；新的 GB300 EP8 DeepEP/Kineto 对齐复测仍待补。

### deepseek-ai/DeepGEMM：SM90 仍未形成统一 upstream

- [#323](https://github.com/deepseek-ai/DeepGEMM/pull/323)：SM90 FP8 fused path，OPEN、CONFLICTING；8 月 14 日有公开活动，但最后代码提交仍早于本轮。
- [#360](https://github.com/deepseek-ai/DeepGEMM/pull/360)：cooperative 单 kernel 方案，OPEN、CONFLICTING，8 月 3 日后无更新。
- [#383](https://github.com/deepseek-ai/DeepGEMM/pull/383)：destination-rank pull、L1/L2 两个大 kernel，OPEN、CONFLICTING，目标仍是 `nv_dev`。8 月 17 日新增了与 `sgl-project/DeepGEMM#36` 的 H20 对比讨论，但没有解决 upstream 冲突。
- [#411](https://github.com/deepseek-ai/DeepGEMM/pull/411)：FP8 activation × MXFP4 weight persistent kernel，OPEN、MERGEABLE、`CLEAN`，目标为 `nv_dev`。它将 SM100 ring-buffer/wave-interleaved scheduler 适配到 Hopper，并保持 routed weights 为 MXFP4、运行时解码为 FP8 WGMMA 输入。当前 review bot 已指出两个直接 blocker：集中路由可能错误，不等 rank token 数可能死锁；这些契约问题解决前不能按 production-ready 评估。

这意味着 deepseek-ai/DeepGEMM 的 Hopper 路线已从“所有候选都冲突”推进到出现一个可 rebase 的新实现，但尚未解决分布式路由正确性；不能据此忽略下方 `sgl-project/DeepGEMM` 已被 SGLang 实际消费的独立 fork 路线。

### sgl-project/DeepGEMM：SGLang 自维护链已经形成

`sgl-project/DeepGEMM` 是独立维护的 fork。它的 MegaMoE PR 多数目标为 `dev`、阶段分支或 release 分支，而仓库默认分支是 `main`；因此下表中的 `MERGED` 只表示进入所列目标分支，是否进入 SGLang 主线还要继续核对 SGLang 的 wheel/version bump 或集成 PR。

两条代码线的能力侧重点对比如下。这里的 SGL release 指 [sgl-project/DeepGEMM `release/v0.1.5`](https://github.com/sgl-project/DeepGEMM/tree/release/v0.1.5)，DeepGEMM main 指 [deepseek-ai/DeepGEMM `main`](https://github.com/deepseek-ai/DeepGEMM/tree/main)：

| 优化项 | SGL `release/v0.1.5` | DeepGEMM `main` |
| --- | :---: | :---: |
| SM90/Hopper MegaMoE | ✅ | ❌ |
| SM90 小 batch swapAB | ✅ | ❌ |
| SM90 连续 FP32 activation scale | ✅ | ❌ |
| FP4 activation + MXF4 双 CTA | ✅ | ❌ |
| FP8 combine，降低 All-to-All 通信量 | ✅ | ❌ |
| Kimi-K3 SiTU 激活 | ✅ | ❌ |
| 独立 fused pre-dispatch | ✅ | ❌ |
| 新版 persistent scheduler | ❌ | ✅ |
| L1/L2 block 交错调度 | ❌ | ✅ |
| shared experts 融合进 MegaMoE | ❌ | ✅ |
| routed/shared 合并为单 kernel | ❌ | ✅ |

`✅` 只表示对应分支包含该能力，不表示默认启用、已经完成 serving 集成或两边硬件范围等价。尤其是后四项，DeepGEMM `main` 的现有实现主要是 SM100 路线；不能据此推导 upstream `main` 已经具备 SM90 persistent/shared-expert MegaMoE。SGL `release/v0.1.5` 的 SM90 能力更完整，但新版 scheduler 与 shared-expert 单 kernel 融合仍在 fork 的开放 PR [#69](https://github.com/sgl-project/DeepGEMM/pull/69) 中。

已经合入 fork 分支的主要演进如下：

| PR | 目标分支 / 状态 | 定位 |
| --- | --- | --- |
| [#27](https://github.com/sgl-project/DeepGEMM/pull/27) / [#28](https://github.com/sgl-project/DeepGEMM/pull/28) | `dev-0426` / MERGED | 增加 FP4 activation + MXF4、fused `mega_moe_pre_dispatch`，以及 second all-to-all 的 FP8 combine；随后由 [#33](https://github.com/sgl-project/DeepGEMM/pull/33) 合入 `release-0426`。 |
| [#36](https://github.com/sgl-project/DeepGEMM/pull/36) | `dev` / MERGED，2026-06-16（北京时间） | SM90 FP8 MegaMoE 主实现；同作者较早的 [#24](https://github.com/sgl-project/DeepGEMM/pull/24) 未合入关闭，由这条 PR 接续。其消费方 SGLang [#29016](https://github.com/sgl-project/sglang/pull/29016) 已进入 `main`。 |
| [#45](https://github.com/sgl-project/DeepGEMM/pull/45) / [#46](https://github.com/sgl-project/DeepGEMM/pull/46) | `dev` / MERGED | 将 Hopper MegaMoE 测试迁入 `sgl_deep_gemm`，并加入 pre-release workflow。 |
| [#48](https://github.com/sgl-project/DeepGEMM/pull/48) / [#57](https://github.com/sgl-project/DeepGEMM/pull/57) | `dev` / MERGED | 优化 SM90 FP8 small-batch decode swap-AB，并将 MegaMoE barrier timeout 提高到 300 秒。 |
| [#63](https://github.com/sgl-project/DeepGEMM/pull/63) | `dev` / MERGED | 将 SM90 activation scale 从 UE8M0 改为连续 FP32，避免 amax 跨越 2 的幂边界时整行 E4M3 grid 发生 2× 跳变；PR 报告 GPQA `no_answer` 从 6.6% 降至 0.03%。 |
| [#67](https://github.com/sgl-project/DeepGEMM/pull/67) / [#78](https://github.com/sgl-project/DeepGEMM/pull/78) | `dev` / MERGED | 先加入 Kimi-K3 SiTU activation，再以显式 `activation="situ"` 取代 sentinel 选择；对应 SGLang [#34883](https://github.com/sgl-project/sglang/pull/34883) 已进入 `main`。 |

截至本次查询，开放 PR 的状态为：

| PR | 当前状态 | 进展与阻塞 |
| --- | --- | --- |
| [#44](https://github.com/sgl-project/DeepGEMM/pull/44) | OPEN、CONFLICTING | SM100 packed FP4×FP4/W4A4 specialization；配套 SGLang [#28210](https://github.com/sgl-project/sglang/pull/28210) 仍 OPEN。最后代码提交是 6 月 12 日（北京时间），尚无 review 或公开 checks，不能按活跃合入候选评估。 |
| [#53](https://github.com/sgl-project/DeepGEMM/pull/53) | OPEN、MERGEABLE、`CLEAN` | SM90 FP8 activation × FP4 weight + small-batch swap-AB。作者 8 月 14 日明确说明 SGLang 集成**尚未 upstream**，当前只在外部分支验证，需先合入本 PR；8 月 20 日新增的 PD-disaggregation 适配问题尚未得到答复。无 review/公开 checks。 |
| [#68](https://github.com/sgl-project/DeepGEMM/pull/68) | OPEN、MERGEABLE、`CLEAN` | 修复 pruned FP8 weight block 保留随机字节、tiny nonzero scale 被 reciprocal 放大后导致错误输出的问题；H20D 最小复现从 `max_abs=1870659584.0` 降到 `0.0`。尚无 review/公开 checks，SGLang workaround 仍需等待合入并发布 wheel 后才能移除。 |
| [#69](https://github.com/sgl-project/DeepGEMM/pull/69) | OPEN、MERGEABLE、`CLEAN` | SM90 FP8 优化总线：interleaved L1/L2 scheduler、fused shared expert、L2 bank swizzle、4-WG heuristic、decode swap-AB。8 月 20 日仍有代码更新并请求 review；PR 提供 8×H20/EP8 sweep，但尚无 approval 或公开 checks。 |
| [#74](https://github.com/sgl-project/DeepGEMM/pull/74) | OPEN、MERGEABLE、`CLEAN` | 为 nvcc 13.x 的 SM90 FP8 MegaMoE 补 `.template` dependent-name disambiguation，是纯编译修复；尚无 review/公开 checks。 |

另外，[#75](https://github.com/sgl-project/DeepGEMM/pull/75) 不是新的 MegaMoE kernel，但与部署链直接相关：它为 SM120 MXFP4 scale-factor layout 增加 CUDA tensor 校验和 `(1, 32)` 回归测试，并报告已发布的 `sgl-deep-gemm 0.1.5.post2` wheel 在 tvm-ffi tensor 边界损坏。该 PR 当前 OPEN、MERGEABLE、`CLEAN`，源代码构建可绕开问题，但正式消费仍依赖重新发 wheel。

这条 fork 路线的判断应分三层：SM90 FP8 已由 `#36/#48/#63` 进入 `dev` 且 SGLang 已有主线消费；SM90 FP4 `#53` 仍缺正式 SGLang upstream；`#68/#69/#74` 虽无文本冲突，但仍缺 review/CI 证据。它和 deepseek-ai/DeepGEMM `#323/#360/#383/#411` 不能混成同一条 upstream 队列。

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
| [#4589](https://github.com/flashinfer-ai/flashinfer/pull/4589) | SM90 NVFP4 checkpoint；packed/folded/hot/dual residency policy | OPEN、MERGEABLE、BLOCKED、REVIEW_REQUIRED；aggregate summary 失败。 |

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
- [#52445](https://github.com/vllm-project/vllm/pull/52445)：修正 Kimi-K3 SiTU 参数名，避免首次 MegaMoE forward 直接报错；
- [#53040](https://github.com/vllm-project/vllm/pull/53040)：将 replicated FP8 shared expert 融入 DeepGEMM persistent MegaMoE，2026-08-21（北京时间）MERGED。

[vLLM #49636](https://github.com/vllm-project/vllm/pull/49636) 仍是 FlashInfer `moe_ep` 的关键入口。它增加 `flashinfer_moe_ep_mega_deep_gemm` 和 `flashinfer_moe_ep_mega_cutedsl`，保持 opt-in，并明确限制为 DeepSeek-V4。当前已 APPROVED、MERGEABLE、`CLEAN`，可见 checks 无失败；状态已比昨天进一步推进，但仍未实际合入。

`#53040` 合入后，NVIDIA native shared-expert 融合已不再是候选。PR 作者报告 batch-size-1 输出吞吐约 +13.7%–15.0%，并发 64 workload 约 +9.4%；这些仍是 PR 指定的 B200/DeepSeek-V4 workload 数据。

仍开放的 native/EPLB 增量包括：

- [#52705](https://github.com/vllm-project/vllm/pull/52705)：相邻 draft，增加 DeepGEMM NVFP4 + fused shared expert，当前 pre-commit 失败；评审时应避免把两条分支的证据混在一起。
- [#50647](https://github.com/vllm-project/vllm/pull/50647)：Kimi-K3 EPLB，OPEN draft、MERGEABLE、BLOCKED，仍在扩展 physical/redundant expert metadata。
- [#53022](https://github.com/vllm-project/vllm/pull/53022)：DeepSeek-V4/MegaMoE static expert maps，OPEN draft、CONFLICTING；支持 permutation 和 replicated physical slots，但尚未 rebase。

ROCm 消费方 [#51918](https://github.com/vllm-project/vllm/pull/51918) 是 opt-in FlyDSL MegaMoEV2 backend，当前 OPEN draft、CONFLICTING，且 `pre-commit`/`pre-run-check` 失败。[#53161](https://github.com/vllm-project/vllm/pull/53161) 则是不同的 TP8/no-EP heterogeneous shared/routed expert 融合，当前 OPEN draft、MERGEABLE、BLOCKED，并明确等待 AITER `#4891`，不能与 `#51918` 当作同一路线。

### SGLang

已合入的 native DeepGEMM 路径继续扩展：

- [#23882](https://github.com/sgl-project/sglang/pull/23882)：DeepSeek-V4 主接入；
- [#25052](https://github.com/sgl-project/sglang/pull/25052)：W4A4 MegaMoE；
- [#29016](https://github.com/sgl-project/sglang/pull/29016)：SM90 FP8 MegaMoE；
- [#34883](https://github.com/sgl-project/sglang/pull/34883)：Kimi-K3 显式使用 SiTU activation，8 月 15 日 MERGED；
- [#35372](https://github.com/sgl-project/sglang/pull/35372)：扩展 `mega_moe_pre_dispatch` 的 wide-row 支持，8 月 20 日 MERGED。

这些 serving PR 背后的库侧实现主要来自 `sgl-project/DeepGEMM`，不是 `deepseek-ai/DeepGEMM` 的同编号 upstream 队列：SM90 FP8 对应 fork 的 `#36/#48/#63`，SiTU 显式选择对应 `#78`。当前 fork 侧 `#53` 的 SM90 FP4 SGLang 集成仍未 upstream，`#68/#69/#74` 也还没有 review/CI 闭环；详见上方独立小节。

FlashInfer 消费方仍未合入：

- [#31470](https://github.com/sgl-project/sglang/pull/31470)：OPEN、MERGEABLE、BLOCKED、REVIEW_REQUIRED。它在 8 月 14/18/19 日持续合并 `main`，是当前更活跃的候选，但 GitHub aggregate checks 仍有多项失败。
- [#33571](https://github.com/sgl-project/sglang/pull/33571)：OPEN draft、CONFLICTING。它依赖的 `flashinfer#4341` 已合入，但自身最后提交仍是 8 月 5 日，尚未 rebase 或吸收到 `#31470`。

新的性能/精度候选尚处早期：

- [#35098](https://github.com/sgl-project/sglang/pull/35098)：direct TopK output 与 side-stream pre-dispatch overlap，OPEN、CONFLICTING、CHANGES_REQUESTED；PR 的 13.6%–19.9% 只针对前端 micro-chain，不是整机吞吐。
- [#35459](https://github.com/sgl-project/sglang/pull/35459)：MXFP8×BF16 MegaMoE，OPEN draft、CONFLICTING，描述、测试和 benchmark 尚未补齐。
- [#35619](https://github.com/sgl-project/sglang/pull/35619)：AITER/FlyDSL MegaMoEV2 for DeepSeek-V4，OPEN draft、MERGEABLE、BLOCKED，aggregate checks 有多项失败；PR 报告 8×MI355X 数据，但明确写明 rebase 到当前 `main` 后的 GPU e2e revalidation 仍待完成。

## 5. AMD、Ascend 及其他硬件生态

### AMD

- [FlyDSL #876](https://github.com/ROCm/FlyDSL/pull/876)：MI355X/gfx950 A8W4 MegaMoEV2，MERGED。
- [FlyDSL #972](https://github.com/ROCm/FlyDSL/pull/972)：A4W4 + FP8 blockwise P2P，OPEN、MERGEABLE、BLOCKED、REVIEW_REQUIRED；MI325/MI355 multi-GPU communication checks 当前失败。
- [AITER #4439](https://github.com/ROCm/aiter/pull/4439)：A8W4 MegaMoEV2 engine integration，MERGED。
- [AITER #4757](https://github.com/ROCm/aiter/pull/4757)：Mori HIP dispatch backend 已合入 `yanbo/mega_stage2_gfx1250` feature branch，不是 `main`。
- [AITER #4785](https://github.com/ROCm/aiter/pull/4785)：gfx1250 MegaMoE stage2 CI/feature，OPEN、CONFLICTING。
- [vLLM #53161](https://github.com/vllm-project/vllm/pull/53161)：gfx950 TP8/no-EP heterogeneous shared/routed expert 融合，OPEN draft、MERGEABLE、BLOCKED，依赖尚未合入的 AITER `#4891`。

AMD 当前的关键问题已经从“是否有 kernel”转为“vLLM/SGLang 的 opt-in engine 接入能否 rebase、补齐 CI 并形成可复现的默认选择条件”。

### Ascend

- [vllm-ascend #11701](https://github.com/vllm-project/vllm-ascend/pull/11701)：A3 CANN `mega_moe`，MERGED。
- [#11137](https://github.com/vllm-project/vllm-ascend/pull/11137)：旧 A5 MegaMoe 单 C kernel，OPEN、CONFLICTING。
- [#13655](https://github.com/vllm-project/vllm-ascend/pull/13655)：A5 Kimi-K3 W4A8 MXFP + SiTU，OPEN、CONFLICTING，目标为 `releases/v0.26.0rc`。
- [#13994](https://github.com/vllm-project/vllm-ascend/pull/13994)：MegaMoE EPLB，OPEN、MERGEABLE、BLOCKED、REVIEW_REQUIRED，目标为 `main`。
- [#14358](https://github.com/vllm-project/vllm-ascend/pull/14358)：MegaMoE prefill buffer sizing 修复，8 月 20 日合入 `releases/v0.26.0rc`。
- [#14449](https://github.com/vllm-project/vllm-ascend/pull/14449)：A5 fused backend，OPEN、MERGEABLE、`CLEAN`，但目标是 `rfc/vllm_cann` 而非 `main`。
- [#14495](https://github.com/vllm-project/vllm-ascend/pull/14495)：Kimi-K3 enablement，OPEN、CONFLICTING，目标为 `releases/v0.26.0rc`。
- [#14658](https://github.com/vllm-project/vllm-ascend/pull/14658)：面向 `main` 的 A5 MegaMoe 路径，OPEN、MERGEABLE、BLOCKED、REVIEW_REQUIRED，CI/pre-commit 当前失败。
- [#14664](https://github.com/vllm-project/vllm-ascend/pull/14664)：将 MegaMoE 设为唯一 FusedMC2 path 的重构，OPEN、MERGEABLE、BLOCKED，DCO/CI/pre-commit 当前失败。

这里必须按目标分支分别判断：release 分支已合入 bugfix，不等于新 backend 已进入 `main`；RFC 分支可合并，也不等于 production 路线已定。

### 其他

- [FastDeploy #7943](https://github.com/PaddlePaddle/FastDeploy/pull/7943) / [#8038](https://github.com/PaddlePaddle/FastDeploy/pull/8038)：WFP4A8 MegaMoE，MERGED。
- [cuDNN Frontend #448](https://github.com/NVIDIA/cudnn-frontend/pull/448)：统一 `MoeEp` Python API + CuTeDSL backend，OPEN draft、REVIEW_REQUIRED；8 月 10 日后无更新。

## 建议跟踪顺序

1. **deepseek-ai/DeepGEMM `#411` 的分布式正确性**：先修集中路由错误和不等 token 死锁，再讨论 H20 性能与 `#383` 的取舍；`CLEAN` 不能覆盖 review 中的算法/协议 blocker。
2. **sgl-project/DeepGEMM 的合入闭环**：先确认 `#68` 正确性修复和 `#74` 编译修复进入 `dev/release`，再看 `#69` 的 fused shared-expert 优化能否完成 review；`#53` 还需补正式 SGLang upstream 与 PD-disaggregation 验证。
3. **vLLM `#49636` 是否实际合入**：它已 APPROVED、MERGEABLE、`CLEAN`，下一步是确认 merge 时点以及主线消费的 FlashInfer 版本。
4. **DeepGEMM shared expert 的可信 baseline**：vLLM `#53040` 已合入，但 deepseek-ai/DeepGEMM `#409` 已撤回旧表；需等待 GB300 EP8 DeepEP/Kineto 对齐复测后再引用 kernel fusion gain。
5. **SGLang FlashInfer 路线是否收敛**：优先看 `#31470` 能否清理 CI/review，并决定是否吸收 `#33571` 的 int32 router ID + output view 优化。
6. **SM120/Rubin 依赖链**：`#4387 -> #4632` 需要先解决冲突和净 diff，`#4601` 需要等待可公开消费的 Rubin CuTe DSL 并补性能数据；同时跟踪 sgl-deep-gemm wheel 是否修复 `#75` 报告的 tvm-ffi tensor corruption。
7. **ROCm/Ascend engine 接入**：ROCm 重点看 AITER `#4891`、vLLM `#51918/#53161` 与 SGLang `#35619`；Ascend 重点看 `#14658/#14664` 是否把 RFC/release 分支上的 A5 路线收口到 `main`。

## 证据边界

- `MERGEABLE` 只说明当前 diff 没有文本冲突；`BLOCKED`、review、CI、目标分支策略和硬件验证仍可能阻止合入。
- `CLEAN` 也不覆盖 review 中发现的算法、同步协议或数值正确性问题；DeepGEMM `#411` 是本轮的直接例子。
- `APPROVED` 不等于已经进入 merge queue；`MERGED` 也不等于默认启用、正式 release 或所有硬件都验证通过。
- PR benchmark 与本地 bench 的 GPU、EP/TP/DP 拓扑、模型 geometry、token bucket、量化格式、计时边界和 baseline 不统一，不能直接横向排名。
- 对非 `main` 目标分支的 merged PR，本文明确标出 feature/release branch；不能仅根据 GitHub 的 `MERGED` 状态写成“已进入主线”。
- 本文混合两类证据：GitHub PR 状态回答“是否进入指定分支”，本地实验回答“在指定环境下表现如何”；两者不能相互替代。
