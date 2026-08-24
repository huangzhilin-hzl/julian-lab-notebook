# MegaMoE 主流实现与接入进展（2026-08-24）

> 状态快照：2026-08-24，Asia/Shanghai。本文中的 `OPEN`、`MERGED`、`MERGEABLE`、`CONFLICTING`、`APPROVED`、`BLOCKED` 和 checks 状态以本次 GitHub 查询为准。

## 结论

截至 2026-08-24，相比 8 月 21 日快照有七项值得立即调整的判断：

1. **DeepGEMM `#409` 已补回可审计的 GB300 EP8 benchmark，但仍有 correctness blocker**：PR 已从 draft 转为普通 OPEN，并用相同 BF16 输入输出边界比较 FlashInfer modular、DeepGEMM unfused 和 fused shared-expert 三条路径；报告的 DeepGEMM fusion gain 为 9.78%–19.82%。但 review bot 指出 FP4 API 未校验 symmetric-buffer layout，默认 FP8 布局可能通过大小检查后静默读错偏移；当前仍无人工批准或公开 checks。
2. **DeepGEMM `#411` 完成 H20 性能更新，分布式正确性仍未闭环**：PR 现在给出 8×H20、`fast_math=1` 的 `#383`/DeepEP 对照，本地同环境 A/B 也验证了最新提交的主要目标区间约 0.9%–2.6% 提升；但集中路由错误、不等 rank token 死锁和与 `#383` 的公共 API 合并策略仍没有公开 re-review 结论。
3. **Kimi K3 出现新的 shared-expert 发布/尾部融合链**：DeepGEMM `#416` 将 BF16 shared expert 与 SM100 FP8/FP4 MegaMoE 一起调度并发布 TP shard，vLLM `#53556` 在 latent up-projection 尾部消费；后者报告多组 2.16%–8.61% 吞吐改善。但 `#416` review 已指出部分 shared-hidden/并行度组合会产生下溢偏移和远端越界写，因此整条依赖链当前不能合入。
4. **FlashInfer 新增 SM90 FP8 优化总线 `#4688`**：它引入 multi-CTA CGA、ping-pong scheduler、两级 FC2 store、heuristic/autotune 和 persistent knob cache。PR 当前 MERGEABLE 但 BLOCKED，2 项 checks 失败、1 项 pending；review 仍有 collective hang、partial-tile TMA store 和多 rank autotune 同步等风险。
5. **Serving 主线有实质合入，也有新的早期候选**：TRT-LLM `#17865` 已把 Kimi K3 NVFP4 CUTLASS/CuTeDSL MegaMoE SiTU 合入 `main`；SGLang `#35918` 已用 `--enable-w4a4-megamoe` 取代两项手工环境变量。vLLM `#53527` 开始接入 SGL fork 的 SM90 FP8 MegaMoE，但仍依赖尚未合入的 vLLM `#53503` 和外部 DeepGEMM fork，不能按主线可用评估。
6. **ROCm 的底层依赖已推进，框架消费仍待收口**：AITER `#4785`（gfx1250 stage2/combine overlap）和 `#4891`（gfx950 native-I384 FHMoE，覆盖 M≤2048）均已合入 `main`。这解除了 vLLM `#53161` 的核心 AITER 依赖，但 `#53161` 本身仍是 draft、BLOCKED 且有 pending check；SGLang `#35619` 也尚未 rebase/完成 CI。
7. **Ascend 已合入 EPLB 与 release hang 修复，但 A5 main 路线重新冲突**：`#13994` 已将 MegaMoE EPLB 合入 `main`，`#14747` 已在 `releases/v0.26.0rc` 修复主/草稿模型量化不一致时的 hang；与此同时 `#14658/#14664` 都变为 CONFLICTING。新 `#14784` 面向 `main` 接入 MiniMax M3 CANN MegaMoE，但尚无 NPU/service 验证。

当前主链可以简化为：

```text
DeepGEMM SM100:
  #304 -> #316 -> #364 -> #377 -> #396                  已合入
  #404 cache policy                                     开放、CLEAN
  #409 GLM NVFP4 shared / #416 Kimi K3 shared publish   开放、CLEAN、review 有 correctness blocker
DeepGEMM SM90:
  #323 / #360 / #383                                    开放、冲突
  #411 FP8 x MXFP4                                      开放、CLEAN、H20 benchmark 已补，正确性阻塞未闭环

sgl-project/DeepGEMM fork:
  SM90 FP8: #36 -> #48 -> #63                           已合入 dev
  SM90 FP4: #53 / correctness: #68 / perf: #69          开放、CLEAN
  nvcc13 compile fix: #74                               开放、CLEAN

FlashInfer:
  SM100 base: #3686 -> #3852 -> #3980 -> #4079 -> #4101 已合入
  serving:    #4341 zero-copy / #4531 zero-token fix     已合入
  precision:  #4386 BF16                                 已合入
  Hopper:     #4113 pull / #4069 push                    已合入
              #4589 NVFP4 W4A8 / #4688 FP8 perf          开放、review/CI 阻塞
  SM120:      #4387 MXFP8 -> #4632 W4A8                  开放、冲突
  SM107:      #4601 Rubin                                开放、冲突

Serving integrations:
  TensorRT-LLM: #17624 / #17865                          Kimi K3 已合入 main
  vLLM native:  #40860 / #43339 / #43632 / #52445 / #53040 已合入
  vLLM new:     #53527 SM90 / #53556 Kimi shared         开放、BLOCKED
  vLLM FlashInfer: #49636                                APPROVED、仍开放、有 pending check
  SGLang native: #23882 / #25052 / #29016 / #34883 / #35918 已合入
  SGLang fix:   #36007 MXFP8 scale stride                开放、BLOCKED
  SGLang FlashInfer: #31470 / #33571                     开放

Other hardware:
  ROCm: AITER #4785/#4891                                已合入
        FlyDSL #972, vLLM #51918/#53161, SGLang #35619   开放
  Ascend: #13994 EPLB / #14747 release hang fix          已合入
          #14658/#14664/#14784                           开放
```

## 相比 2026-08-21 的主要变化

| 路线 | PR | 2026-08-24 状态 | 本轮变化或新信息 |
| --- | --- | --- | --- |
| DeepGEMM GLM shared expert | [DeepGEMM#409](https://github.com/deepseek-ai/DeepGEMM/pull/409) | OPEN、非 draft、MERGEABLE、`CLEAN` | 已补齐对齐后的 8×GB300/EP8 benchmark，报告 fusion gain 9.78%–19.82%；但 symmetric-buffer layout 未校验可能静默错读，仍是 blocker。 |
| DeepGEMM SM90 MXFP4 | [DeepGEMM#411](https://github.com/deepseek-ai/DeepGEMM/pull/411) | OPEN、MERGEABLE、`CLEAN` | 新增 8×H20 `#383`/DeepEP 对照和最新优化；原有集中路由、不等 token 完成协议/API 合并问题尚无 re-review 结论。 |
| DeepGEMM Kimi shared publish | [DeepGEMM#416](https://github.com/deepseek-ai/DeepGEMM/pull/416) | OPEN、MERGEABLE、`CLEAN` | 新增 Kimi K3 BF16 shared-expert destination-scatter；review 指出 offset 下溢会造成远端越界写。 |
| FlashInfer SM90 perf | [flashinfer#4688](https://github.com/flashinfer-ai/flashinfer/pull/4688) | OPEN、MERGEABLE、BLOCKED、REVIEW_REQUIRED | 新增 multi-CTA/ping-pong/autotune/knob cache；2 项 checks 失败、1 项 pending，review 仍有 collective/store 正确性问题。 |
| TRT-LLM Kimi NVFP4 | [TensorRT-LLM#17865](https://github.com/NVIDIA/TensorRT-LLM/pull/17865) | MERGED，2026-08-21 | CUTLASS 与 CuTeDSL MegaMoE SiTU 已进入 `main`；DEP8 和 disaggregated aggregate 不在该 PR 初始验证范围。 |
| SGLang W4A4 config | [sglang#35918](https://github.com/sgl-project/sglang/pull/35918) | MERGED，2026-08-22 | 新增 `--enable-w4a4-megamoe`，统一设置 DeepGEMM FP4 activation/MXF4 开关；不改变 kernel。 |
| SGLang MXFP8 stride | [sglang#36007](https://github.com/sgl-project/sglang/pull/36007) | OPEN、MERGEABLE、BLOCKED、REVIEW_REQUIRED | 修复 `H % 512 != 0` 时 pre-dispatch scale row 的 padded stride；GB300 focused test 通过，但 aggregate CI 多项失败。 |
| vLLM SM90 | [vLLM#53527](https://github.com/vllm-project/vllm/pull/53527) | OPEN、MERGEABLE、BLOCKED、REVIEW_REQUIRED | 新增 H200/DeepSeek-V4 FP8 `deep_gemm_mega_moe` 候选；依赖仍 OPEN/BLOCKED 的 [vLLM#53503](https://github.com/vllm-project/vllm/pull/53503) 和外部 DeepGEMM fork，性能图尚不足以替代主线验证。 |
| Kimi shared tail | [vLLM#53556](https://github.com/vllm-project/vllm/pull/53556) | OPEN draft、MERGEABLE、BLOCKED | 消费 DeepGEMM `#416` 的 published shared fragments；报告 2.16%–8.61% 吞吐改善，但上游依赖存在越界写 blocker，当前 checks 也有失败。 |
| ROCm stage2 / FHMoE | [AITER#4785](https://github.com/ROCm/aiter/pull/4785) / [#4891](https://github.com/ROCm/aiter/pull/4891) | MERGED，2026-08-21 / 08-22 | gfx1250 stage2-combine overlap 与 gfx950 native-I384 FHMoE 已进入 `main`；后者只对连续配置覆盖的 M≤2048 启用。 |
| Ascend EPLB | [vllm-ascend#13994](https://github.com/vllm-project/vllm-ascend/pull/13994) | MERGED，2026-08-21 | MegaMoE 现已支持 EPLB 的 original INT8/NZ weight-list 路径并进入 `main`。 |
| Ascend A5 main | [#14658](https://github.com/vllm-project/vllm-ascend/pull/14658) / [#14664](https://github.com/vllm-project/vllm-ascend/pull/14664) | 全部 OPEN、CONFLICTING、REVIEW_REQUIRED | 两条原本可合并的 A5/FusedMC2 主线候选已重新冲突，且 checks 仍失败。 |
| Ascend MiniMax M3 | [#14784](https://github.com/vllm-project/vllm-ascend/pull/14784) | OPEN、MERGEABLE、BLOCKED、REVIEW_REQUIRED | 复用 CANN MegaMoE、传递 `swigluoai` 参数并对 BS>4096 分块；尚无 Ascend NPU/service 验证。 |

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
- [#409](https://github.com/deepseek-ai/DeepGEMM/pull/409)：OPEN、非 draft、MERGEABLE、`CLEAN`，目标分支为 `nv_dev`。它将 BF16 shared expert 融合进 SM100 NVFP4 MegaMoE，并在最终 add 前应用 routed scaling。作者已用统一 BF16 输入输出边界重做 8×GB300/EP8 benchmark：相对 DeepGEMM routed-only + serial shared，fused 路径报告 9.78%–19.82% kernel fusion gain；相对 FlashInfer modular baseline 报告 15.18%–34.17%。计时仍是选定 CUDA kernel 行之和，排除了 router/top-k、数据与权重处理、autotune 和 CPU gaps。更重要的是，review bot 指出 FP4 API 没有保存/校验 symmetric-buffer layout，默认 FP8 布局可能通过 byte-size 检查后静默错读；8 月 24 日提交只处理了 combine-slot invariant 等问题，当前公开讨论未显示该 critical blocker 已关闭。
- [#416](https://github.com/deepseek-ai/DeepGEMM/pull/416)：OPEN、MERGEABLE、`CLEAN`，目标分支为 `nv_dev`。它面向 Kimi K3，在现有 SM100 FP8/FP4 MegaMoE 中同时调度 BF16 shared expert 和 routed RMSNorm，并把 TP-sharded shared partial destination-scatter 到 symmetric workspace，供 vLLM `#53556` 的 latent up-projection tail 消费。PR 报告 producer 开销在噪声内、端到端吞吐 +2.97%–8.18%；但 review 已指出部分 shared-hidden/并行度组合的 offset 下溢会造成远端显存越界写，必须先修复。

### deepseek-ai/DeepGEMM：SM90 仍未形成统一 upstream

- [#323](https://github.com/deepseek-ai/DeepGEMM/pull/323)：SM90 FP8 fused path，OPEN、CONFLICTING；8 月 14 日有公开活动，但最后代码提交仍早于本轮。
- [#360](https://github.com/deepseek-ai/DeepGEMM/pull/360)：cooperative 单 kernel 方案，OPEN、CONFLICTING，8 月 3 日后无更新。
- [#383](https://github.com/deepseek-ai/DeepGEMM/pull/383)：destination-rank pull、L1/L2 两个大 kernel，OPEN、CONFLICTING，目标仍是 `nv_dev`。8 月 21 日仍有公开活动，但没有解决 upstream 冲突。
- [#411](https://github.com/deepseek-ai/DeepGEMM/pull/411)：FP8 activation × MXFP4 weight persistent kernel，OPEN、MERGEABLE、`CLEAN`，目标为 `nv_dev`。它将 SM100 ring-buffer/wave-interleaved scheduler 适配到 Hopper，并保持 routed weights 为 MXFP4、运行时解码为 FP8 WGMMA 输入。8 月 24 日加入 packed-FP16 WGMMA workload gate 等优化，并补充 8×H20、`fast_math=1`、cold-L2 的 `#383`/DeepEP 对照；PR 自身 TODO 仍包括更全面的数值验证和 framework integration。原 review 指出的集中路由错误、不等 rank token 完成协议死锁及与 `#383` 公共 API 冲突，尚无新的公开 re-review 证明已解决。

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

- [humming_mxfp4afp8_deepep_pr383_h20_3e_20260811.md](./sm90_humming_mxfp4afp8_megamoe/humming_mxfp4afp8_deepep_pr383_h20_3e_20260811.md) 提供了 H20/H20-3e 描述性比较。由于执行图和计时范围不同，这些结果不能当作可替换 backend 的端到端加速比。
- [humming_mxfp4afp8_megamoe_pr411_h20_fastmath1_20260824.md](./sm90_humming_mxfp4afp8_megamoe/humming_mxfp4afp8_megamoe_pr411_h20_fastmath1_20260824.md) 对 `#411` 最新优化提交做了 8×H20 同环境直接父子提交 A/B：22 个 shape 的 rank0/max-rank 几何平均提升 1.04%/1.68%，9 个主要目标 shape 为 1.60%/1.67%；该测试是 persistent kernel-only、shared expert off，不替代分布式路由正确性或 serving e2e 验证。

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
| [#4589](https://github.com/flashinfer-ai/flashinfer/pull/4589) | SM90 NVFP4 checkpoint；packed/folded/hot/dual residency policy | OPEN、REVIEW_REQUIRED；GitHub mergeability 当前为 UNKNOWN，1 项 check 失败、1 项 pending。 |
| [#4688](https://github.com/flashinfer-ai/flashinfer/pull/4688) | 现有 SM90 FP8 pull backend 的 multi-CTA/ping-pong/autotune 优化 | OPEN、MERGEABLE、BLOCKED、REVIEW_REQUIRED；2 项 checks 失败、1 项 pending。 |

`#4589` 的 folded 路径本质上是在 load time 将 NVFP4 权重转换为现有 FP8 push engine 可消费的格式；packed 路径才是 W4A8 in-kernel decode。两者的显存和性能权衡不同，不能只按同一个 “SM90 NVFP4 backend” 口径比较。

`#4688` 不是第三条独立 backend，而是对 `#4113` 所在 SM90 FP8 pull kernel tree 的大规模性能与调优改造：增加 CGA multicast、双 workgroup ping-pong、FC2 TMA-store pipeline、token-back modes、离线/在线 autotune 和 persistent knob cache。由于 review 仍指出多 rank autotune barrier 数不一致可能 hang、partial hidden tile 的 unpredicated TMA store 可能写错，以及 benchmark CSV 口径混用等问题，不能用 H200/EP4 报表替代合入条件。

### SM120 与 Rubin：功能进入评审，生产条件尚未满足

- [#4387](https://github.com/flashinfer-ai/flashinfer/pull/4387)：SM120/SM121 MXFP8 swap-AB fused backend，APPROVED 但仍 CONFLICTING；1 项 check 失败、1 项 pending。PR 明确写明仍在性能调优，并对已知的 in-kernel reduce 和 cluster 配置问题做拒绝式 guard。
- [#4632](https://github.com/flashinfer-ai/flashinfer/pull/4632)：stacked 在 `#4387` 上的 SM120 W4A8 backend。标题含 `[Draft]`，但 GitHub `isDraft=false`；仍 OPEN、CONFLICTING、REVIEW_REQUIRED，且尚未拆出依赖后的净 diff。
- [#4601](https://github.com/flashinfer-ai/flashinfer/pull/4601)：SM107 Rubin NVFP4/MXFP8 backends，OPEN、REVIEW_REQUIRED；GitHub mergeability 当前为 UNKNOWN，1 项 check 失败、1 项 pending。它依赖尚未公开发布的 Rubin CuTe DSL codegen，PR 也尚未给出性能数据。
- [#4604](https://github.com/flashinfer-ai/flashinfer/pull/4604)：MXFP8×BF16 integration，当前已不是 draft，但仍 OPEN、REVIEW_REQUIRED，mergeability 为 UNKNOWN，1 项 check 失败、1 项 pending；尚未形成可合入验证闭环。

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
- [#17865](https://github.com/NVIDIA/TensorRT-LLM/pull/17865)：Kimi K3 NVFP4 checkpoint 的 CUTLASS 与 MegaMoE CuTeDSL SiTU bring-up，8 月 21 日 MERGED 到 `main`；验证覆盖 DEP16，DEP8 和 disaggregated aggregate 不在初始范围。

8 月 13 日后的增量包括：

- [#17532](https://github.com/NVIDIA/TensorRT-LLM/pull/17532)：MERGED，将 MoE backend 选择改为可复现、可报告的 resolution contract。
- [#17907](https://github.com/NVIDIA/TensorRT-LLM/pull/17907)：MiniMax-M3 的 SwiGLUBias MegaMoE，已合入 `feat/m3_with_msa`，不是 `main`。
- [#17956](https://github.com/NVIDIA/TensorRT-LLM/pull/17956)：将 mixed-CGA、work-claiming 和 epilogue 优化移植到 TRT-LLM，OPEN draft，尚无性能表。
- [#18059](https://github.com/NVIDIA/TensorRT-LLM/pull/18059)：Kimi K3 EP8 的 expert-weight staging 与 mega-format transform 峰值显存修复，并移除不可追踪的 TP/EP 环境变量覆盖；OPEN、APPROVED，仍有 pending check，PR 明确未覆盖 EP8 served-model correctness。
- [#18067](https://github.com/NVIDIA/TensorRT-LLM/pull/18067)：将 `activation/situ_*` 显式参数同时传给 `MegaMoECuteDsl`；OPEN、REVIEW_REQUIRED，仍有 pending check。
- [#18068](https://github.com/NVIDIA/TensorRT-LLM/pull/18068)：解除并恢复 MegaMoE CuTeDSL 测试；OPEN、REVIEW_REQUIRED，仍有 pending check。

因此 Kimi-K3 DeepGEMM 与 NVFP4 CuTeDSL SiTU 基础路线都已进入 `main`，但 DEP8 内存、显式 SiTU 参数传递和测试恢复仍在后续 PR 中；MiniMax-M3 仍需区分 feature branch 和主线状态。

## 4. vLLM 与 SGLang

### vLLM

已合入的 native DeepGEMM 路径：

- [#40860](https://github.com/vllm-project/vllm/pull/40860)：DeepSeek-V4 和 `deep_gemm_mega_moe` 主接入；
- [#43339](https://github.com/vllm-project/vllm/pull/43339)：MegaMoE EPLB；
- [#43632](https://github.com/vllm-project/vllm/pull/43632)：input-prep kernel 迁移到 `nvidia/ops`；
- [#51146](https://github.com/vllm-project/vllm/pull/51146)：修正 Kimi-K3 MegaMoE path 的额外 add；
- [#52445](https://github.com/vllm-project/vllm/pull/52445)：修正 Kimi-K3 SiTU 参数名，避免首次 MegaMoE forward 直接报错；
- [#53040](https://github.com/vllm-project/vllm/pull/53040)：将 replicated FP8 shared expert 融入 DeepGEMM persistent MegaMoE，2026-08-21（北京时间）MERGED。

[vLLM #49636](https://github.com/vllm-project/vllm/pull/49636) 仍是 FlashInfer `moe_ep` 的关键入口。它增加 `flashinfer_moe_ep_mega_deep_gemm` 和 `flashinfer_moe_ep_mega_cutedsl`，保持 opt-in，并明确限制为 DeepSeek-V4。当前仍 APPROVED、OPEN，但 GitHub mergeability 为 UNKNOWN，且有 1 项 pending check；不能继续写成 `CLEAN` 或已经进入主线。

`#53040` 合入后，NVIDIA native shared-expert 融合已不再是候选。PR 作者报告 batch-size-1 输出吞吐约 +13.7%–15.0%，并发 64 workload 约 +9.4%；这些仍是 PR 指定的 B200/DeepSeek-V4 workload 数据。

仍开放的 native/EPLB 增量包括：

- [#52705](https://github.com/vllm-project/vllm/pull/52705)：相邻 draft，增加 GLM-5.2 DeepGEMM NVFP4 + fused shared expert；当前 MERGEABLE、BLOCKED，1 项 check 失败、1 项 pending。评审时应避免与 Kimi K3 的 `#53556` 混在一起。
- [#50647](https://github.com/vllm-project/vllm/pull/50647)：Kimi-K3 EPLB，OPEN draft、REVIEW_REQUIRED。8 月 24 日已补齐 main/MTP 的 physical/redundant expert metadata 和完整 GSM8K 对照；同步 EPLB 将 median rank balancedness 从 0.9356 提高到 0.9919，但 24 次 rearrangement 累计 CUDA 时间 933.8 秒，使测试吞吐降至 control 的约 31%。这说明模型注册已前进，但当前同步策略不适合直接作为性能结论。
- [#53022](https://github.com/vllm-project/vllm/pull/53022)：DeepSeek-V4/MegaMoE static expert maps，OPEN draft、CONFLICTING；支持 permutation 和 replicated physical slots，但尚未 rebase。
- [#53379](https://github.com/vllm-project/vllm/pull/53379)：修复 Kimi K3 streaming loader 遇到 `language_model/vision/language_model` 交错 prefix 时过早 finalize MegaMoE weights 的问题；OPEN、MERGEABLE、BLOCKED，4 项 checks 失败、1 项 pending，完整 S3 model-load canary 尚未执行。
- [#53527](https://github.com/vllm-project/vllm/pull/53527)：将 SGL fork 的 SM90 FP8 `deep_gemm_mega_moe` 路径接入 H200/DeepSeek-V4；OPEN、MERGEABLE、BLOCKED，尚无人工 review。它依赖仍 OPEN、MERGEABLE、BLOCKED 的 [#53503](https://github.com/vllm-project/vllm/pull/53503) 和外部 DeepGEMM fork/临时 wheel，PR 的 5%–60% 结论来自一次 sweep 图，仍需主线依赖、精度与可复现实验闭环。
- [#53556](https://github.com/vllm-project/vllm/pull/53556)：Kimi K3 published-shared latent tail，OPEN draft、MERGEABLE、BLOCKED，4 项 checks 失败、1 项 pending。它依赖 DeepGEMM `#416`；在该上游越界写 blocker 修复前，即使 PR 报告 2.16%–8.61% serving 吞吐改善也不能合入。

ROCm 消费方 [#51918](https://github.com/vllm-project/vllm/pull/51918) 是 opt-in FlyDSL MegaMoEV2 backend，当前 OPEN draft、CONFLICTING，7 项 checks 失败、1 项 pending。[#53161](https://github.com/vllm-project/vllm/pull/53161) 则是不同的 TP8/no-EP heterogeneous shared/routed expert 融合：它依赖的 AITER `#4891` 已于 8 月 22 日合入，但该 vLLM PR 本身仍 OPEN draft、MERGEABLE、BLOCKED，并有 pending check；两条路线仍不能合并评价。

### SGLang

已合入的 native DeepGEMM 路径继续扩展：

- [#23882](https://github.com/sgl-project/sglang/pull/23882)：DeepSeek-V4 主接入；
- [#25052](https://github.com/sgl-project/sglang/pull/25052)：W4A4 MegaMoE；
- [#29016](https://github.com/sgl-project/sglang/pull/29016)：SM90 FP8 MegaMoE；
- [#34883](https://github.com/sgl-project/sglang/pull/34883)：Kimi-K3 显式使用 SiTU activation，8 月 15 日 MERGED；
- [#35372](https://github.com/sgl-project/sglang/pull/35372)：扩展 `mega_moe_pre_dispatch` 的 wide-row 支持，8 月 20 日 MERGED；
- [#35918](https://github.com/sgl-project/sglang/pull/35918)：增加 `--enable-w4a4-megamoe`，统一设置 `DG_USE_FP4_ACTS/DG_USE_MXF4_KIND` 并弃用两项 SGLang 环境变量，8 月 22 日 MERGED。

这些 serving PR 背后的库侧实现主要来自 `sgl-project/DeepGEMM`，不是 `deepseek-ai/DeepGEMM` 的同编号 upstream 队列：SM90 FP8 对应 fork 的 `#36/#48/#63`，SiTU 显式选择对应 `#78`。当前 fork 侧 `#53` 的 SM90 FP4 SGLang 集成仍未 upstream，`#68/#69/#74` 也还没有 review/CI 闭环；详见上方独立小节。

FlashInfer 消费方仍未合入：

- [#31470](https://github.com/sgl-project/sglang/pull/31470)：OPEN、REVIEW_REQUIRED，GitHub mergeability 当前为 UNKNOWN，aggregate checks 仍有 26 项失败；8 月 21 日后没有新的公开代码活动。
- [#33571](https://github.com/sgl-project/sglang/pull/33571)：OPEN draft、REVIEW_REQUIRED，mergeability 当前为 UNKNOWN，17 项 checks 失败。它依赖的 `flashinfer#4341` 已合入，但自身最后公开活动仍是 8 月 5 日，尚未 rebase 或吸收到 `#31470`。

新的性能/精度候选尚处早期：

- [#35098](https://github.com/sgl-project/sglang/pull/35098)：direct TopK output 与 side-stream pre-dispatch overlap，OPEN、CONFLICTING、CHANGES_REQUESTED；PR 的 13.6%–19.9% 只针对前端 micro-chain，不是整机吞吐。
- [#35459](https://github.com/sgl-project/sglang/pull/35459)：MXFP8×BF16 MegaMoE，OPEN draft、REVIEW_REQUIRED，GitHub mergeability 为 UNKNOWN，16 项 checks 失败；描述、测试和 benchmark 尚未补齐。
- [#35619](https://github.com/sgl-project/sglang/pull/35619)：AITER/FlyDSL MegaMoEV2 for DeepSeek-V4，OPEN draft、REVIEW_REQUIRED，mergeability 为 UNKNOWN，16 项 checks 失败；PR 报告 8×MI355X 数据，但 rebase 到当前 `main` 后的 GPU e2e revalidation 仍待完成。
- [#36007](https://github.com/sgl-project/sglang/pull/36007)：修复 `mega_moe_pre_dispatch` 对 MXFP8 scale row 使用 packed stride 的错误；当 `H % 512 != 0` 时，DeepGEMM 的 16-byte TMA 对齐会产生 padding，旧地址计算从 token 1 起错位。当前 OPEN、MERGEABLE、BLOCKED、REVIEW_REQUIRED，GB300 focused regression 通过，但 aggregate CI 有 32 项失败。

## 5. AMD、Ascend 及其他硬件生态

### AMD

- [FlyDSL #876](https://github.com/ROCm/FlyDSL/pull/876)：MI355X/gfx950 A8W4 MegaMoEV2，MERGED。
- [FlyDSL #972](https://github.com/ROCm/FlyDSL/pull/972)：A4W4 + FP8 blockwise P2P，OPEN、REVIEW_REQUIRED，GitHub mergeability 当前为 UNKNOWN；8 月 24 日仍有活动，2 项 checks 失败、1 项 pending。
- [AITER #4439](https://github.com/ROCm/aiter/pull/4439)：A8W4 MegaMoEV2 engine integration，MERGED。
- [AITER #4757](https://github.com/ROCm/aiter/pull/4757)：Mori HIP dispatch backend 已合入 `yanbo/mega_stage2_gfx1250` feature branch，不是 `main`。
- [AITER #4785](https://github.com/ROCm/aiter/pull/4785)：gfx1250 stage2 GEMM2 + combine P2P overlap，8 月 21 日 MERGED 到 `main`；PR 报告 4×gfx1250 的 per-layer latency 降低 9.8%–15.4%。
- [AITER #4891](https://github.com/ROCm/aiter/pull/4891)：gfx950 DeepSeek-V4 native-I384 FHMoE config/AOT，8 月 22 日 MERGED 到 `main`；只在 CSV 连续覆盖的 M≤2048 启用，M>2048 fail closed 到 routed/shared 分离路径。
- [vLLM #53161](https://github.com/vllm-project/vllm/pull/53161)：gfx950 TP8/no-EP heterogeneous shared/routed expert 融合。底层 AITER `#4891` 已合入，但本 PR 仍 OPEN draft、MERGEABLE、BLOCKED，并有 pending check。

AMD 当前的关键问题已经从“是否有 kernel/配置”转为“vLLM/SGLang 的 opt-in engine 接入能否更新 AITER pin、rebase、补齐 CI 并形成可复现的默认选择条件”。

### Ascend

- [vllm-ascend #11701](https://github.com/vllm-project/vllm-ascend/pull/11701)：A3 CANN `mega_moe`，MERGED。
- [#11137](https://github.com/vllm-project/vllm-ascend/pull/11137)：旧 A5 MegaMoe 单 C kernel，OPEN、CONFLICTING。
- [#13655](https://github.com/vllm-project/vllm-ascend/pull/13655)：A5 Kimi-K3 W4A8 MXFP + SiTU，OPEN、CONFLICTING，目标为 `releases/v0.26.0rc`。
- [#13994](https://github.com/vllm-project/vllm-ascend/pull/13994)：MegaMoE EPLB，8 月 21 日 MERGED 到 `main`；在 `use_expert_weight_list` 下保留 MegaMoE 所需的原始 INT8/NZ tensor，并同步 scale/bias list。
- [#14358](https://github.com/vllm-project/vllm-ascend/pull/14358)：MegaMoE prefill buffer sizing 修复，8 月 20 日合入 `releases/v0.26.0rc`。
- [#14449](https://github.com/vllm-project/vllm-ascend/pull/14449)：A5 fused backend，OPEN、MERGEABLE、`CLEAN`，但目标是 `rfc/vllm_cann` 而非 `main`。
- [#14495](https://github.com/vllm-project/vllm-ascend/pull/14495)：Kimi-K3 enablement，OPEN、CONFLICTING，目标为 `releases/v0.26.0rc`。
- [#14658](https://github.com/vllm-project/vllm-ascend/pull/14658)：面向 `main` 的 A5 MegaMoe 路径，8 月 24 日仍有活动，但已变为 OPEN、CONFLICTING、REVIEW_REQUIRED，1 项 check 失败。PR 的 A3 GLM-5.2 128K/2K 数据显示 MegaMoE 吞吐约低 1.4%，属于“基本持平”而非加速。
- [#14664](https://github.com/vllm-project/vllm-ascend/pull/14664)：将 MegaMoE 设为唯一 FusedMC2 path 的重构，OPEN、CONFLICTING、REVIEW_REQUIRED，5 项 checks 失败、2 项 pending。
- [#14730](https://github.com/vllm-project/vllm-ascend/pull/14730)：统一使用 `additional_config.mega_moe_max_tokens` 配置 CANN MegaMoE symmetric-buffer receive/output capacity，目标为 `releases/v0.26.0rc`；当前 OPEN、MERGEABLE、`UNSTABLE`，checks 已通过但尚未合入。
- [#14747](https://github.com/vllm-project/vllm-ascend/pull/14747)：主模型与 MTP draft 量化格式不一致时回退小算子，避免 MegaMoE tiling hang；8 月 22 日 MERGED 到 `releases/v0.26.0rc`，不是 `main` 的最终方案。
- [#14771](https://github.com/vllm-project/vllm-ascend/pull/14771)：为 Qwen3.5 MTP draft 禁用 CANN MegaMoE，避免 target/draft 共用 symmetric-buffer collective 导致首请求 timeout；OPEN、CONFLICTING，尚无 A3 NPU 验证。
- [#14780](https://github.com/vllm-project/vllm-ascend/pull/14780)：另一条 A5 MegaMoE draft，目标为 `releases/v0.26.0rc`，当前 OPEN draft、CONFLICTING，checks 有失败/pending；需与 `#13655/#14658` 收敛而不是并行长期维护。
- [#14784](https://github.com/vllm-project/vllm-ascend/pull/14784)：面向 `main` 的 MiniMax M3 CANN MegaMoE，传递 `swigluoai` activation 参数并对 BS>4096 分块；OPEN、MERGEABLE、BLOCKED、REVIEW_REQUIRED，2 项 checks pending，尚无 NPU inference/service 验证。

这里必须按目标分支分别判断：`#13994` 已真正进入 `main`；`#14358/#14747` 只是 release 分支修复；`#14449` 仍在 RFC；`#14658/#14664/#14784` 才是当前 main 方向，但尚未形成可合入闭环。

### 其他

- [FastDeploy #7943](https://github.com/PaddlePaddle/FastDeploy/pull/7943) / [#8038](https://github.com/PaddlePaddle/FastDeploy/pull/8038)：WFP4A8 MegaMoE，MERGED。
- [cuDNN Frontend #448](https://github.com/NVIDIA/cudnn-frontend/pull/448)：统一 `MoeEp` Python API + CuTeDSL backend，OPEN draft、REVIEW_REQUIRED；8 月 10 日后无更新。

## 建议跟踪顺序

1. **DeepGEMM shared-expert correctness**：先关闭 `#409` 的 symmetric-buffer layout 静默错读和 `#416` 的 destination-scatter 越界写；两条 PR 的新 benchmark 都不能覆盖这些 blocker。
2. **DeepGEMM `#411` 的分布式协议与合并策略**：H20 性能证据已经足够讨论优化幅度，下一步必须是集中路由、不等 token 完成协议和 `#383` 公共 API 冲突的复审闭环。
3. **新 serving 依赖链是否可消费**：vLLM `#53556` 必须等待 DeepGEMM `#416` 修复；`#53527` 必须摆脱外部 fork/临时 wheel，并补主线精度与 CI；TRT-LLM `#18059` 还需 EP8 served-model correctness。
4. **FlashInfer Hopper 是否能清理 correctness/CI**：`#4688` 先处理 collective/TMA-store/autotune review 问题，`#4589` 继续收敛 W4A8 policy 与 checks；vLLM `#49636` 和 SGLang `#31470` 再基于可发布版本完成 framework 接入。
5. **ROCm 消费方更新 AITER pin**：`#4785/#4891` 已合入，重点转为 vLLM `#53161`、SGLang `#35619` 和 FlyDSL `#972` 的 rebase、CI 与 M>2048 fallback 验证。
6. **SGLang correctness 与 fork 合入闭环**：优先处理 `#36007` 的 padded scale stride；同时确认 sgl-project/DeepGEMM `#68/#74` 进入 release、`#69` 完成 review、`#53` 补正式 upstream 与 PD-disaggregation 验证。
7. **Ascend main 路线收敛**：`#13994` 已合入，但 `#14658/#14664` 重新冲突；需要决定 A5 主实现、吸收 release/RFC 修复，并为 `#14784` 补真实 NPU/service 验证。
8. **SM120/Rubin 依赖链**：`#4387 -> #4632` 需要先解决冲突和净 diff，`#4601` 需要等待可公开消费的 Rubin CuTe DSL 并补性能数据；同时跟踪 sgl-deep-gemm wheel 是否修复 `#75` 报告的 tvm-ffi tensor corruption。

## 证据边界

- `MERGEABLE` 只说明当前 diff 没有文本冲突；`BLOCKED`、review、CI、目标分支策略和硬件验证仍可能阻止合入。
- `CLEAN` 也不覆盖 review 中发现的布局、越界写、算法或同步协议问题；DeepGEMM `#409/#411/#416` 是本轮的直接例子。
- benchmark、focused test 或 aggregate CI 任一单项通过都不能替代另外两项；FlashInfer `#4688` 与 SGLang `#36007` 分别展示了性能证据存在但 review/CI 尚未闭环的情况。
- `APPROVED` 不等于已经进入 merge queue；`MERGED` 也不等于默认启用、正式 release 或所有硬件都验证通过。
- PR benchmark 与本地 bench 的 GPU、EP/TP/DP 拓扑、模型 geometry、token bucket、量化格式、计时边界和 baseline 不统一，不能直接横向排名。
- 对非 `main` 目标分支的 merged PR，本文明确标出 feature/release branch；不能仅根据 GitHub 的 `MERGED` 状态写成“已进入主线”。
- 本文混合两类证据：GitHub PR 状态回答“是否进入指定分支”，本地实验回答“在指定环境下表现如何”；两者不能相互替代。
