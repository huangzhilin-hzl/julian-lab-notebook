# FlashInfer × MegaMoE PR 汇总（2026-07-23）

> 状态快照：2026-07-23，Asia/Shanghai。PR 状态、review 和 GitHub checks 会继续变化；性能数字来自各 PR 作者给出的测试环境，不应直接跨硬件或跨 workload 比较。

## 结论

FlashInfer 的 MegaMoE 工作可以分成三部分：

1. **Blackwell 主线已经具备内核与统一框架**：`#3852` 完成首次集成，`#3980` 完成 CuTeDSL kernel 重组和性能调优。
2. **面向推理框架的生产化工作尚未合入**：`#4079` 补齐 CUDA Graph、预量化权重、workspace pool 和离线调优缓存；`#4101` 叠加在其上，解决 CuTeDSL 4.5.2 的性能回退。
3. **Hopper 与下游框架接入仍在推进**：FlashInfer `#4069` 提供 SM90 push-FP8 路线，但当前有合并冲突；SGLang `#31470` 仍是 draft，现有 serving benchmark 也没有证明它适合作为低并发默认后端。

当前依赖关系可以简化为：

```text
#3686 MoE-EP 基础
    ↓
#3852 MegaMoE 首次集成
    ↓
#3980 CuTeDSL kernel 重组与调优
    ↓
#4079 serving-ready 框架能力
    ↓
#4101 CuTeDSL 4.5.2 性能 WAR

#4069 SM90 push-FP8 backend       （并行的 Hopper 路线）
#31408 → #31470 SGLang integration（下游消费方）
```

## FlashInfer 主线 PR

| PR | 状态 | 定位 | 关键内容 |
| --- | --- | --- | --- |
| [#3686](https://github.com/flashinfer-ai/flashinfer/pull/3686) | 已合入，2026-06-30 | MoE-EP 基础 | 建立 `MoEEpLayer`，支持 NCCL-EP / NIXL-EP 的 dispatch → grouped GEMM → combine；本身主要是 split path。 |
| [#3852](https://github.com/flashinfer-ai/flashinfer/pull/3852) | 已合入，2026-07-14 | MegaMoE 正式集成 | 将 split/mega 统一到 `MoEEpLayer`，加入 `deep_gemm_mega`、CuTeDSL NVFP4、CuTeDSL MXFP8 三个 SM100+ backend。 |
| [#3980](https://github.com/flashinfer-ai/flashinfer/pull/3980) | 已合入，2026-07-16 | Kernel 重组与调优 | 建立 `kernel_src/cutedsl_megamoe` + `shim` 边界，加入 in-kernel FC2 reduce、量化 combine、默认调优 profile 和在线 autotune。 |
| [#4079](https://github.com/flashinfer-ai/flashinfer/pull/4079) | 开放，可合并但被 gate 阻塞 | Serving-ready | CUDA Graph、fused quant+stage、预量化权重、跨层 workspace pool、持久化 knob cache 和离线 tuner。 |
| [#4101](https://github.com/flashinfer-ai/flashinfer/pull/4101) | 开放，依赖 `#4079` | 运行时兼容与性能 | 对 CuTeDSL 4.5.2 的 swap-AB FC12 mainloop 做版本限定 WAR，将支持口径从“4.6.1 性能下限”降为“4.5.2+ 全性能”。 |

### `#3852`：真正的 MegaMoE 起点

`#3686` 先建立可插拔的 MoE-EP 框架，`#3852` 才把 whole-layer MegaMoE 放入这套抽象中。它引入：

- `DeepGemmMegaMoeConfig`：DeepGEMM FP8/FP4；
- `Nvfp4CutedslMegaMoeConfig`：CuTeDSL NVFP4；
- `Mxfp8CutedslMegaMoeConfig`：CuTeDSL MXFP8；
- `MegaConfig → MoEEpMegaLayer` 的统一入口；
- 权重预处理、输入 staging、对称内存 workspace 和多卡正确性测试。

因此，“FlashInfer 已支持 MegaMoE”如果指低层集成，落点是 `#3852`；如果指推理框架可以直接稳定消费，则还需要后续 `#4079`。

### `#3980`：从首次集成变成可维护、可调优的 kernel drop

该 PR 将 CuTeDSL 源码迁移到规范的 `kernel_src/cutedsl_megamoe/src`，FlashInfer backend 只能通过 `shim` 访问 kernel，实现上游 kernel drop 与框架适配层解耦。

作者在 4×GB200、EP=4、DeepSeek-V3-like geometry 下报告：

- 8192 token/rank 时，`nvfp4 + combine_nvfp4` 为 1644.0 µs；
- `deep_gemm_mega` 为 3105.2 µs；
- 对应约 **1.89×** speedup。

这组数字只说明该指定 GB200 workload 下的 kernel 潜力。PR 同时记录 CuTeDSL 4.5.2 生成代码明显变慢的问题，后续由 `#4101` 处理。

### `#4079`：推理引擎接入所需的生命周期能力

该 PR 不只是继续调 kernel，而是补齐 serving engine 的使用契约：

- warmup 后支持 CUDA Graph capture/replay；
- fused quantization + input staging，减少热路径 launch；
- `Unquantized | Prequantized` 权重包；
- preprocess 后释放源权重，降低加载期 OOM 风险；
- 多个 MoE layer 共享对称内存 workspace；
- 离线生成、运行时纯 lookup 的 knob cache，避免引擎进程内 autotune。

PR 给出的 4×GB200 数据：

- Microbenchmark：大 token 场景相对 `deep_gemm_mega` 约 1.6–1.9×；
- vLLM 0.25.1、DeepSeek-V4-Flash：prefill 约 **+18%**，decode 约 **+7%**。

当前 GitHub 显示该 PR 可合并，但需要 review，`Test Results Summary` 为红色。本次汇总没有继续追踪内部 leaf job，因此不能仅凭 aggregate check 判断为代码逻辑失败。

### `#4101`：消除 CuTeDSL 4.5.2 性能断层

vLLM 0.25.1 固定 CuTeDSL 4.5.2，而 `#3980` 发现该版本对 NVFP4 swap-AB FC12 kernel 的代码生成慢 34–54%。`#4101` 对 4.5.2 单独启用 mainloop peel WAR：

- 4.5.2 + WAR：1024/2048/8192 token 下为 428.6/621.5/1896.4 µs；
- 4.6.1 参考：428.5/625.6/1923.5 µs；
- 4.5.3+ 不走 WAR，生成路径保持不变。

该 PR 明确要求 `#4079` 先合入。当前 41 个 changed files 中大部分来自叠加的 `#4079`，不能把它当作独立的单文件修复评审。

## Hopper 并行路线

### [`#4069`](https://github.com/flashinfer-ai/flashinfer/pull/4069)：SM90 push-FP8 MegaMoE

该 PR 面向单节点 NVLink Hopper，将量化 payload 直接写入 peer-mapped symmetric memory，并融合：

```text
dispatch
→ FP8 block-scale FC1
→ SwiGLU + activation quantization
→ FC2
→ grouped combine
```

主要优化包括 deduplicated dispatch、owner-side grouped combine 和可选 fused FC1 epilogue。作者在 H800 上报告，大吞吐场景相对 NCCL all-to-all + `cutlass_fused_moe` 可达到约 2× 以上，但 DSV3 decode 小 batch 的提升只有约 1.06–1.10×。

当前状态：

- 非 draft；
- `REVIEW_REQUIRED`；
- 与 `main` 冲突，GitHub 标记 `CONFLICTING / DIRTY`；
- `Test Results Summary` 为红色，尚未在本文中追踪内部 leaf failure。

因此它当前最明确的合入前动作是先 rebase/解决冲突，再重新判断 CI 根因。

## SGLang 消费方

| PR | 状态 | 说明 |
| --- | --- | --- |
| [SGLang #31408](https://github.com/sgl-project/sglang/pull/31408) | 已关闭、未合入 | 首版 FlashInfer NVFP4/MXFP8 MegaMoE 接入。 |
| [SGLang #31470](https://github.com/sgl-project/sglang/pull/31470) | 开放 draft | 接替 `#31408`，加入 `flashinfer_megamoe` runner/A2A backend，直接包装 `flashinfer.moe_ep`。 |

`#31470` 使用同一个 fused backend 同时承担 expert compute 和 EP communication，因此不能再叠加外部 dispatcher/combine。它覆盖 DeepGEMM FP4、CuTeDSL NVFP4/MXFP8 的权重准备与 FlashInfer `MoEEpMegaLayer` 生命周期。

PR 中 DeepSeek-V4-Flash NVFP4 serving benchmark 相对 `flashinfer_trtllm_routed + FlashInfer A2A`：

| 最大并发 | 输出吞吐变化 | Mean TPOT 变化 | 判断 |
| ---: | ---: | ---: | --- |
| 32 | -23.1% | +12.4% | baseline 明显更好 |
| 128 | -10.1% | -2.8% | MegaMoE TPOT 略好，但吞吐较差 |
| 1024 | +0.7% | -6.6% | 吞吐基本持平，TPOT 略好 |

这组对比排除了存在已知问题的 in-kernel FC2-reduce 变体。现有结果说明当前接入在低/中并发没有形成默认后端优势，高并发时主要是持平。PR 仍为 draft，当前还存在直接的 `lint` 失败；其余多个 `pr-gate`/finish 红灯需要在 lint 修复后重新判断。

## 相邻但不纳入主线

- [FlashInfer #3424](https://github.com/flashinfer-ai/flashinfer/pull/3424)：`monomoe` 小 batch megakernel，面向 SM90a、TopK=8、H=2048、I=512、BS≤8；不是统一 `moe_ep` MegaMoE 路线。
- [FlashInfer #4075](https://github.com/flashinfer-ai/flashinfer/pull/4075)：补齐 split path 的 NIXL transport 契约。
- [FlashInfer #4098](https://github.com/flashinfer-ai/flashinfer/pull/4098)：普通 MXFP8 CUTLASS GEMM 测试修复，搜索会命中 MegaMoE 关键词，但不修改 MegaMoE 主链。
- [vLLM #47939](https://github.com/vllm-project/vllm/pull/47939)：直接调用 DeepGEMM MegaMoE，不是 FlashInfer consumer。

## 建议跟踪顺序

1. 先看 `#4079` 是否完成 review，以及红色 `Test Results Summary` 对应的真实 leaf job。
2. `#4079` 合入后重看 `#4101` 的净 diff，并验证 CuTeDSL 4.5.2 支持口径。
3. 对 `#4069` 先解决与 `main` 的冲突，再重新跑 SM90 CI/硬件验证。
4. 对 SGLang `#31470` 先修 lint 和集成完整性，再判断低并发性能差距能否通过 graph/调度或 kernel knob 解决。
5. 不把 `#3424` 的单卡小 batch `monomoe` 性能结论外推到 whole-layer expert-parallel MegaMoE。

## 入口

- FlashInfer MegaMoE umbrella issue：[flashinfer-ai/flashinfer#3692](https://github.com/flashinfer-ai/flashinfer/issues/3692)
- MegaMoE integration API RFC：[flashinfer-ai/flashinfer#3704](https://github.com/flashinfer-ai/flashinfer/issues/3704)
- SM90 tracking issue：[flashinfer-ai/flashinfer#3780](https://github.com/flashinfer-ai/flashinfer/issues/3780)
- SM100 tracking issue：[flashinfer-ai/flashinfer#3781](https://github.com/flashinfer-ai/flashinfer/issues/3781)
