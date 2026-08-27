# FlashInfer GDN BF16 MTP 相关 PR 进展

更新日期：2026-08-27

## 1. 当前状态

FlashInfer 已合入 BF16-state GDN MTP kernel 和面向 speculative verify 的 BF16 output-only kernel。SGLang 也已合入 verify backend 分发修复，允许显式选择 Triton verify，避免 BF16 state 进入旧的 FP32-only FlashInfer MTP 路径。

当前尚未闭环的是：SGLang 的 FlashInfer verify backend 根据 state dtype 自动选择 FlashInfer BF16 MTP/verify kernel。仅升级 FlashInfer 版本不能保证上层 dispatcher 自动进入 BF16 kernel。

## 2. 已合入 PR

| 日期 | 仓库 | PR | 状态 | 主要进展 |
| --- | --- | --- | --- | --- |
| 2026-04-02 | FlashInfer | [#2679 feat(gdn): add BF16 state kernel with MTP support](https://github.com/flashinfer-ai/flashinfer/pull/2679) | 已合入 | 增加 BF16-state GDN decode/MTP kernel；state 在显存中使用 BF16，计算时提升到 FP32 |
| 2026-04-14 | FlashInfer | [#3042 bump version to 0.6.8](https://github.com/flashinfer-ai/flashinfer/pull/3042) | 已合入 | 将包含 PR #2679 的代码纳入 `0.6.8` 版本线 |
| 2026-07-10 | FlashInfer | [#3720 feat(gdn): add output-only BF16-state WY MTP decode kernel](https://github.com/flashinfer-ai/flashinfer/pull/3720) | 已合入 | 增加面向 MTP speculative verify 的 BF16-state output-only WY kernel |
| 2026-08-14 | SGLang | [#34592 GDN: honor configured linear-attn verify backend](https://github.com/sgl-project/sglang/pull/34592) | 已合入 | 修复 `--linear-attn-verify-backend` 被 dispatcher 忽略的问题；显式配置的 Triton verify 现在可以覆盖自动选择 |

## 3. 仍在推进的 PR

| 仓库 | PR | 当前状态 | 方向 |
| --- | --- | --- | --- |
| FlashInfer | [#2706 feat(gdn): add unified decode API and deprecation shims](https://github.com/flashinfer-ai/flashinfer/pull/2706) | Open | 统一 FP32/BF16 decode 与 MTP API，减少调用方直接绑定旧 kernel 的风险 |
| FlashInfer | [#3118 perf(gdn): fix BF16-state T=1 overhead and add pool/padding](https://github.com/flashinfer-ai/flashinfer/pull/3118) | Open | 优化 BF16-state 单 token 调用，并补充 pool 和 padding 能力 |
| FlashInfer | [#3127 optimize GDN decode BF16-state kernel for MTP with caching](https://github.com/flashinfer-ai/flashinfer/pull/3127) | Open | 优化带 intermediate caching 的 BF16-state MTP decode kernel |

## 4. PR 之间的关系

```text
FlashInfer #2679
  BF16-state GDN MTP kernel
        |
        +-- FlashInfer #3042
        |     进入 0.6.8 版本线
        |
        +-- FlashInfer #3720
              增加 BF16 output-only speculative verify kernel

FlashInfer #2706
  统一 decode/MTP API，仍在推进

SGLang #34592
  允许显式选择 verify backend
  当前可用兼容路径：BF16 state + Triton verify
```

## 5. 尚缺的集成进展

目前没有记录到一个已合入的 SGLang PR，能够在 FlashInfer verify backend 内根据 `initial_state.dtype` 自动完成以下分发：

```text
FP32 state -> legacy FlashInfer gated_delta_rule_mtp
BF16 state -> FlashInfer BF16-state MTP/output-only verify kernel
```

因此后续重点跟踪：

1. FlashInfer #2706 是否完成统一 API 并被正式版本采用。
2. SGLang 是否出现接入 FlashInfer BF16 verify kernel 的 PR。
3. SGLang 自动 verify dispatcher 是否能够按 state dtype 选择 kernel。
4. 相关集成是否覆盖 CUDA Graph、padding、cache hit 和不同 MTP token 数。

## 6. 简要结论

- BF16-state MTP kernel：已合入 FlashInfer #2679，并进入 `0.6.8` 版本线。
- BF16 output-only verify kernel：已合入 FlashInfer #3720。
- SGLang verify backend 显式覆盖：已合入 SGLang #34592。
- FlashInfer BF16 verify 的自动端到端分发：尚未发现已合入的 SGLang 集成 PR。
