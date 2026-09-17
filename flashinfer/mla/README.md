# MLA 实现与优化资料

覆盖 FlashInfer、TensorRT-LLM、CUTLASS / CuTe DSL、TileLang、cuTile / TileGym、FlashMLA、FA4 和 TokenSpeed 的实现与框架选型。

| 日期 | 资料 | 内容 |
| --- | --- | --- |
| 2026-09-17 | [Blackwell MLA 后端选型与 TokenSpeed](blackwell-mla-backend-selection-and-tokenspeed-20260917.md) | SGLang / vLLM / TokenSpeed 的 dense/sparse 分派、wrapper 与 TRTLLM API 的区别、TRTLLM-gen cubin 边界、CuTe DSL 限制、FlashMLA padding / KV 格式，以及 55 处固定 commit 源码位置 |
| 2026-09-14 | [MLA 实现相关 PR 地图](mla-implementation-pr-map-20260914.md) | 73 个重点 PR 链接、合并状态、dense/sparse 与 prefill/decode 区分、实现差异和源码入口 |

对应证据：[后端选型源码索引](evidence/20260917/backend-source-index.json)、[93 项候选 PR 元数据](evidence/20260914/pr-index.json)、[源码文件历史](evidence/20260914/file-histories.json)。
