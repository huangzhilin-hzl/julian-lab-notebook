# mHC 状态证据：2026-09-11

- `snapshot.json`：266 个 PR 的逐项 REST 状态、合入时间、目标分支、head SHA、作者正文；重点候选附 review/check 快照。`queries` 记录检索语句、结果数量与覆盖上限；`sources` 记录固定源码 commit。
- `*_source_index.json`：本次读取的仓库/分支版本与 mHC 相关目录路径。
- `source/`：主要 kernel 和 dispatch 源码的固定版本副本。对应上游仓库及 commit 由 source index 确定。
- `issues/`：直接关联的 runtime zero-JIT tracker。

时间窗口为 2026-09-11 09:38–09:44 UTC（17:38–17:44 Asia/Shanghai）。这是查询期间形成的状态快照，GitHub 后续状态可能继续变化。PR 的性能与测试信息来自作者报告，本轮未执行 GPU 测试。
