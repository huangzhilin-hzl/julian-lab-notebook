# infer-tasks

本目录存放 Harbor benchmark 任务，每个任务单独一个子目录。新增任务请新建子目录，不要改已有任务的文件。格式与校验说明见 [HARBOR_NOTES.md](HARBOR_NOTES.md)。

- [`sglang-bench-template`](sglang-bench-template/)：最小模板，按给定模型与启动参数拉起 sglang 并跑一轮 `bench_serving`，只检查结果 JSON 完整，不设性能阈值。
- [`dsv41-flash-h200-fp4-lowlatency`](dsv41-flash-h200-fp4-lowlatency/)：sglang cookbook 的 DeepSeek-V4.1 Flash（H200 · FP4 · Low-Latency · 单机）Harbor 任务。
