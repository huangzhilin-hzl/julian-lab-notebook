# Profile

GPU kernel 性能分析与可视化的实践笔记。

## Topics

| Topic | Content |
| --- | --- |
| [cuda_pipeline_debug.md](cuda_pipeline_debug.md) | 判断 CUDA kernel pipeline 是否充分使用的工具、指标、工作流；含 Nsight Compute / Nsight Systems / SASS / 自埋点；针对 Warp Specialization (Hopper WGMMA + TMA) kernel 的特殊关注点。 |
| [examples/humming_walkthrough.md](examples/humming_walkthrough.md) | 端到端手把手 walkthrough：用 `open-huming` 的 `bench_humming.py` 实操跑一遍 nsys + ncu，分 7 个 Step 看懂 `humming_ws` GEMM 的 pipeline 利用情况，每步带 Checkpoint。 |
| [examples/humming_m64_m4096_diagnosis.html](examples/humming_m64_m4096_diagnosis.html) | **完整一次实战记录**（独立 HTML 单文件）：在 H20 上对 humming MXFP4×FP8 GEMM 做 M=64 vs M=4096 两组 profile，列出所有 ncu 命令、原始数值、stall 分布对照表、判定结论；附 ncu 使用踩坑笔记。 |
