# SGLang

This directory records SGLang optimization data, benchmark summaries, and related reproducibility notes.

## Topics

| Topic | Content |
| --- | --- |
| [GLM-5.3 / B300 / 128k prefill 跨引擎优化调查](glm53_b300_prefill128k_20260914/cross_engine_optimization_research.md) | vLLM、TensorRT-LLM、TokenSpeed、SGLang 源码、PR 和博客对照，含瓶颈时间预算、适用限制、实验顺序及 PR 来源记录（2026-09-14）。 |
| [dsv4](dsv4/) | DeepSeek-V4 optimization notes and benchmark data. |
| [deepseek_v4_roadmap_pr_summary_20260610.html](dsv4/deepseek_v4_roadmap_pr_summary_20260610.html) | DeepSeek V4 SGLang/vLLM roadmap items mapped to concrete PRs, issues, and tracker gaps. |
| [mimo_v25_sglang_pr_mapping.md](mimo_v25_sglang_pr_mapping.md) | MiMo-V2.5 inference optimization points mapped to related SGLang PRs and RFCs. |
| [qwen3_8_flash_next_sglang_pr_mapping_20260828.md](qwen3_8_flash_next_sglang_pr_mapping_20260828.md) | Qwen3.8-Flash-Next architecture, QSA, IndexShare MTP, HyperConnection, PLE offload, and follow-up PR mapping. |
| [minimax_h3_h200_optimization_pr_map_20260827.md](minimax_h3_h200_optimization_pr_map_20260827.md) | MiniMax-H3 on 8×H200 optimization layers, benchmark trade-offs, and related SGLang PR mapping. |
| [dspark_related_work_map_20260629.html](dspark_related_work_map_20260629.html) | DSpark Related Work reference map covering speculative decoding algorithms, system-aware scheduling, and parallel generation / NAT background. |
