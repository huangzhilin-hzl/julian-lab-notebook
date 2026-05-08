# SGLang DeepSeek-V4 Humming Quant

This directory stores notes for the SGLang PR24289 H20 DeepSeek-V4 Humming quantization test.

## Documents

| File | Content |
| --- | --- |
| [pr24289_h20_recipe_compare_summary.md](pr24289_h20_recipe_compare_summary.md) | Human-readable comparison across the four DeepSeek-V4 recipes: Low-Latency, Balanced, Max-Throughput, and Context-Parallel. |
| [pr24289_h20_recipe_compare_summary.csv](pr24289_h20_recipe_compare_summary.csv) | Structured benchmark table with startup, sanity-check accuracy, throughput, TTFT, and TPOT fields. |
| [pr24289_h20_test_commands.md](pr24289_h20_test_commands.md) | Command templates for server launch and benchmark replay. |

## Test Scope

- `fp8_baseline` uses the converted pure FP8 checkpoint.
- `marlin_w4a16`, `humming_w4a16`, and `humming_w4a8` use the official mixed FP4/FP8 checkpoint.
- The Marlin and Humming rows are the same-checkpoint comparison group.
- Context-Parallel uses the H20-adjusted DeepEP config with `num_sms=20`.
