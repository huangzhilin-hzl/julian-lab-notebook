# Humming

This directory stores Humming benchmark data, reports, and replay scripts.

## Documents

| File | Content |
| --- | --- |
| [dsv4_flash_mxfp4a8_moe_20260513_092516.html](dsv4_flash_mxfp4a8_moe_20260513_092516.html) | Top-level HTML copy for quick viewing. |
| [bench_results/dsv4_flash_mxfp4a8_moe_20260513_092516/summary.html](bench_results/dsv4_flash_mxfp4a8_moe_20260513_092516/summary.html) | Self-contained HTML report for the H20 DSV4 Flash MXFP4A8 MoE microbench. |
| [bench_results/dsv4_flash_mxfp4a8_moe_20260513_092516/summary.md](bench_results/dsv4_flash_mxfp4a8_moe_20260513_092516/summary.md) | Markdown summary with per-leg and combined per-rank MoE GEMM tables. |
| [bench_results/dsv4_flash_mxfp4a8_moe_20260513_092516/dsv4_flash_mxfp4a8_moe.csv](bench_results/dsv4_flash_mxfp4a8_moe_20260513_092516/dsv4_flash_mxfp4a8_moe.csv) | Structured benchmark CSV for TP4/TP8 and w13/w2 cases. |
| [bench_results/dsv4_flash_mxfp4a8_moe_20260513_092516/raw/](bench_results/dsv4_flash_mxfp4a8_moe_20260513_092516/raw/) | Raw JSON outputs for TP4/TP8 x w13/w2. |
| [bench_results/dsv4_flash_mxfp4a8_moe_20260513_092516/run.log](bench_results/dsv4_flash_mxfp4a8_moe_20260513_092516/run.log) | Pod-side execution log. |
| [scripts/bench_dsv4_flash_mxfp4a8_moe.py](scripts/bench_dsv4_flash_mxfp4a8_moe.py) | Benchmark driver copied from open-huming. |

## Test Scope

- Workload: `dsv4-flash`
- Hardware: H20 pod workflow
- Benchmark type: Humming kernel microbench only, excluding serving startup, router latency, and all-to-all latency
- Quantization: FP8 activation x MXFP4 weight, with E8M0 weight scales
- TP sizes: `4`, `8`
- Legs: `w13`, `w2`
- Global decode tokens: `1,2,4,8,16,32,64,128,256`

## Headline Results

| Metric | Value |
| --- | --- |
| Best combined throughput | `29.45 TOPS` at TP4, 256 global decode tokens |
| Lowest combined latency | `0.0281 ms` at TP4, 1 global decode token |
| Best TP8 combined throughput | `19.64 TOPS` at 256 global decode tokens |
| Rows validated | `36` |

## Replay Notes

Use the original open-huming checkout on the H20 pod with:

```bash
export HUMMING_COMPILER=nvcc
export PYTHONPATH=$PWD
python benchmarks/bench_dsv4_flash_mxfp4a8_moe.py \
  --tp-sizes 4,8 \
  --legs w13,w2 \
  --global-tokens 1,2,4,8,16,32,64,128,256 \
  --output-dir benchmarks/results/h20/dsv4_flash_mxfp4a8_moe_<timestamp>
```
