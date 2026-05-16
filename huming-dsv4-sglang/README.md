# Humming / MXFP4A16 DeepSeek-V4 Flash benchmark notebook

This folder contains a sanitized snapshot of the CP8/TP8, No-DeepEP, disable-radix-cache benchmark artifacts used to compare MXFP4A16 MoE backends on DeepSeek-V4 Flash.

## Contents

| Path | Purpose |
| --- | --- |
| `scripts/run_dsv4_flash_mxfp4_backends.py` | Benchmark driver with private paths and served-model names replaced by placeholders. |
| `patches/` | Temporary FlashInfer MXFP4A16 SM90 Cutlass patch helper and copied quantization method used for the FlashInfer run. |
| `data/mxfp4_best_runs_combined.csv` | Combined best-run table for Marlin, Humming, and FlashInfer MXFP4A16. |
| `data/mxfp4_backend_comparison.md` | Markdown summary tables for TTFT, TPOT, and output throughput. |
| `data/marlin_humming/` | Sanitized aggregate outputs for Marlin MXFP4A16 and Humming MXFP4A16. |
| `data/flashinfer/` | Sanitized aggregate outputs for FlashInfer MXFP4A16. |

## Scope

- Model: `deepseek-ai/DeepSeek-V4-Flash`
- Baseline model in comparison tables: `sgl-project/DeepSeek-V4-Flash-FP8`
- Hardware口径: single-node 8x H20
- Parallelism: TP8 + CP8
- DeepEP: disabled
- Radix cache: disabled by server argument
- TTFT: input 16K / 32K / 64K / 128K, output 256, BS=1, 3 rounds
- TPOT: input 1K / 32K / 64K / 128K, output 1024, BS=1 / 4 / 8, 3 rounds

## Redaction policy

The snapshot removes private machine identifiers, runtime container names, workspace paths, personal local paths, internal build URLs, and raw logs. The retained data is limited to aggregate metrics, sanitized runtime metadata, and benchmark scripts needed for reproducibility review.
