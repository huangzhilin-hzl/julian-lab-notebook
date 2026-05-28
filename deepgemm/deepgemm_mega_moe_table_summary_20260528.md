# DeepGEMM Mega MoE Table Summary

Run date: 2026-05-28

## Command

```bash
python3 -m modal run deepgemm/modal_deepgemm_mega_moe.py \
  --task table \
  --model both \
  --gpu B200:8 \
  --deepgemm-ref main \
  --batch-sizes 1,512,8192,32768
```

## Environment

- Modal run: https://modal.com/apps/huangzhilin-hzl/main/ap-Udpo9b5tiZusC9H5jQ7Fgx
- Modal result dir: `/cache/deepgemm/results/20260528-044053`
- DeepGEMM commit: `714dd1a4a980`
- DeepEP commit: `not-installed`
- GPU: `8 x NVIDIA B200`
- PyTorch: `2.12.0a0+0291f960b6.nv26.04.48445190`
- CUDA reported by PyTorch: `13.2`
- DeepGEMM package version: `2.5.0`

## Corrected 8-Rank Summary

The Modal run completed all 8 benchmark cases. The original generated
`summary.csv` under-counted rank rows because multiple rank performance lines
were sometimes coalesced onto the same log line. The corrected summary below
parses every `EP:` occurrence from the raw logs and uses all 8 ranks per case.

### DeepSeek-V4-Flash

| Batch Size | Time (us) | Compute (TFLOPS) | Overlap (TFLOPS) | HBM (GB/s) | NVL (GB/s) | Ranks |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 69.5 | 4 | 4 | 1064 | 1 | 8/8 |
| 512 | 225.8 | 686 | 700 | 2050 | 171 | 8/8 |
| 8192 | 1320.4 | 1874 | 1982 | 968 | 484 | 8/8 |
| 32768 | 4883.2 | 2027 | 2154 | 789 | 526 | 8/8 |

### DeepSeek-V4-Pro

| Batch Size | Time (us) | Compute (TFLOPS) | Overlap (TFLOPS) | HBM (GB/s) | NVL (GB/s) | Ranks |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 186.8 | 4 | 4 | 1017 | 1 | 8/8 |
| 512 | 412.0 | 986 | 1005 | 4135 | 164 | 8/8 |
| 8192 | 2868.2 | 2264 | 2369 | 1074 | 385 | 8/8 |
| 32768 | 10610.9 | 2448 | 2571 | 695 | 418 | 8/8 |

## Conclusions

- The formal Mega MoE table sweep completed successfully on 8 x B200.
- Flash scales from about `686 TFLOPS` at batch size 512 to about `2027 TFLOPS`
  at batch size 32768.
- Pro reaches the highest compute throughput in this run, about `2448 TFLOPS`
  at batch size 32768.
- The `speedup_vs_legacy` column is not meaningful in this run because DeepEP
  was not installed in the image; every case printed `No module named
  'deep_ep', skip baseline benchmarking`.
- A parser fix was added to `modal_deepgemm_mega_moe.py` so future summaries
  count multiple rank rows on the same physical log line.

## Artifacts

- Corrected summary CSV: `deepgemm/results/20260528-044053/corrected_summary.csv`
- Corrected summary JSON: `deepgemm/results/20260528-044053/corrected_summary.json`
- Original Modal summary CSV: `deepgemm/results/20260528-044053/summary.csv`
- Original Modal summary JSON: `deepgemm/results/20260528-044053/summary.json`
- Raw logs: `deepgemm/results/20260528-044053/*.log`
