# DeepSeek-V4 Flash CP8 TP8 No-DeepEP Compare

Best run per scenario, taking the minimum mean TTFT/TPOT across 3 rounds.

## TTFT

| variant | input_len | output_len | batch_size | round | metric_ms | output_tput_tok_s |
|---|---:|---:|---:|---:|---:|---:|
| humming_mxfp4a16 | 16384 | 256 | 1 | 2 | 806.32 | 129.17 |
| humming_mxfp4a16 | 32768 | 256 | 1 | 1 | 1598.96 | 91.78 |
| humming_mxfp4a16 | 65536 | 256 | 1 | 1 | 3256.50 | 56.78 |
| humming_mxfp4a16 | 131072 | 256 | 1 | 1 | 6722.61 | 31.78 |
| marlin_mxfp4a16 | 16384 | 256 | 1 | 1 | 1069.50 | 111.88 |
| marlin_mxfp4a16 | 32768 | 256 | 1 | 1 | 2130.21 | 76.06 |
| marlin_mxfp4a16 | 65536 | 256 | 1 | 3 | 4285.22 | 46.04 |
| marlin_mxfp4a16 | 131072 | 256 | 1 | 2 | 8841.49 | 25.18 |

## TPOT

| variant | input_len | output_len | batch_size | round | metric_ms | output_tput_tok_s |
|---|---:|---:|---:|---:|---:|---:|
| humming_mxfp4a16 | 1024 | 1024 | 1 | 1 | 4.31 | 220.89 |
| humming_mxfp4a16 | 1024 | 1024 | 4 | 3 | 5.01 | 721.63 |
| humming_mxfp4a16 | 1024 | 1024 | 8 | 1 | 6.08 | 1130.95 |
| humming_mxfp4a16 | 32768 | 1024 | 1 | 2 | 4.49 | 165.59 |
| humming_mxfp4a16 | 32768 | 1024 | 4 | 1 | 7.46 | 343.38 |
| humming_mxfp4a16 | 32768 | 1024 | 8 | 2 | 11.35 | 424.39 |
| humming_mxfp4a16 | 65536 | 1024 | 1 | 3 | 4.77 | 126.72 |
| humming_mxfp4a16 | 65536 | 1024 | 4 | 2 | 9.95 | 221.46 |
| humming_mxfp4a16 | 65536 | 1024 | 8 | 2 | 17.07 | 252.30 |
| humming_mxfp4a16 | 131072 | 1024 | 1 | 1 | 4.82 | 88.40 |
| humming_mxfp4a16 | 131072 | 1024 | 4 | 2 | 15.03 | 125.25 |
| humming_mxfp4a16 | 131072 | 1024 | 8 | 3 | 29.29 | 136.08 |
| marlin_mxfp4a16 | 1024 | 1024 | 1 | 3 | 4.39 | 218.78 |
| marlin_mxfp4a16 | 1024 | 1024 | 4 | 1 | 5.15 | 668.50 |
| marlin_mxfp4a16 | 1024 | 1024 | 8 | 2 | 6.22 | 1071.33 |
| marlin_mxfp4a16 | 32768 | 1024 | 1 | 2 | 4.54 | 151.72 |
| marlin_mxfp4a16 | 32768 | 1024 | 4 | 2 | 8.29 | 287.91 |
| marlin_mxfp4a16 | 32768 | 1024 | 8 | 2 | 13.12 | 344.55 |
| marlin_mxfp4a16 | 65536 | 1024 | 1 | 2 | 4.64 | 114.14 |
| marlin_mxfp4a16 | 65536 | 1024 | 4 | 3 | 11.64 | 178.35 |
| marlin_mxfp4a16 | 65536 | 1024 | 8 | 1 | 20.85 | 198.98 |
| marlin_mxfp4a16 | 131072 | 1024 | 1 | 2 | 4.85 | 74.31 |
| marlin_mxfp4a16 | 131072 | 1024 | 4 | 1 | 18.37 | 99.97 |
| marlin_mxfp4a16 | 131072 | 1024 | 8 | 3 | 36.40 | 106.31 |

