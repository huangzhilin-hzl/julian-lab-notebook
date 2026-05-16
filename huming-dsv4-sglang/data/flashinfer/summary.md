# DeepSeek-V4 Flash CP8 TP8 No-DeepEP Compare

Best run per scenario, taking the minimum mean TTFT/TPOT across 3 rounds.

## TTFT

| variant | input_len | output_len | batch_size | round | metric_ms | output_tput_tok_s |
|---|---:|---:|---:|---:|---:|---:|
| flashinfer_mxfp4a16 | 16384 | 256 | 1 | 3 | 1108.10 | 105.26 |
| flashinfer_mxfp4a16 | 32768 | 256 | 1 | 2 | 2211.63 | 72.40 |
| flashinfer_mxfp4a16 | 65536 | 256 | 1 | 2 | 4462.79 | 43.78 |
| flashinfer_mxfp4a16 | 131072 | 256 | 1 | 3 | 9174.90 | 23.98 |

## TPOT

| variant | input_len | output_len | batch_size | round | metric_ms | output_tput_tok_s |
|---|---:|---:|---:|---:|---:|---:|
| flashinfer_mxfp4a16 | 1024 | 1024 | 1 | 1 | 4.82 | 200.23 |
| flashinfer_mxfp4a16 | 1024 | 1024 | 4 | 1 | 6.06 | 572.71 |
| flashinfer_mxfp4a16 | 1024 | 1024 | 8 | 2 | 7.30 | 939.68 |
| flashinfer_mxfp4a16 | 32768 | 1024 | 1 | 1 | 5.01 | 140.08 |
| flashinfer_mxfp4a16 | 32768 | 1024 | 4 | 1 | 9.28 | 264.91 |
| flashinfer_mxfp4a16 | 32768 | 1024 | 8 | 3 | 14.53 | 319.23 |
| flashinfer_mxfp4a16 | 65536 | 1024 | 1 | 2 | 5.17 | 105.63 |
| flashinfer_mxfp4a16 | 65536 | 1024 | 4 | 2 | 12.71 | 167.06 |
| flashinfer_mxfp4a16 | 65536 | 1024 | 8 | 1 | 22.34 | 188.24 |
| flashinfer_mxfp4a16 | 131072 | 1024 | 1 | 1 | 5.36 | 70.22 |
| flashinfer_mxfp4a16 | 131072 | 1024 | 4 | 1 | 19.66 | 94.53 |
| flashinfer_mxfp4a16 | 131072 | 1024 | 8 | 3 | 38.73 | 100.62 |

