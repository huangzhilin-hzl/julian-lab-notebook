# PR24289 H20 DeepSeek-V4 recipe comparison

## Analysis

- Low-Latency is effectively throughput-neutral across the mixed-checkpoint W4A16 paths. Humming W4A16 reaches 173.48 output tok/s, only +0.4% over Marlin W4A16 and -2.7% versus the FP8 reference. Its main advantage in this recipe is latency shape: mean TTFT is the lowest at 153.58 ms, and mean TPOT is also the lowest at 4.99 ms.
- Balanced is the strongest general-purpose result for Humming. Humming W4A16 reaches 446.29 output tok/s, +23.7% over Marlin W4A16 and +48.5% versus the FP8 reference. Humming W4A8 is close at 440.26 output tok/s, but W4A16 has better TTFT and TPOT in this recipe.
- Max-Throughput shows the clearest Humming gain. Humming W4A8 is the best result at 655.47 output tok/s, +57.1% over Marlin W4A16 and +36.2% versus the FP8 reference. Humming W4A16 is also strong at 619.74 output tok/s. Marlin W4A16 is weak in this recipe because both TTFT and TPOT regress heavily.
- Context-Parallel is functionally improved but not performance-ready in this run. Humming W4A16 and W4A8 both start and complete the benchmark, while Marlin W4A16 is not usable for this path. However, Humming CP throughput is much lower than the FP8 reference: 19.15 and 16.61 output tok/s versus 81.89 output tok/s.
- Accuracy sanity is stable for the first three recipes: all available configs pass 5/5. Context-Parallel remains noisy: FP8 is 3/5, Humming W4A16 is 2/5, and Humming W4A8 is 4/5.

## Conclusion

Humming is the preferred mixed-checkpoint backend for Balanced and Max-Throughput on H20. W4A16 is the safer default for Balanced because it keeps better latency while delivering the highest throughput in that recipe. W4A8 is the best Max-Throughput option and gives the top overall output tok/s in this run.

Low-Latency does not show a meaningful throughput win from Humming, though Humming W4A16 has the best TTFT/TPOT shape. Context-Parallel should be treated as startup/functionality validation only for now: the fix lets Humming run through the CP path, but the measured throughput and sanity results are not yet competitive with the FP8 reference.

## low-latency

| config | startup | sanity | req/s | output tok/s | vs fp8 | mean TTFT ms | mean TPOT ms | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fp8_baseline | True | 5/5 | 1.16 | 178.29 | +0.0% | 166.29 | 5.09 | - |
| marlin_w4a16 | True | 5/5 | 1.12 | 172.73 | -3.1% | 157.57 | 5.34 | - |
| humming_w4a16 | True | 5/5 | 1.13 | 173.48 | -2.7% | 153.58 | 4.99 | - |
| humming_w4a8 | True | 5/5 | 1.11 | 170.46 | -4.4% | 171.19 | 5.19 | - |

## balanced

| config | startup | sanity | req/s | output tok/s | vs fp8 | mean TTFT ms | mean TPOT ms | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fp8_baseline | True | 5/5 | 2.21 | 300.62 | +0.0% | 546.99 | 22.52 | - |
| marlin_w4a16 | True | 5/5 | 2.66 | 360.83 | +20.0% | 985.34 | 14.05 | - |
| humming_w4a16 | True | 5/5 | 3.29 | 446.29 | +48.5% | 423.88 | 15.07 | - |
| humming_w4a8 | True | 5/5 | 3.24 | 440.26 | +46.5% | 465.98 | 18.12 | - |

## max-throughput

| config | startup | sanity | req/s | output tok/s | vs fp8 | mean TTFT ms | mean TPOT ms | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fp8_baseline | True | 5/5 | 4.36 | 481.2 | +0.0% | 1072.61 | 54.34 | - |
| marlin_w4a16 | True | 5/5 | 3.78 | 417.12 | -13.3% | 3256.0 | 63.39 | - |
| humming_w4a16 | True | 5/5 | 5.62 | 619.74 | +28.8% | 857.3 | 43.97 | - |
| humming_w4a8 | True | 5/5 | 5.94 | 655.47 | +36.2% | 778.15 | 41.71 | - |

## context-parallel

| config | startup | sanity | req/s | output tok/s | vs fp8 | mean TTFT ms | mean TPOT ms | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fp8_baseline | True | 3/5 | 1.2 | 81.89 | +0.0% | 358.28 | 19.37 | - |
| marlin_w4a16 | False | - | - | - | - | - | - | unsupported_by_docs_for_h200_fp4_and_marlin_lacks_deepep_permute_path |
| humming_w4a16 | True | 2/5 | 0.28 | 19.15 | -76.6% | 501.28 | 95.38 | - |
| humming_w4a8 | True | 4/5 | 0.24 | 16.61 | -79.7% | 500.74 | 111.31 | - |
