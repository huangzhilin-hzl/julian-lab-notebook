# FlashInfer PR 3084 H20 Benchmark

Run ID: `pr3084_h20_20260514_162622`

## Scope

This entry records the H20 measurement for FlashInfer PR #3084's SM90 mixed-input MoE path.

- Execution target: internal H20 benchmark pod (redacted)
- Node: redacted
- GPU: `NVIDIA H20`, compute capability `sm90`
- Python: `3.12.3` in the benchmark environment
- Torch: `2.11.0+cu129`
- FlashInfer source: `0.6.11` source checkout (path redacted)
- Timing: CUDA events through `flashinfer.testing.bench_gpu_time`
- Iterations: `30`

## Workload

- API: `flashinfer.fused_moe.cutlass_fused_moe`
- Quant path: MXFP4xBF16 / W4A16 with `use_w4_group_scaling=True`
- PR #3084 layout helpers enabled:
  - `interleave_moe_weights_for_sm90_mixed_gemm`
  - `interleave_moe_scales_for_sm90_mixed_gemm`
- Shape: `hidden=4096`, `intermediate=2048`, `experts=256`, `topk=6`
- `tune_max_num_tokens=16384`

## Results

| Batch | Autotune | Median ms | Std ms | Approx TFLOPS |
| ---: | :---: | ---: | ---: | ---: |
| 4 | no | 0.316032 | 0.020795 | 3.82 |
| 4 | yes | 0.285072 | 0.014960 | 4.24 |
| 16 | no | 0.878512 | 0.015811 | 5.50 |
| 16 | yes | 0.879040 | 0.014306 | 5.50 |
| 64 | no | 2.039408 | 0.016834 | 9.48 |
| 64 | yes | 2.035824 | 0.014183 | 9.49 |

`approx_tflops` is the script's rough MoE estimate:

```text
6 * batch * topk * hidden * intermediate / median_time
```

It is useful for same-script comparisons only, not as a hardware SOL number.

## Caveat

The installed `flashinfer 0.6.2` baseline was attempted, but first-time full JIT compilation was too slow and was stopped. The incomplete baseline log is intentionally not committed.

## Files

- `scripts/bench_pr3084_moe.py`: benchmark driver

Raw logs and CSV/JSON result artifacts are intentionally not committed. This
keeps the notebook entry limited to the runnable script and the summarized,
redacted benchmark table above.
