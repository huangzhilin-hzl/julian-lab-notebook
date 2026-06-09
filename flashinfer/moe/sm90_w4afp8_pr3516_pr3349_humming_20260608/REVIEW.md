# Review notes

This note records the follow-up review of `scripts/compare_sm90_w4afp8.py` and
the nod35 result artifacts.

## Verdict

The data files are internally consistent and the Julian Lab copy matches the
source result directory byte-for-byte for the runner and result artifacts after
the fixes below. The benchmark is still a mixed-scope observation, not a strict
apples-to-apples kernel comparison.

## High-impact methodology caveats

1. PR3516 times the full `w4a8_mxfp4_moe(...)` wrapper. That wrapper includes
   quantization/cast, routing work, buffer allocation, GEMM calls, and output
   handling inside the measured call.
2. PR3349 uses the FlashInfer benchmark routine and records the CUTLASS
   `mxfp4_fp8` run scope. That routine also uses the benchmark helper's cold-L2
   policy, unlike the local CUDA-event loop used for PR3516 and Humming.
3. Humming is measured as a fuller scripted pipeline: indexed `w13`, torch
   SwiGLU/fixed FP8 quantization, indexed `w2`, and combine. It is not the same
   fused-kernel surface as the FlashInfer paths.
4. Routing is not identical across all backends: PR3516 and Humming use balanced
   cyclic routing in the runner, while PR3349 source-JIT uses its benchmark
   routine's router-logits softmax/topk path.
5. Precision is aligned only at the dtype-family level. PR3516 casts activation
   to FP8 E4M3 without an external scale, while PR3349 and Humming use FP8
   quantization paths with explicit scales.

These caveats mean the reported speedups should be read as the latency observed
for this scripted setup on nod35, not as definitive kernel-level speedups.

## Fixes made after review

- Added missing PR3349 benchmark testlist artifact:
  `results/nod35_w20_r100/pr3349_mxfp4_fp8_testlist.txt`.
- Added GPU idle evidence:
  `results/nod35_w20_r100/gpu_idle_check_nod35_after_benchmark.txt`.
- Expanded HTML and README artifact lists to include per-backend JSON, PR3349
  raw CSV/testlist, and GPU idle evidence.
- Corrected runner metadata wording from the stale `cutlass W4A16` phrase to the
  actual `cutlass_variant=mxfp4_fp8` setup.
- Changed future PR3349 normalization so `mean_time_ms` is blank when the raw
  benchmark CSV does not provide a mean, instead of copying the median into the
  mean field.
- Changed future aggregate baseline keys from only `num_tokens` to the full
  `(num_tokens, hidden_size, intermediate_size, num_experts, top_k)` shape.
- Added per-backend precision contract metadata and CSV fields:
  `source_activation_dtype`, `activation_compute_dtype`, `activation_quantization`,
  `weight_format`, `weight_generation`, `weight_scale_layout`, `router_source`,
  `router_weight_semantics`, `output_dtype`, and `accumulation_dtype`.
- Added `strict_precision_equivalent=false` and `non_equivalent_axes` metadata to
  mark activation scale policy, router semantics, weight generation/layout,
  intermediate dtype flow, timed scope, cache policy, and accumulation visibility
  as not strictly equivalent across the three paths.
- Added future runtime precision assertions for PR3516, PR3349 direct fallback,
  and Humming tensor dtypes/quant-scale signatures.

## Data consistency checks

- `summary_compare.csv` and `summary_compare.json` have 9 matching rows.
- The three per-backend CSV files match the corresponding rows in
  `summary_compare.csv`.
- Each normalized row now carries the precision contract fields used by the HTML
  and README.
- Source result artifacts and Julian Lab result artifacts match by SHA256 after
  the fixes.
