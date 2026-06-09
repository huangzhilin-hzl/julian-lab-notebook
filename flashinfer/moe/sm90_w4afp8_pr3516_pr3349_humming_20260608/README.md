# SM90 W4A/FP8 MoE benchmark: PR3516 vs PR3349 vs Humming

This directory records the clean nod35 run for comparing FlashInfer PR3516,
FlashInfer PR3349, and inclusionAI/Humming on H20 SM90.

## Scope

- Pod: `molou/molou-sglang-dev-h20-nod35-agent`
- GPU: NVIDIA H20, compute capability 9.0
- Runtime: Torch `2.10.0+cu128`, CUDA `12.8`
- Benchmark shape standard: PR3516 body shape, `E=8`, `top_k=1`, `H=4096`, `I=1024`, tokens `2048/8192/16384`
- Sampling: `warmup=20`, `repeat=100`

## Source revisions

- FlashInfer PR3516: `da61ed767f4d35cbd58e5c7c946eb57793058a22`
- FlashInfer PR3349: `60224d34f28b139b01831e77b773fd6f5916d769`
- Humming main: `3590529aade18f521d1d68ed9ef906a3d9a83bf4`

## Precision alignment

The runner now records a per-row precision contract in CSV/JSON. The three
paths are aligned on the broad W4A/FP8 target: BF16 source activation, FP8 E4M3
activation compute, MXFP4/E2M1 group-32 weights, BF16 output, FP32 routing
weights, and `top_k=1`.

| Backend | Activation quantization | Weight format | Intermediate | Output | Accumulation | Alignment note |
|---|---|---|---|---|---|---|
| PR3516 | Internal cast to `torch.float8_e4m3fn` without external scale | Packed MXFP4 E2M1, group 32, uint8 exponent scale | GEMM1 output FP8 E4M3 | BF16 | FP32 | Dtype family aligned; activation scale policy differs |
| PR3349 | Benchmark routine `quantize_fp8` with dequant scales passed to CUTLASS | BF16 random weights quantized to MXFP4 E2M1, group 32, SM90 interleaved | CUTLASS `mxfp4_fp8` hidden activation quantization | BF16 | CUTLASS kernel internal | Dtype family aligned; benchmark routine owns routing/data/cache policy |
| Humming | `ops.quant_input(..., float8e4m3, group_size=0)` for input and activated intermediate | `float4e2m1`, group 32, `float8e8m0` scales | BF16 SwiGLU then FP8 E4M3 for `w2` | BF16 | FP32 via `use_f16_accum=False` | Dtype family aligned; fuller scripted pipeline |

This is still not a strict numeric equivalence test. PR3516, PR3349, and
Humming do not consume identical packed weights and FP8 scales in this runner,
and PR3349 uses its own benchmark routine for data/routing generation. The
result metadata records `strict_precision_equivalent=false` with non-equivalent
axes: activation scale policy, router semantics, weight generation/layout,
intermediate dtype flow, timed scope, cache policy, and accumulation visibility.

## Artifacts

- HTML report: [results/nod35_w20_r100/index.html](results/nod35_w20_r100/index.html)
- Summary CSV: [results/nod35_w20_r100/summary_compare.csv](results/nod35_w20_r100/summary_compare.csv)
- Summary JSON: [results/nod35_w20_r100/summary_compare.json](results/nod35_w20_r100/summary_compare.json)
- PR3516 normalized CSV/JSON: [results/nod35_w20_r100/pr3516_w4a8_mxfp4_moe.csv](results/nod35_w20_r100/pr3516_w4a8_mxfp4_moe.csv), [results/nod35_w20_r100/pr3516_w4a8_mxfp4_moe.json](results/nod35_w20_r100/pr3516_w4a8_mxfp4_moe.json)
- PR3349 normalized CSV/JSON: [results/nod35_w20_r100/pr3349_mxfp4_fp8.csv](results/nod35_w20_r100/pr3349_mxfp4_fp8.csv), [results/nod35_w20_r100/pr3349_mxfp4_fp8.json](results/nod35_w20_r100/pr3349_mxfp4_fp8.json)
- PR3349 raw benchmark CSV: [results/nod35_w20_r100/pr3349_mxfp4_fp8_raw.csv](results/nod35_w20_r100/pr3349_mxfp4_fp8_raw.csv)
- PR3349 benchmark testlist: [results/nod35_w20_r100/pr3349_mxfp4_fp8_testlist.txt](results/nod35_w20_r100/pr3349_mxfp4_fp8_testlist.txt)
- Humming normalized CSV/JSON: [results/nod35_w20_r100/humming_main_fuller_pipeline.csv](results/nod35_w20_r100/humming_main_fuller_pipeline.csv), [results/nod35_w20_r100/humming_main_fuller_pipeline.json](results/nod35_w20_r100/humming_main_fuller_pipeline.json)
- GPU idle check: [results/nod35_w20_r100/gpu_idle_check_nod35_after_benchmark.txt](results/nod35_w20_r100/gpu_idle_check_nod35_after_benchmark.txt)
- Runner: [scripts/compare_sm90_w4afp8.py](scripts/compare_sm90_w4afp8.py)

## Result summary

| Backend | Tokens | Median ms | TFLOPs | Relative to PR3516 |
|---|---:|---:|---:|---:|
| PR3516 W4A8 MXFP4 MoE | 2048 | 0.657 | 78.412 | 1.000x latency |
| PR3516 W4A8 MXFP4 MoE | 8192 | 1.478 | 139.512 | 1.000x latency |
| PR3516 W4A8 MXFP4 MoE | 16384 | 2.557 | 161.258 | 1.000x latency |
| PR3349 MXFP4-FP8 CUTLASS MoE | 2048 | 0.781 | 66.015 | 1.188x latency |
| PR3349 MXFP4-FP8 CUTLASS MoE | 8192 | 2.855 | 72.222 | 1.932x latency |
| PR3349 MXFP4-FP8 CUTLASS MoE | 16384 | 5.656 | 72.905 | 2.212x latency |
| Humming fuller pipeline | 2048 | 0.320 | 161.166 | 2.055x faster |
| Humming fuller pipeline | 8192 | 0.998 | 206.479 | 1.480x faster |
| Humming fuller pipeline | 16384 | 1.938 | 212.759 | 1.319x faster |

## Reproduction commands

The three backends can be run from the pod after placing the three source trees
under `/tmp/flashinfer-pr3516-live`, `/tmp/flashinfer-pr3349-live`, and
`/tmp/humming-main-live`.

```bash
python3 /tmp/pr3516-pr3349-humming-sm90-w4afp8-20260608/compare_sm90_w4afp8.py \
  --backend pr3516 \
  --pr3516-repo /tmp/flashinfer-pr3516-live \
  --output-dir /tmp/pr3516-pr3349-humming-sm90-w4afp8-20260608/nod35_results_w20_r100 \
  --warmup 20 \
  --repeat 100

python3 /tmp/pr3516-pr3349-humming-sm90-w4afp8-20260608/compare_sm90_w4afp8.py \
  --backend pr3349 \
  --pr3349-repo /tmp/flashinfer-pr3349-live \
  --output-dir /tmp/pr3516-pr3349-humming-sm90-w4afp8-20260608/nod35_results_w20_r100 \
  --warmup 20 \
  --repeat 100

python3 /tmp/pr3516-pr3349-humming-sm90-w4afp8-20260608/compare_sm90_w4afp8.py \
  --backend humming \
  --humming-repo /tmp/humming-main-live \
  --output-dir /tmp/pr3516-pr3349-humming-sm90-w4afp8-20260608/nod35_results_w20_r100 \
  --warmup 20 \
  --repeat 100

python3 /tmp/pr3516-pr3349-humming-sm90-w4afp8-20260608/compare_sm90_w4afp8.py \
  --backend aggregate \
  --output-dir /tmp/pr3516-pr3349-humming-sm90-w4afp8-20260608/nod35_results_w20_r100
```

## Notes

- PR3516 is benchmarked as the CuTe DSL W4A8 MXFP4 full MoE path.
- PR3349 is benchmarked through its source-JIT CUTLASS fused MoE path with
  `--cutlass_variant mxfp4_fp8`.
- Humming is measured as a fuller pipeline: indexed `w13`, torch SwiGLU/fixed
  FP8 quantization, indexed `w2`, and combine. It is not the exact same fused
  kernel surface as the FlashInfer paths.
- The comparison is a mixed-scope observation from this script, not a strict
  apples-to-apples kernel benchmark. PR3516 times the full wrapper, PR3349 uses
  its benchmark routine's CUTLASS run scope and cold-L2 policy, and Humming uses
  a scripted fuller pipeline. Treat the speedups as script-level latency
  observations unless the benchmark is redesigned with identical timed scopes,
  routing distribution, and cache policy.
- A prior nod34 run was discarded for formal comparison because the node became
  fully occupied by another 8-GPU workload.
