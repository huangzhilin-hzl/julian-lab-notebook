# GLM-5.2 W4AFP8 fused-MoE benchmark

- Date: 2026-07-22
- Pod: `molou/flashinfer-h20-3e-sglang-dev-07221537`
- GPU: NVIDIA H20-3e, SM90
- Scope: `local_rank_fused_moe_no_router_no_comm`
- Workload: `K=6144`, `N_global=2048`, `N_local=256`, `E=256`, `top_k=8`, `TP=8`, `EP=1`, balanced routing
- Quant mode: signed INT4 group-128 weights with runtime FP8 E4M3 activations
- Timing: CUDA-event p50, warmup 20, repeat 100; wall-clock repeat 20

## Results

`FI/Humming` is the FlashInfer latency divided by Humming latency. Values above 1 mean Humming is faster.

| M | Active experts | Humming p50 (ms) | FlashInfer p50 (ms) | FI/Humming | Faster backend |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 8 | 0.4527 | 0.1324 | 0.292 | FlashInfer 3.42x |
| 8 | 64 | 0.4471 | 0.1913 | 0.428 | FlashInfer 2.34x |
| 32 | 256 | 0.4636 | 0.5669 | 1.223 | Humming 1.22x |
| 128 | 256 | 0.4466 | 0.5729 | 1.283 | Humming 1.28x |
| 512 | 256 | 0.4697 | 0.7024 | 1.496 | Humming 1.50x |
| 1024 | 256 | 0.5968 | 0.8909 | 1.493 | Humming 1.49x |
| 2048 | 256 | 1.1260 | 1.3796 | 1.225 | Humming 1.23x |
| 4096 | 256 | 1.5454 | 2.0797 | 1.346 | Humming 1.35x |
| 8192 | 256 | 3.0050 | 4.0183 | 1.337 | Humming 1.34x |

The crossover is between `M=8` and `M=32`: FlashInfer wins the two smallest-token cases, while Humming wins all seven cases from `M=32` through `M=8192`.

## Runtime provenance

- Humming: SGLang indexed fused-MoE integration, commit `c4ffdc1f9216fa7efe31aa95b4ac09bb56ac2dd6`.
- FlashInfer: source version `0.6.15`, local source commit `b35396c19ba97292a9efb9c7abf1bfce84868f85`, with autotuning enabled. Its isolated SM90 JIT cache was built in the pod; JIT time is outside the measurements.
- The Humming process used the pod's installed FlashInfer `0.6.12` only for the shared timing harness; the FlashInfer backend process used the source version above.
- SGLang: `0.5.13.post2.dev180+g4ed3faca6`.
- PyTorch/CUDA: `2.11.0+cu130` / CUDA 13.0.

## Validation and interpretation

- Both JSON files contain the same normalized workload and the expected nine `M` values.
- Every measured latency is positive, and every backend output sample passed the finite-value check.
- This is a performance comparison, not a strict cross-backend accuracy-parity result. The workload records `comparison_semantics=shared_weight_native_activation_quant` and `cross_backend_contract_matched=false`: weights, inputs and routing seeds are shared, but Humming uses its dynamic activation-quantization path while FlashInfer consumes the workload's static input/prequant scales and per-expert alpha.
- FlashInfer was autotuned for each token point; Humming used its fixed indexed integration path. The comparison represents each backend's native path, not identical algorithm-selection policy.
- The scope excludes router and communication, so these numbers must not be presented as end-to-end model latency.
- Raw results: [`glm52.humming.json`](./glm52.humming.json) and [`glm52.flashinfer.json`](./glm52.flashinfer.json).
