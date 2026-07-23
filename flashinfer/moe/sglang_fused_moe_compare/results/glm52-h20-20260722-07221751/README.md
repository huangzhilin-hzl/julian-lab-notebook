# GLM-5.2 W4AFP8 fused-MoE benchmark

- Date: 2026-07-22
- Pod: `molou/flashinfer-h20-sglang-dev-07221751`
- Pod UID: `595e3f66-f343-493e-b2f2-94c34ba0fa60`
- Node: `lj-2qd401272`
- GPU: NVIDIA H20, SM90; both backends measured on `cuda:1`
- Driver: 550.127.08
- Scope: `local_rank_fused_moe_no_router_no_comm`
- Workload: `K=6144`, `N_global=2048`, `N_local=256`, `E=256`, `top_k=8`, `TP=8`, `EP=1`, balanced routing
- Quant mode: signed INT4 group-128 weights with runtime FP8 E4M3 activations
- Timing: CUDA-event p50, warmup 20, repeat 100; wall-clock repeat 20

## Results

`FI/Humming` is the FlashInfer latency divided by Humming latency. Values above 1 mean Humming is faster.

| M | Active experts | Humming p50 (ms) | FlashInfer p50 (ms) | FI/Humming | Faster backend |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 8 | 0.4693 | 0.1368 | 0.292 | FlashInfer 3.43x |
| 8 | 64 | 0.4664 | 0.1901 | 0.408 | FlashInfer 2.45x |
| 32 | 256 | 0.4817 | 0.5656 | 1.174 | Humming 1.17x |
| 128 | 256 | 0.4652 | 0.5767 | 1.240 | Humming 1.24x |
| 512 | 256 | 0.4947 | 0.7095 | 1.434 | Humming 1.43x |
| 1024 | 256 | 0.6059 | 0.9017 | 1.488 | Humming 1.49x |
| 2048 | 256 | 1.1329 | 1.3952 | 1.232 | Humming 1.23x |
| 4096 | 256 | 1.5721 | 2.1747 | 1.383 | Humming 1.38x |
| 8192 | 256 | 3.0734 | 4.0592 | 1.321 | Humming 1.32x |

The crossover is between `M=8` and `M=32`: FlashInfer wins the two smallest-token cases, while Humming wins all seven cases from `M=32` through `M=8192`.

## Runtime provenance

- Humming: SGLang indexed fused-MoE integration, commit `c4ffdc1f9216fa7efe31aa95b4ac09bb56ac2dd6`.
- FlashInfer: source version `0.6.15`, local source commit `b35396c19ba97292a9efb9c7abf1bfce84868f85`, with per-shape autotuning enabled.
- The source tree used an isolated SM90 JIT cache. JIT compilation and autotuning are outside the recorded latency samples.
- The pod's preinstalled FlashInfer/JIT artifacts were version 0.6.12. The synced 0.6.15 source copy was configured to ignore those old artifacts and build its own `fused_moe_90.so`; the benchmark API and kernels were otherwise unchanged.
- Both backend processes imported FlashInfer 0.6.15 for the shared timing harness.
- SGLang: `0.5.13.post2.dev180+g4ed3faca6`.
- PyTorch/CUDA: `2.11.0+cu130` / CUDA 13.0.
- Container image: `deepep-base@sha256:01a0295be625f33760142b100256bd6d808f10b5033ed72277fdc3be48e310be`.

The originally supplied pod object was deleted at 18:06 CST. These measurements were produced after recreating the same pod name; the UID and node above identify the actual runtime used. GPU0 already had an external driver allocation, so both measurements used the otherwise-idle GPU1.

## Validation and interpretation

- Both JSON files contain the same normalized workload and the expected nine `M` values.
- Every measured GPU/wall latency is positive, and every backend output sample passed the finite-value check.
- This is a performance comparison, not a strict cross-backend accuracy-parity result. The workload records `comparison_semantics=shared_weight_native_activation_quant` and `cross_backend_contract_matched=false`: weights, inputs and routing seeds are shared, but Humming uses its dynamic activation-quantization path while FlashInfer consumes the workload's static input/prequant scales and per-expert alpha.
- FlashInfer was autotuned for each token point; Humming used its fixed indexed integration path. The comparison represents each backend's native path, not identical algorithm-selection policy.
- The scope excludes router and communication, so these numbers must not be presented as end-to-end model latency.
- Raw results: [`glm52.humming.json`](./glm52.humming.json), [`glm52.flashinfer.json`](./glm52.flashinfer.json), and [`comparison.csv`](./comparison.csv).
