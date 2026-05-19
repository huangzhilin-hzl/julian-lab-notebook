# DeepSeek-V4 Flash NVFP4 MoE Benchmark on Modal B200

Date: 2026-05-19

Modal run:
https://modal.com/apps/huangzhilin-hzl/main/ap-pwMXl7LRJ3kS20nd6APrQN

Command:

```bash
FLASHINFER_SRC_DIR=/Users/xielu/julian/flashinfer \
MODAL_GPU=B200 \
python3 -m modal run flashinfer/moe/deepseek_v3_nvfp4/scripts/modal_bench_moe_deepseek.py \
  --model deepseek-v4-flash \
  --phase both \
  --ep 1 \
  --warmup 5 \
  --iters 30
```

Remote logs:

```text
/cache/flashinfer/results/moe/deepseek_v4_flash/20260519-111045/deepseek_v4_flash_nvfp4_prefill.log
/cache/flashinfer/results/moe/deepseek_v4_flash/20260519-121006/deepseek_v4_flash_nvfp4_decode.log
```

Configuration:

- GPU: NVIDIA B200
- FlashInfer source: `huangzhilin-hzl/flashinfer`, branch `codex/dsv4-moe-benchmark`
- FlashInfer commit: `4020ca09` (`benchmarks: add DeepSeek V4 MoE configs`)
- Benchmark: `benchmarks/bench_moe_deepseek.py`
- Model: `deepseek-v4-flash`
- Model shape: hidden=4096, intermediate=2048, experts=256, top_k=6
- EP: 1, 256 local experts
- CUDA graph: enabled
- Routing bias scale: 0.01
- Warmup: 5
- Iterations: 30
- CUPTI was requested by the benchmark, but the runtime warned that CUPTI was not installed and fell back to CUDA events.

Prefill results:

| Tokens | CuteDSL ms | CuteDSL TFLOPS | CUTLASS ms | CUTLASS TFLOPS | TRTLLM ms | TRTLLM TFLOPS | CuteDSL/CUTLASS | CuteDSL/TRTLLM | Winner | Active experts | Expert load min/max/median |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | :--- |
| 128 | 0.554 | 69.8 | 0.553 | 69.8 | 0.490 | 78.9 | 1.00x | 0.88x | TRTLLM | 240 | 0/9/3.00 |
| 256 | 0.597 | 129.5 | 0.605 | 127.8 | 0.525 | 147.2 | 1.01x | 0.88x | TRTLLM | 255 | 0/15/6.00 |
| 512 | 0.603 | 256.6 | 0.630 | 245.3 | 0.539 | 287.0 | 1.05x | 0.89x | TRTLLM | 256 | 2/26/11.00 |
| 1024 | 0.616 | 502.2 | 0.680 | 454.7 | 0.566 | 546.5 | 1.10x | 0.92x | TRTLLM | 256 | 7/46/24.00 |
| 2048 | 0.642 | 962.7 | 0.804 | 768.9 | 0.676 | 914.6 | 1.25x | 1.05x | CuteDSL | 256 | 22/81/47.00 |
| 4096 | 0.694 | 1782.5 | 1.149 | 1076.9 | 0.749 | 1651.6 | 1.66x | 1.08x | CuteDSL | 256 | 45/150/96.00 |
| 8192 | 0.987 | 2507.7 | 1.737 | 1424.5 | 1.267 | 1953.2 | 1.76x | 1.28x | CuteDSL | 256 | 84/296/189.00 |

Decode results:

| Tokens | CuteDSL ms | CuteDSL TFLOPS | CUTLASS ms | CUTLASS TFLOPS | TRTLLM ms | TRTLLM TFLOPS | CuteDSL/CUTLASS | CuteDSL/TRTLLM | Winner | Active experts | Expert load min/max/median |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | :--- |
| 1 | 0.027 | 11.3 | 0.045 | 6.7 | 0.025 | 12.2 | 1.67x | 0.92x | TRTLLM | 6 | 0/1/0.00 |
| 2 | 0.053 | 11.4 | 0.062 | 9.7 | 0.041 | 14.8 | 1.18x | 0.77x | TRTLLM | 12 | 0/1/0.00 |
| 4 | 0.076 | 16.0 | 0.086 | 14.0 | 0.063 | 19.2 | 1.14x | 0.83x | TRTLLM | 23 | 0/2/0.00 |
| 8 | 0.116 | 20.8 | 0.132 | 18.3 | 0.105 | 23.1 | 1.13x | 0.90x | TRTLLM | 44 | 0/3/0.00 |
| 16 | 0.191 | 25.3 | 0.208 | 23.2 | 0.177 | 27.3 | 1.09x | 0.93x | TRTLLM | 81 | 0/3/0.00 |
| 32 | 0.321 | 30.1 | 0.328 | 29.4 | 0.287 | 33.7 | 1.02x | 0.89x | TRTLLM | 139 | 0/4/1.00 |
| 64 | 0.462 | 41.8 | 0.465 | 41.6 | 0.409 | 47.3 | 1.01x | 0.88x | TRTLLM | 201 | 0/6/1.00 |
| 128 | 0.551 | 70.1 | 0.553 | 69.9 | 0.490 | 78.8 | 1.00x | 0.89x | TRTLLM | 240 | 0/9/3.00 |

Notes:

- The Modal runner was updated to support `deepseek-v4-flash`, `deepseek-v4-pro`, and explicit `prefill`, `decode`, or `both` phases.
- Default prefill tokens are `128,256,512,1024,2048,4096,8192`; default decode tokens are `1,2,4,8,16,32,64,128`.
- `--gen-phase` remains supported as a backward-compatible alias for decode when `--phase` is not specified.
