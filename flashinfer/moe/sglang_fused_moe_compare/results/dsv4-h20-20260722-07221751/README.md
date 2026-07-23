# DeepSeek-V4 Flash and Pro fused-MoE benchmark

- Date: 2026-07-22
- Pod: `molou/flashinfer-h20-sglang-dev-07221751`
- Pod UID: `595e3f66-f343-493e-b2f2-94c34ba0fa60`
- Node: `lj-2qd401272`
- GPU: NVIDIA H20, SM90; all measurements used `cuda:1`
- Scope: `local_rank_fused_moe_no_router_no_comm`
- Timing: CUDA-event p50, warmup 20, repeat 100; wall-clock repeat 20
- Routing: balanced; `EP=1`

## Summary

- DSV4 Flash: FlashInfer wins `M=1/8`; `M=32` is near parity in this run
  (FlashInfer p50 is 1.8% lower); Humming wins `M=128–8192`.
- DSV4 Pro: FlashInfer wins `M=1/8`; Humming wins `M=32–8192`.
- SGLang Marlin W4A16-g128 已补齐 Flash/Pro 的全部 9 个 M 点，p50 分别为
  `0.1851–8.8467 ms` 和 `0.1842–11.7063 ms`；它们是独立 quant contract，
  不与上面的原生 MXFP4 结果计算 speedup。

## DeepSeek-V4 Flash MXFP4AFP8

Workload: `K=4096`, `N_global=2048`, `N_local=512`, `E=256`, `top_k=6`,
`TP=4`. `FI/Humming` is FlashInfer latency divided by Humming latency; values
above 1 mean Humming is faster.

| M | Active experts | Humming p50 (ms) | FlashInfer p50 (ms) | FI/Humming | Faster backend |
|---:|---:|---:|---:|---:|---|
| 1 | 6 | 0.4804 | 0.2656 | 0.553 | FlashInfer 1.81x |
| 8 | 48 | 0.4884 | 0.3079 | 0.630 | FlashInfer 1.59x |
| 32 | 192 | 0.5032 | 0.4945 | 0.983 | FlashInfer 1.02x |
| 128 | 256 | 0.4963 | 0.6246 | 1.258 | Humming 1.26x |
| 512 | 256 | 0.5072 | 0.6859 | 1.352 | Humming 1.35x |
| 1024 | 256 | 0.4981 | 0.8250 | 1.656 | Humming 1.66x |
| 2048 | 256 | 0.9096 | 1.3656 | 1.501 | Humming 1.50x |
| 4096 | 256 | 1.7212 | 2.0766 | 1.206 | Humming 1.21x |
| 8192 | 256 | 2.6089 | 3.9022 | 1.496 | Humming 1.50x |

The sign changes between `M=32` and `M=128`; the `M=32` point itself is near
parity and should not be treated as a robust advantage without repeated runs.

## DeepSeek-V4 Pro MXFP4AFP8

Workload: `K=7168`, `N_global=3072`, `N_local=384`, `E=384`, `top_k=6`,
`TP=8`.

| M | Active experts | Humming p50 (ms) | FlashInfer p50 (ms) | FI/Humming | Faster backend |
|---:|---:|---:|---:|---:|---|
| 1 | 6 | 0.4826 | 0.2730 | 0.566 | FlashInfer 1.77x |
| 8 | 48 | 0.4871 | 0.3421 | 0.702 | FlashInfer 1.42x |
| 32 | 192 | 0.4969 | 0.5803 | 1.168 | Humming 1.17x |
| 128 | 384 | 0.7348 | 0.9226 | 1.256 | Humming 1.26x |
| 512 | 384 | 0.8093 | 1.0188 | 1.259 | Humming 1.26x |
| 1024 | 384 | 0.9918 | 1.2898 | 1.300 | Humming 1.30x |
| 2048 | 384 | 1.3427 | 1.6051 | 1.195 | Humming 1.20x |
| 4096 | 384 | 2.5616 | 2.9018 | 1.133 | Humming 1.13x |
| 8192 | 384 | 3.4839 | 4.8012 | 1.378 | Humming 1.38x |

The crossover is between `M=8` and `M=32`.

## SGLang Marlin W4A16-g128（替代量化 contract）

以下测试复用 DSV4 Flash/Pro 的 problem shape、TP、routing 和 seed，但把 expert
weight contract 改为 symmetric signed INT4/BF16-g128，activation 保留 BF16。
DSV4 的原始 workload 是 MXFP4AFP8；因此这些是 shape-matched backend coverage
结果，不是 DSV4 checkpoint 的原生量化布局，也不能和 MXFP4 表直接计算 backend
speedup。

### DeepSeek-V4 Flash shape

| M | Active experts | SGLang Marlin W4A16 p50 (ms) |
|---:|---:|---:|
| 1 | 6 | 0.1851 |
| 8 | 48 | 0.1944 |
| 32 | 192 | 0.4246 |
| 128 | 256 | 0.5654 |
| 512 | 256 | 0.8118 |
| 1024 | 256 | 1.5321 |
| 2048 | 256 | 2.9152 |
| 4096 | 256 | 5.7652 |
| 8192 | 256 | 8.8467 |

### DeepSeek-V4 Pro shape

| M | Active experts | SGLang Marlin W4A16 p50 (ms) |
|---:|---:|---:|
| 1 | 6 | 0.1842 |
| 8 | 48 | 0.1918 |
| 32 | 192 | 0.5545 |
| 128 | 384 | 1.0702 |
| 512 | 384 | 1.5249 |
| 1024 | 384 | 2.9356 |
| 2048 | 384 | 4.2614 |
| 4096 | 384 | 5.8748 |
| 8192 | 384 | 11.7063 |

两份 Marlin 结果的 9 个输出 probe 均为有限值。GPTQ-to-Marlin repack 和 scale
permutation 位于计时区间外；计时 callable 走 SGLang `MoeRunner(MARLIN)`。

## Runtime provenance and interpretation

- Humming: SGLang indexed fused-MoE integration, commit
  `c4ffdc1f9216fa7efe31aa95b4ac09bb56ac2dd6`.
- FlashInfer: source version `0.6.15`, local source commit
  `b35396c19ba97292a9efb9c7abf1bfce84868f85`, with per-shape autotuning.
- SGLang: `0.5.13.post2.dev180+g4ed3faca6`.
- PyTorch/CUDA: `2.11.0+cu130` / CUDA 13.0.
- Container image: `deepep-base@sha256:01a0295be625f33760142b100256bd6d808f10b5033ed72277fdc3be48e310be`.
- The original Humming/FlashInfer sections use contract-matched MXFP4
  E2M1/E8M0-g32 × FP8 E4M3 providers. The new Marlin sections use the separate
  signed INT4/BF16-g128 × BF16 contract. FlashInfer autotuning and all weight
  transforms/repacking are outside the recorded latency interval.
- The weights, inputs, and routes are synthetic but deterministic. The original
  MXFP4 workloads reproduce model problem sizes and quantization layout; the
  Marlin variants reproduce only the problem sizes. None loads checkpoint tensor
  values.
- The benchmark excludes router and communication, so these numbers are not
  end-to-end model latency.

Raw artifacts:

- [`dsv4-flash.humming.json`](./dsv4-flash.humming.json)
- [`dsv4-flash.flashinfer.json`](./dsv4-flash.flashinfer.json)
- [`dsv4-pro.humming.json`](./dsv4-pro.humming.json)
- [`dsv4-pro.flashinfer.json`](./dsv4-pro.flashinfer.json)
- [`comparison.csv`](./comparison.csv)
- [`dsv4-flash.wint4-a16-g128.sglang-marlin.json`](./dsv4-flash.wint4-a16-g128.sglang-marlin.json)
- [`dsv4-pro.wint4-a16-g128.sglang-marlin.json`](./dsv4-pro.wint4-a16-g128.sglang-marlin.json)
- [`marlin-w4a16-g128.csv`](./marlin-w4a16-g128.csv)
- [`workload.dsv4-flash.wint4-a16-g128.json`](../../workload.dsv4-flash.wint4-a16-g128.json)
- [`workload.dsv4-pro.wint4-a16-g128.json`](../../workload.dsv4-pro.wint4-a16-g128.json)
