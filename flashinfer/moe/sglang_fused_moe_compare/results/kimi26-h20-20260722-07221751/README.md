# Kimi K2.6 fused-MoE benchmark

- Date: 2026-07-22
- Pod: `molou/flashinfer-h20-sglang-dev-07221751`
- Pod UID: `595e3f66-f343-493e-b2f2-94c34ba0fa60`
- Node: `lj-2qd401272`
- GPU: NVIDIA H20, SM90; measurement used `cuda:1`
- Scope: `local_rank_fused_moe_no_router_no_comm`
- Timing: CUDA-event p50, warmup 20, repeat 100; wall-clock repeat 20
- Routing: balanced; `EP=1`

## Checkpoint-compatible group-32 W4A8 result

Workload: `K=7168`, `N_global=2048`, `N_local=256`, `E=384`, `top_k=8`,
`TP=8`. The synthetic weights use signed INT4 with BF16 group-32 scales; the
runtime activation path uses FP8 E4M3.

| M | Active experts | Humming p50 (ms) |
|---:|---:|---:|
| 1 | 8 | 0.4672 |
| 8 | 64 | 0.4715 |
| 32 | 256 | 0.4743 |
| 128 | 384 | 0.5772 |
| 512 | 384 | 0.6564 |
| 1024 | 384 | 0.8184 |
| 2048 | 384 | 1.1720 |
| 4096 | 384 | 2.2120 |
| 8192 | 384 | 4.2984 |

FlashInfer was deliberately fail-fast tested against this same workload. It
exited before weight construction with:

```text
FlashInfer's current SM90 packed-INT4 fused-MoE kernel requires
weight_group_size=128; got 32.
```

This is an implementation capability gap, not a failed performance sample, so
no FlashInfer latency or cross-backend speedup is reported for Kimi.

## Group-128 backend-coverage variants

为了覆盖 SGLang CUTLASS W4A8 与 Marlin W4A16，另建了两份复用相同 Kimi
problem shape、routing 和 seed 的 group-128 workload：

- CUTLASS：signed INT4/BF16-g128 weight × FP8 E4M3 activation；
- Marlin：signed INT4/BF16-g128 weight × BF16 activation。

group-128 是 backend coverage 用的合成变体，不代表 Kimi K2.6 checkpoint 的实际
group-32 weight layout。两列 activation contract 不同，因此只并列报告原生路径
latency，不计算或解读 cross-backend speedup。

| M | Active experts | SGLang CUTLASS W4A8 p50 (ms) | SGLang Marlin W4A16 p50 (ms) |
|---:|---:|---:|---:|
| 1 | 8 | 0.2417 | 0.1519 |
| 8 | 64 | 0.2742 | 0.1583 |
| 32 | 256 | 0.6809 | 0.5090 |
| 128 | 384 | 0.9856 | 0.7519 |
| 512 | 384 | 1.0607 | 1.0749 |
| 1024 | 384 | 1.1782 | 2.1225 |
| 2048 | 384 | 1.6690 | 3.0976 |
| 4096 | 384 | 3.1285 | 7.8986 |
| 8192 | 384 | 4.7172 | 12.1904 |

两条路径的 9 个输出 probe 均为有限值。该检查只用于发现明显执行错误，不等价于
跨 quant contract 的数值一致性验证。

## Runtime provenance and interpretation

- Humming: SGLang indexed fused-MoE integration, commit
  `c4ffdc1f9216fa7efe31aa95b4ac09bb56ac2dd6`.
- CUTLASS/Marlin: SGLang native fused-MoE runners. Marlin 的 GPTQ-to-Marlin
  repack 与 scale permutation 位于计时区间外。
- SGLang: `0.5.13.post2.dev180+g4ed3faca6`.
- PyTorch/CUDA: `2.11.0+cu130` / CUDA 13.0.
- The weights, inputs, and routes are synthetic but deterministic. The workload
  reproduces the Kimi problem size; the original Humming workload uses the
  checkpoint-compatible INT4 group-32 layout, while the two new workloads are
  explicit group-128 backend-coverage variants. None loads checkpoint tensor
  values.
- The benchmark excludes router and communication, so this is not end-to-end
  model latency.

Raw artifacts:

- [`kimi26.humming.json`](./kimi26.humming.json)
- [`kimi26.flashinfer.unsupported.log`](./kimi26.flashinfer.unsupported.log)
- [`kimi26.wint4-afp8-g128.sglang-cutlass.json`](./kimi26.wint4-afp8-g128.sglang-cutlass.json)
- [`kimi26.wint4-a16-g128.sglang-marlin.json`](./kimi26.wint4-a16-g128.sglang-marlin.json)
- [`backend-variants-g128.csv`](./backend-variants-g128.csv)
- [`workload.kimi-k2.6.wint4-afp8-g128.json`](../../workload.kimi-k2.6.wint4-afp8-g128.json)
- [`workload.kimi-k2.6.wint4-a16-g128.json`](../../workload.kimi-k2.6.wint4-a16-g128.json)
