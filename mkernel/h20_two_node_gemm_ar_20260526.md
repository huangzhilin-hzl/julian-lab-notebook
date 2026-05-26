# mKernel H20 Two-Node GEMM-AR Benchmark

Date: 2026-05-26 Asia/Shanghai

Source: [uccl-project/mKernel](https://github.com/uccl-project/mKernel)

## Privacy Note

This note is intentionally sanitized for a public/personal notebook. It omits exact Kubernetes pod names, node hostnames, cluster IPs, SSH aliases, key paths, internal filesystem paths, and raw command logs. The retained information is limited to public repository context, generalized hardware/runtime facts, benchmark methodology, numerical results, and non-sensitive debugging conclusions.

## Test Scope

- Hardware: private two-node NVIDIA H20 environment, `2 x 8` GPUs, `world=16`.
- Runtime: CUDA 13 class environment with PyTorch and NCCL available in the benchmark containers.
- Benchmark target: homepage-style mKernel multi-node tests.
- Completed valid sweep: `gemm_ar`.
- Baseline: CuBLAS+NCCL all-reduce GEMM baseline using the same H20 pair and shape list.
- Measurement settings: `WARMUP=2`, `ITERS=10`.
- Shape list: `M=N=2048,4096,8192,16384,32768`, BF16 GEMM-style throughput reported as TFLOPS per GPU.

## Result Summary

| M=N | CuBLAS+NCCL ms | CuBLAS+NCCL TFLOPS/GPU | mKernel ms | mKernel TFLOPS/GPU | mKernel / baseline |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 0.449 | 2.4 | 0.265 | 4.1 | 1.71x |
| 4096 | 0.746 | 11.5 | 0.473 | 18.2 | 1.58x |
| 8192 | 2.685 | 25.6 | 1.733 | 39.7 | 1.55x |
| 16384 | 10.905 | 50.4 | 8.936 | 61.5 | 1.22x |
| 32768 | 59.000 | 74.5 | 65.309 | 67.3 | 0.90x |

The completed `gemm_ar` sweep passed correctness for all measured shapes. mKernel was faster than the CuBLAS+NCCL baseline through `M=N=16384`; at `M=N=32768`, the baseline was faster in this run.

## CUDA 13 Compile Fixes

Two source-level fixes were needed to build the repository in this CUDA 13 environment:

- `ForwardNotify::local_offset` was widened from 32-bit to 64-bit. The forwarding notification path stores offsets derived from `TransferCmd`, whose local and remote offsets are 64-bit byte offsets. CUDA 13 rejects the old aggregate initialization as a narrowing conversion when `cmd.remote_offset` is assigned into the 32-bit notification field.
- A stale `reserved2` copy was removed from the proxy-side notification snapshot. After widening `local_offset`, the padding field no longer exists, while `sizeof(ForwardNotify)` remains fixed at 32 bytes.

These two changes are compile and structure-layout synchronization fixes. They do not change the intended forwarding semantics.

## Runtime Finding

The remaining homepage kernels were blocked by GPU RDMA memory registration in this environment. Kernels that require the VMM DMA-BUF path failed during `ibv_reg_dmabuf_mr` with `EINVAL`; the fallback `ibv_reg_mr` path then failed for the VMM pointer with a bad-address style failure. Retrying DMA-BUF registration without `IBV_ACCESS_RELAXED_ORDERING` did not resolve the issue, so the observed blocker is not only relaxed-ordering flag compatibility.

Validated state:

- `gemm_ar`: completed and correctness passed.
- `ag_gemm`, `gemm_rs`, `dispatch_gemm`: blocked by VMM DMA-BUF MR registration.
- `ring_attention`: not treated as a valid completed curve because it uses the same blocked class of zero-copy/VMM registration path.

## Takeaways

- The H20 two-node environment is sufficient to build and validate the `gemm_ar` path after the CUDA 13 structure fixes.
- The VMM DMA-BUF MR failure is an environment/driver/OFED compatibility blocker for the remaining zero-copy mKernel paths, not a plotting or benchmark harness issue.
- Future reruns should first resolve DMA-BUF MR registration for VMM-exported allocations or add a clearly separated non-VMM fallback path, because a staging fallback would change the benchmark semantics.
