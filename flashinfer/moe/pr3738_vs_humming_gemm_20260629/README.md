# FlashInfer PR #3738 vs Humming GEMM Benchmark

This directory contains a reusable GEMM-scope benchmark driver for comparing FlashInfer PR #3738 against upstream Humming indexed grouped GEMM. It is intentionally separate from cluster deployment details.

## Report

- [gemm_latency_report.html](gemm_latency_report.html): sanitized static report with benchmark configuration, detailed latency table, and latency charts.

## Scope

| Item | Value |
| --- | --- |
| Comparison | FlashInfer PR #3738 `run_gemm_profile` GEMM1/GEMM2 path vs Humming `HummingLayer` indexed grouped GEMM path |
| Measurement scope | GEMM only; not end-to-end MoE latency |
| Timing method | CUDA events for the primary sweep |
| Default timing | warmup `20`, repeat `100`, tactic warmup `3`, tactic repeat `8` |
| Workloads | DSv4 Flash and DSv4 Pro synthetic GEMM shapes |
| Topologies | TP/EP = `1:8`, `2:4`, `4:2`, `8:1` |
| Batch sizes per rank | `8`, `16`, `32`, `64`, `128`, `256`, `512`, `1024`, `2048`, `4096`, `8192` |

## Tested Environment

| Item | Value |
| --- | --- |
| GPU | 8 x NVIDIA H20, SM90 |
| CUDA / PyTorch | CUDA 12.8 / PyTorch 2.9.1+cu128 for the recorded run |
| Alternative image checked | `docker.1ms.run/lmsysorg/sglang:dev-cu13`, digest `sha256:68ccccd7e71208fd11953d71ec25bdd98c613a544449611d9b94b424d2375bf5` |
| Profiler tools in checked image | `ncu` from `cuda-nsight-compute-13-0`; `nsys` from `nsight-systems-cli` |

Cluster deployment identifiers, internal image URL, host paths, and local absolute paths are intentionally not recorded in this artifact.

## Tested Commits

| Component | Commit |
| --- | --- |
| FlashInfer PR #3738 | [`96397d18b5801c4bae10931bb258de4a25c64270`](https://github.com/flashinfer-ai/flashinfer/commit/96397d18b5801c4bae10931bb258de4a25c64270) |
| Humming baseline | [`f6241bba8d507c19ca9ce4e5958a5d0641fc8eb4`](https://github.com/huangzhilin-hzl/humming/commit/f6241bba8d507c19ca9ce4e5958a5d0641fc8eb4) |

## Install

Use clean source checkouts for both projects. Keep the concrete checkout paths outside committed logs and reports.

```bash
python3 -m pip install --no-build-isolation --no-deps -e <flashinfer-repo> -v
SETUPTOOLS_SCM_PRETEND_VERSION=0.1.5 \
  python3 -m pip install --no-build-isolation --no-deps -e <humming-repo> -v
```

If Humming's NVRTC include discovery misses CUDA target headers, set:

```bash
export CUDA_TARGET_INCLUDE_PATH="${CUDA_HOME}/targets/x86_64-linux/include"
```

## Run

```bash
python3 scripts/bench_pr3738_vs_humming_gemm.py \
  --output-dir <output-dir> \
  --flashinfer-repo <flashinfer-repo> \
  --humming-repo <humming-repo>
```

The script writes:

| File | Content |
| --- | --- |
| `raw_rows.csv/json` | Per-backend, per-segment timing rows for w13 and w2 |
| `summary_rows.csv/json` | FlashInfer vs Humming summary with w13, w2, total GEMM, and speedup |
| `environment.json` | Non-sensitive runtime metadata: GPU model/capability, Torch/CUDA versions, driver version, and repo commits |
| `run.log` | Sweep progress and non-sensitive metadata |
| `pr3738_vs_humming_gemm.png` | Summary plot when matplotlib is available |

## NCU Spot Check

Use NCU only to validate kernel scope. Do not merge NCU output into the primary CUDA-event timing table.

```bash
ncu --target-processes all --set full \
  -o <ncu-output-prefix> \
  python3 scripts/bench_pr3738_vs_humming_gemm.py \
    --mode ncu \
    --output-dir <output-dir> \
    --flashinfer-repo <flashinfer-repo> \
    --humming-repo <humming-repo> \
    --workloads dsv4-flash \
    --topologies 8:1 \
    --batches 8192
```

Repeat with `--workloads dsv4-pro` for the DSv4 Pro spot check.

## Script Contract

- The script does not accept cluster deployment metadata as CLI parameters.
- It does not log cluster deployment identifiers, image, install-command, repo path, or output path values.
- Commit IDs are derived from the supplied git checkouts.
- `FlashInfer total GEMM` means `w13_grouped_gemm_ms + w2_grouped_gemm_ms`.
- `speedup` means `humming_gemm_total_ms / flashinfer_gemm_total_ms`.
