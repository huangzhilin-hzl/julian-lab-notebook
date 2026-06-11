# MiniMax-AI/MSA Modal Benchmark

This directory contains a Modal runner for the benchmark commands documented by
[MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA).

MSA targets NVIDIA SM100. Use a Modal B200 or equivalent SM100 GPU. H100, H20,
and other SM90/older GPUs are expected to fail the runtime capability check.

## Files

- `modal_msa_benchmark.py`: clones MiniMax-AI/MSA remotely, installs it, runs
  `benchmarks/bench_sparse_attention_ops.py`, and writes TSV/JSON/Markdown
  artifacts to a Modal Volume.

## Prerequisites

Install and authenticate the Modal CLI if it is not already available:

```bash
python -m pip install modal
modal setup
```

## Quick Start

```bash
modal run msa-modal/modal_msa_benchmark.py --task smoke
```

The smoke task runs the upstream smoke test:

```bash
python tests/smoke/test_sparse_topk_forced.py
```

Run the README quick benchmark:

```bash
modal run msa-modal/modal_msa_benchmark.py --preset smoke
```

Run the README benchmark presets:

```bash
modal run msa-modal/modal_msa_benchmark.py --preset fp8
modal run msa-modal/modal_msa_benchmark.py --preset bf16
modal run msa-modal/modal_msa_benchmark.py --preset nvfp4
```

Run all README benchmark presets in one Modal call:

```bash
modal run msa-modal/modal_msa_benchmark.py --preset readme
```

Full sweeps can be slow because the first run JIT-compiles SM100 kernels. The
default output path is:

```text
/mnt/msa-cache/msa/results/<run_id>/
```

Each run writes:

- `<preset>.tsv`: upstream benchmark TSV output
- `<preset>.log`: full console log
- `summary.json`: provenance, commands, and parsed rows
- `results.json`: parsed benchmark rows only
- `summary.md`: compact Markdown report

## README Alignment

| MSA README goal | Modal preset | Upstream command shape |
|---|---|---|
| FP8 full sweep | `--preset fp8` | `python benchmarks/bench_sparse_attention_ops.py --dtype fp8 --sections all --output_mode o -o ...` |
| BF16 full sweep | `--preset bf16` | `python benchmarks/bench_sparse_attention_ops.py --dtype bf16 --sections all --output_mode o -o ...` |
| NVFP4 sparse prefill | `--preset nvfp4` | `python benchmarks/bench_sparse_attention_ops.py --dtype nvfp4 --sections sparse_prefill --output_mode o -o ...` |
| Quick CI smoke | `--preset smoke` | `python benchmarks/bench_sparse_attention_ops.py --dtype fp8 --sections prefill,decode,sparse_decode --seqs 8192,16384 --tp 1,4 --decode-k 8192,131072 --decode-b 32 --dry-run-ms 50 --repeat-ms 200 -o ...` |

## Custom Benchmark

The custom preset forwards the same major flags as the upstream benchmark:

```bash
modal run msa-modal/modal_msa_benchmark.py \
  --preset custom \
  --dtype fp8 \
  --sections prefill,paged_prefill,sparse_prefill \
  --seqs 8192,16384 \
  --tp 1,4 \
  --decode-k 8192,131072 \
  --decode-b 32 \
  --dry-run-ms 50 \
  --repeat-ms 200
```

For NVFP4, keep `--sections sparse_prefill`; the upstream benchmark only
supports NVFP4 for sparse prefill.

## Source And Runtime Options

The Modal image clones upstream MSA by default:

```text
https://github.com/MiniMax-AI/MSA.git
```

Override source or GPU at run time:

```bash
modal run msa-modal/modal_msa_benchmark.py \
  --msa-ref main \
  --modal-gpu B200 \
  --preset smoke
```

Environment overrides:

```bash
MSA_REPO_URL=https://github.com/MiniMax-AI/MSA.git
MSA_REF=main
MODAL_GPU=B200
MSA_BASE_IMAGE=nvcr.io/nvidia/pytorch:26.04-py3
MODAL_CACHE_VOLUME_NAME=msa-cache
```

The runner records MSA commit, Modal GPU request, visible GPU model, compute
capability, Torch/CUDA version, and `nvcc --version` in `summary.json`.
