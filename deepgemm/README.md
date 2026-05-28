# DeepGEMM Modal Mega MoE

This directory contains a Modal wrapper for reproducing the DeepGEMM PR #316
Mega MoE benchmark sweep.

## Requirements

- Modal CLI login on the local machine.
- Modal access to an 8-GPU SM100 machine. The default request is `B200:8`.
- Network access from Modal to clone the target DeepGEMM repository.

## Quick smoke test

```bash
modal run deepgemm/modal_deepgemm_mega_moe.py --task smoke
```

The smoke task runs the fused Mega MoE kernel for a tiny Flash case and skips
legacy baseline import.

## PR #316 table sweep

```bash
modal run deepgemm/modal_deepgemm_mega_moe.py \
  --task table \
  --model both \
  --gpu B200:8 \
  --deepgemm-ref main \
  --batch-sizes 1,512,8192,32768
```

The wrapper runs these two model configs:

| Model | Experts | Top-k | Hidden | Intermediate |
|---|---:|---:|---:|---:|
| DeepSeek-V4-Flash | 256 | 6 | 4096 | 2048 |
| DeepSeek-V4-Pro | 384 | 6 | 7168 | 3072 |

Each batch size maps to both `--num-tokens` and
`--num-max-tokens-per-rank` in `tests/test_mega_moe.py`.

The printed Markdown table and `summary.csv` include the DeepGEMM source
commit, DeepEP commit when available, GPU model, and visible GPU count.

By default the Modal image clones:

```text
https://github.com/deepseek-ai/DeepGEMM.git
```

Override the source with `--deepgemm-repo` and `--deepgemm-ref`. The ref may be
a branch, tag, or commit SHA.

## Single case

```bash
modal run deepgemm/modal_deepgemm_mega_moe.py \
  --task case \
  --model pro \
  --batch-size 8192 \
  --gpu B200:8
```

## Legacy speedup column

The fused kernel can run without the legacy baseline. In that mode the
`Speedup (vs legacy)` column is not meaningful, and the DeepEP commit column
will show `not-installed`.

To require the baseline, build the Modal image with:

```bash
DEEPGEMM_INSTALL_BASELINE=1 \
modal run deepgemm/modal_deepgemm_mega_moe.py \
  --task case \
  --model flash \
  --batch-size 512 \
  --require-baseline
```

This attempts to install `tilelang` and DeepEP during the Modal image build.
If either package fails to build in the selected base image, keep
`DEEPGEMM_INSTALL_BASELINE=0` and use the fused-kernel timing columns only.

## Outputs

Logs and summary files are written to:

```text
/cache/deepgemm/results/<run_id>/
```

inside the Modal Volume named by `MODAL_CACHE_VOLUME_NAME`, defaulting to
`deepgemm-cache`. Each run writes:

- one log file per model and batch size
- `summary.csv`
- `summary.json`

The wrapper patches `deep_gemm.utils.dist.dist_print` in the remote copy so all
local ranks print performance lines. The reported table is averaged across
those rank rows. The DeepGEMM commit is captured from the remote clone inside
the Modal image. The DeepEP commit is read from Python package metadata or the
installed package source tree when baseline dependencies are installed.

## Optional knobs

```bash
MODAL_GPU=B200:8
DEEPGEMM_REPO_URL=https://github.com/deepseek-ai/DeepGEMM.git
DEEPGEMM_REF=main
DEEPGEMM_BASE_IMAGE=nvcr.io/nvidia/pytorch:26.04-py3
MODAL_CACHE_VOLUME_NAME=deepgemm-cache
MAX_JOBS=16
```

`DEEPGEMM_BASE_IMAGE` should provide CUDA 12.9+ and a PyTorch build with
`torch.distributed._symmetric_memory`.
