# TokenSpeed Modal

## Files

| Relative Path | Content |
| --- | --- |
| `modal_tokenspeed_test.py` | Modal app for uploading a local TokenSpeed checkout, building it from the `lightseekorg/tokenspeed-runner:latest` image, running `tokenspeed env` smoke checks, running TokenSpeed kernel pytest suites, reproducing the Kimi K2.5 agentic benchmark sweep, running TokenSpeed MLA standalone benchmark cases, and optionally launching a server plus `tokenspeed bench serve`. |

## Local Setup

```bash
git clone https://github.com/lightseekorg/tokenspeed.git /path/to/tokenspeed
pip install -U modal
modal setup
modal volume create --version=2 tokenspeed-cache
```

If the target model requires Hugging Face credentials:

```bash
modal secret create hf-secret HF_TOKEN=<your-huggingface-token>
```

## Smoke Test

Set `TOKENSPEED_SRC_DIR` to your local TokenSpeed checkout.

```bash
cd /path/to/julian-lab-notebook

TOKENSPEED_SRC_DIR=/path/to/tokenspeed \
MODAL_GPU=B200 \
modal run tokenspeed-modal/modal_tokenspeed_test.py --task smoke
```

## Kernel Tests

Run the same kernel pytest sequence declared by TokenSpeed CI:

```bash
cd /path/to/julian-lab-notebook

TOKENSPEED_SRC_DIR=/path/to/tokenspeed \
MODAL_GPU=B200 \
modal run tokenspeed-modal/modal_tokenspeed_test.py \
  --task kernel \
  --kernel-mode ci
```

Faster focused runs:

```bash
# Native CUDA third-party kernel tests only.
TOKENSPEED_SRC_DIR=/path/to/tokenspeed \
MODAL_GPU=B200 \
modal run tokenspeed-modal/modal_tokenspeed_test.py \
  --task kernel \
  --kernel-mode cuda

# Numerics verification tests only.
TOKENSPEED_SRC_DIR=/path/to/tokenspeed \
MODAL_GPU=B200 \
modal run tokenspeed-modal/modal_tokenspeed_test.py \
  --task kernel \
  --kernel-mode numerics

# One specific pytest target.
TOKENSPEED_SRC_DIR=/path/to/tokenspeed \
MODAL_GPU=B200 \
modal run tokenspeed-modal/modal_tokenspeed_test.py \
  --task kernel \
  --kernel-target tokenspeed-kernel/test/ops/test_tokenspeed_mla.py
```

Optional pytest flags can be passed as a JSON string:

```bash
modal run tokenspeed-modal/modal_tokenspeed_test.py \
  --task kernel \
  --kernel-target tokenspeed-kernel/test/thirdparty/test_cuda.py \
  --kernel-keyword rope \
  --extra-pytest-args-json '["-s"]'
```

## Blog Reproduction: Kimi K2.5 On B200

This runs TokenSpeed's checked-in SWE-Smith agentic benchmark sweep:

```bash
cd /path/to/julian-lab-notebook

TOKENSPEED_SRC_DIR=/path/to/tokenspeed \
MODAL_GPU=B200:8 \
MODAL_HF_SECRET_NAME=hf-secret \
modal run tokenspeed-modal/modal_tokenspeed_test.py --task agentic
```

Use `B200:8` for the full checked-in sweep because the matrix includes TP8/DP8
layouts. For a quick TP4-only validation, edit
`/path/to/tokenspeed/test/agentic_benchmark/tokenspeed/agentic_bench.sh`
and keep only `attn_tp4_moe_tp4` / `attn_tp4_moe_ep4`, then run with
`MODAL_GPU=B200:4`.

Artifacts are copied to `/cache/results/agentic/<run_id>` in the
`tokenspeed-cache` Modal Volume, including `sweep.csv` and server logs.

## Blog Reproduction: TokenSpeed MLA On B200

Run the standalone MLA prefill/decode cases from `tokenspeed-mla/README.md`:

```bash
cd /path/to/julian-lab-notebook

TOKENSPEED_SRC_DIR=/path/to/tokenspeed \
MODAL_GPU=B200 \
modal run tokenspeed-modal/modal_tokenspeed_test.py --task mla
```

Focused runs:

```bash
# Prefill cases only, using the open-source CuTe DSL backend.
TOKENSPEED_SRC_DIR=/path/to/tokenspeed \
MODAL_GPU=B200 \
modal run tokenspeed-modal/modal_tokenspeed_test.py \
  --task mla \
  --mla-mode prefill

# Decode cases only.
TOKENSPEED_SRC_DIR=/path/to/tokenspeed \
MODAL_GPU=B200 \
modal run tokenspeed-modal/modal_tokenspeed_test.py \
  --task mla \
  --mla-mode decode

# Binary prefill path. This requires the compatible AOT .so to exist in the
# package or TOKENSPEED_MLA_FMHA_BINARY_SO to point to it.
TOKENSPEED_SRC_DIR=/path/to/tokenspeed \
MODAL_GPU=B200 \
modal run tokenspeed-modal/modal_tokenspeed_test.py \
  --task mla \
  --mla-mode prefill \
  --mla-prefill-backend binary
```

MLA logs are written to `/cache/results/mla/<run_id>`.

## Serving Benchmark

```bash
cd /path/to/julian-lab-notebook

TOKENSPEED_SRC_DIR=/path/to/tokenspeed \
MODAL_GPU=B200 \
MODAL_HF_SECRET_NAME=hf-secret \
modal run tokenspeed-modal/modal_tokenspeed_test.py \
  --task bench \
  --model openai/gpt-oss-20b \
  --served-model-name gpt-oss-20b \
  --num-prompts 32 \
  --input-len 1024 \
  --output-len 128
```

Results are written under `/cache/results` in the `tokenspeed-cache` Modal
Volume.

## Useful Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `TOKENSPEED_SRC_DIR` | required | Local TokenSpeed checkout uploaded into the Modal image. |
| `MODAL_GPU` | `B200` | GPU resource used by the Modal functions, for example `B200`, `H100!`, or `H200`. |
| `MODAL_HF_SECRET_NAME` | unset | Modal Secret name that exposes `HF_TOKEN` in the remote container. |
| `MODAL_CACHE_VOLUME_NAME` | `tokenspeed-cache` | Modal Volume mounted at `/cache` for Hugging Face cache and benchmark outputs. |
| `MODAL_CACHE_VOLUME_VERSION` | `2` | Modal VolumeFS version used if the cache Volume is created lazily by the script. |
| `FLASHINFER_CUDA_ARCH_LIST` | `9.0a 10.0a` | CUDA arch list used when building TokenSpeed kernels. |
| `MAX_JOBS` | `16` | Parallelism for native kernel compilation. |
| `TOKENSPEED_MLA_FMHA_BINARY_SO` | unset | Remote path to a compatible TokenSpeed MLA binary prefill `.so`, required for exact binary-prefill reproduction. |

## Kernel Test Modes

| Mode | Pytest Scope |
| --- | --- |
| `ci` | TokenSpeed CI sequence for `test_numerics.py`, `test_trtllm_comm.py`, `test_cuda.py`, then the rest of `tokenspeed-kernel/test/`. |
| `cuda` | `tokenspeed-kernel/test/thirdparty/test_cuda.py`. |
| `numerics` | `tokenspeed-kernel/test/test_numerics.py`. |
| `ops` | `tokenspeed-kernel/test/ops/`. |
| `all` | `tokenspeed-kernel/test/` in one pytest invocation. |
