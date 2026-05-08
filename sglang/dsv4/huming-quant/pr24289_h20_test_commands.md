# PR24289 H20 DeepSeek-V4 Test Commands

This document records the command templates used for the H20 DeepSeek-V4 recipe comparison.

## Result Interpretation Notes

- CP uses the H20-adjusted DeepEP config: `num_sms=20`, `mem-fraction-static=0.82`, `max-running-requests=48`, `cuda-graph-max-bs=48`, and `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256`.
- This follows the CP batch/memory shape from SGLang PR #24139, but lowers `num_sms` because H20 has 78 SMs and `num_sms=96` triggers DeepEP intranode cooperative launch failure.
- This is not a same-checkpoint backend A/B. `fp8_baseline` uses the converted pure FP8 checkpoint `sgl-project/DeepSeek-V4-Flash-FP8`.
- `marlin_w4a16`, `humming_w4a16`, and `humming_w4a8` use the official `deepseek-ai/DeepSeek-V4-Flash` mixed FP4/FP8 checkpoint. The W4A16/W4A8 labels describe the MoE runner activation path.
- The accuracy column in the summary is the 5-prompt deterministic sanity check used during this run. GSM8K is not included because the attempted command used `--host 127.0.0.1` instead of `--host http://127.0.0.1`, so those attempted results are invalid.

## Common Setup

```bash
export SGLANG_REPO=<path_to_sglang_repo>
export MODEL_FP8=<path_to_sgl-project_DeepSeek-V4-Flash-FP8>
export MODEL_MIXED=<path_to_deepseek-ai_DeepSeek-V4-Flash>
export OUT_DIR=<path_to_benchmark_output_dir>
export PORT=12321
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd "${SGLANG_REPO}"
export PYTHONPATH="${SGLANG_REPO}/python"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## Shared DeepEP Config For H20

The upstream recipe used a larger `num_sms`; for H20 this was lowered to `20` to avoid DeepEP intranode cooperative launch failures.

```bash
export DEEPEP_CONFIG='{"normal_dispatch":{"num_sms":20,"num_max_nvl_chunked_send_tokens":16,"num_max_nvl_chunked_recv_tokens":256,"num_max_rdma_chunked_send_tokens":6,"num_max_rdma_chunked_recv_tokens":128},"normal_combine":{"num_sms":20,"num_max_nvl_chunked_send_tokens":6,"num_max_nvl_chunked_recv_tokens":256,"num_max_rdma_chunked_send_tokens":6,"num_max_rdma_chunked_recv_tokens":128}}'
```

## Variant Environment

```bash
# Pure FP8 baseline.
export SGLANG_DSV4_FP4_EXPERTS=0

# Humming W4A16.
export HUMMING_COMPILER=nvcc
unset SGLANG_HUMMING_INPUT_QUANT_CONFIG

# Humming W4A8.
export HUMMING_COMPILER=nvcc
export SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3"}'
```

## Benchmark Workloads

```bash
# Low-latency
export NUM_PROMPTS=12
export MAX_CONCURRENCY=1
export RANDOM_INPUT_LEN=1024
export RANDOM_OUTPUT_LEN=256

# Balanced
export NUM_PROMPTS=32
export MAX_CONCURRENCY=8
export RANDOM_INPUT_LEN=1024
export RANDOM_OUTPUT_LEN=256

# Max-throughput
export NUM_PROMPTS=64
export MAX_CONCURRENCY=32
export RANDOM_INPUT_LEN=1024
export RANDOM_OUTPUT_LEN=256

# Context-parallel
export NUM_PROMPTS=8
export MAX_CONCURRENCY=2
export RANDOM_INPUT_LEN=4096
export RANDOM_OUTPUT_LEN=128
```

## Benchmark Command

Use the model path and served name that match the running server.

```bash
python3 -m sglang.bench_serving \
  --backend sglang-oai \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --dataset-name random-ids \
  --random-input-len "${RANDOM_INPUT_LEN}" \
  --random-output-len "${RANDOM_OUTPUT_LEN}" \
  --random-range-ratio 0 \
  --num-prompts "${NUM_PROMPTS}" \
  --max-concurrency "${MAX_CONCURRENCY}" \
  --output-file "${OUT_DIR}/bench_details_${RECIPE}_${CONFIG}.jsonl" \
  --output-details \
  --disable-tqdm
```

## Low-Latency Recipe

Pure FP8 baseline:

```bash
export MODEL_PATH="${MODEL_FP8}"
export SERVED_MODEL_NAME=sgl-project/DeepSeek-V4-Flash-FP8
export SGLANG_DSV4_FP4_EXPERTS=0
export SGLANG_ENABLE_SPEC_V2=1

python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --tp-size 4 \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

Marlin W4A16:

```bash
export MODEL_PATH="${MODEL_MIXED}"
export SERVED_MODEL_NAME=deepseek-ai/DeepSeek-V4-Flash
export SGLANG_ENABLE_SPEC_V2=1
unset SGLANG_DSV4_FP4_EXPERTS

python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --tp-size 4 \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --moe-runner-backend marlin \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

Humming W4A16 / W4A8:

```bash
export MODEL_PATH="${MODEL_MIXED}"
export SERVED_MODEL_NAME=deepseek-ai/DeepSeek-V4-Flash
export HUMMING_COMPILER=nvcc
export SGLANG_ENABLE_SPEC_V2=1
# For W4A8 only:
# export SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3"}'

python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --tp-size 4 \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --moe-runner-backend humming \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

## Balanced Recipe

Pure FP8 baseline:

```bash
export MODEL_PATH="${MODEL_FP8}"
export SERVED_MODEL_NAME=sgl-project/DeepSeek-V4-Flash-FP8
export SGLANG_DSV4_FP4_EXPERTS=0
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256

python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --tp-size 4 \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --dp-size 4 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --cuda-graph-max-bs 128 \
  --max-running-requests 128 \
  --deepep-config "${DEEPEP_CONFIG}" \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2
```

Marlin W4A16:

```bash
export MODEL_PATH="${MODEL_MIXED}"
export SERVED_MODEL_NAME=deepseek-ai/DeepSeek-V4-Flash
export SGLANG_ENABLE_SPEC_V2=1

python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --tp-size 4 \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --moe-runner-backend marlin \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2
```

Humming W4A16 / W4A8:

```bash
export MODEL_PATH="${MODEL_MIXED}"
export SERVED_MODEL_NAME=deepseek-ai/DeepSeek-V4-Flash
export HUMMING_COMPILER=nvcc
export SGLANG_ENABLE_SPEC_V2=1
# For W4A8 only:
# export SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3"}'

python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --tp-size 4 \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --moe-runner-backend humming \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2
```

## Max-Throughput Recipe

Pure FP8 baseline:

```bash
export MODEL_PATH="${MODEL_FP8}"
export SERVED_MODEL_NAME=sgl-project/DeepSeek-V4-Flash-FP8
export SGLANG_DSV4_FP4_EXPERTS=0
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256

python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --tp-size 4 \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --dp-size 4 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --cuda-graph-max-bs 128 \
  --max-running-requests 256 \
  --deepep-config "${DEEPEP_CONFIG}"
```

Marlin W4A16:

```bash
export MODEL_PATH="${MODEL_MIXED}"
export SERVED_MODEL_NAME=deepseek-ai/DeepSeek-V4-Flash

python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --tp-size 4 \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --moe-runner-backend marlin
```

Humming W4A16 / W4A8:

```bash
export MODEL_PATH="${MODEL_MIXED}"
export SERVED_MODEL_NAME=deepseek-ai/DeepSeek-V4-Flash
export HUMMING_COMPILER=nvcc
# For W4A8 only:
# export SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3"}'

python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --tp-size 4 \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --moe-runner-backend humming
```

## Context-Parallel Recipe

Marlin W4A16 was skipped for this recipe because the tested CP/DeepEP path did not support it.

Pure FP8 baseline:

```bash
export MODEL_PATH="${MODEL_FP8}"
export SERVED_MODEL_NAME=sgl-project/DeepSeek-V4-Flash-FP8
export SGLANG_DSV4_FP4_EXPERTS=0
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=1
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256

python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --tp-size 4 \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --moe-a2a-backend deepep \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --chunked-prefill-size 16384 \
  --mem-fraction-static 0.82 \
  --max-running-requests 48 \
  --cuda-graph-max-bs 48 \
  --deepep-config "${DEEPEP_CONFIG}"
```

Humming W4A16 / W4A8:

```bash
export MODEL_PATH="${MODEL_MIXED}"
export SERVED_MODEL_NAME=deepseek-ai/DeepSeek-V4-Flash
export HUMMING_COMPILER=nvcc
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=1
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
# For W4A8 only:
# export SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3"}'

python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --tp-size 4 \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --moe-a2a-backend deepep \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --chunked-prefill-size 16384 \
  --mem-fraction-static 0.82 \
  --max-running-requests 48 \
  --cuda-graph-max-bs 48 \
  --deepep-config "${DEEPEP_CONFIG}" \
  --moe-runner-backend humming
```
