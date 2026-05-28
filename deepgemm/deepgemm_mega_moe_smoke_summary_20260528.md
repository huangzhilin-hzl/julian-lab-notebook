# DeepGEMM Mega MoE Smoke Summary

Run date: 2026-05-28

## Command

```bash
python3 -m modal run deepgemm/modal_deepgemm_mega_moe.py --task smoke
```

## Environment

- Modal app: `deepgemm-mega-moe-bench`
- Modal run: https://modal.com/apps/huangzhilin-hzl/main/ap-muBcUu5MlqImIEPTkdQOQ2
- DeepGEMM source commit: `714dd1a4a980`
- Base image: `nvcr.io/nvidia/pytorch:26.04-py3`
- PyTorch: `2.12.0a0+0291f960b6.nv26.04.48445190`
- CUDA reported by PyTorch: `13.2`
- Visible GPU count: `8`
- GPU model: `NVIDIA B200`
- DeepGEMM package version: `2.5.0`

## Smoke Case

- Model config: DeepSeek-V4-Flash
- Tokens per rank: `1`
- Total tokens: `1`
- Hidden size: `4096`
- Intermediate hidden size: `2048`
- Experts: `256`
- Top-k experts: `6`
- Processes / ranks: `8`
- Per-rank buffer: `0.233 GiB`
- Mode: `--ncu-profile-only`
- Correctness tests: `0`

## Result

The smoke task completed successfully. All 8 ranks entered the fused Mega MoE
kernel path and exited cleanly with `Done, exiting`.

The run validates that the Modal image can:

- clone DeepGEMM and its submodules,
- build the `deep_gemm._C` extension,
- allocate an 8-card B200 worker,
- import PyTorch, `torch.distributed._symmetric_memory`, and `deep_gemm`,
- launch `tests/test_mega_moe.py` across 8 local ranks.

## Data Conclusion

This run is a functional smoke test, not a performance sweep. Because the smoke
entrypoint sets `ncu_profile_only=True`, the benchmark does not emit parsed
performance rows such as TFLOPS, latency, HBM bandwidth, NVLink bandwidth, or
legacy speedup. Therefore there is no numeric performance table to summarize
from this run.

The actionable conclusion is that the DeepGEMM Mega MoE Modal pipeline is ready
for a full table run, for example:

```bash
python3 -m modal run deepgemm/modal_deepgemm_mega_moe.py \
  --task table \
  --model both \
  --gpu B200:8 \
  --deepgemm-ref main \
  --batch-sizes 1,512,8192,32768
```

## Artifacts

- Local log copy: `deepgemm/results/20260528-041135/flash_bsz1.log`
- Modal Volume log: `/cache/deepgemm/results/20260528-041135/flash_bsz1.log`
