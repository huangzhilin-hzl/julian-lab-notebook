# Prime Flash MoE benchmark

Modal B200 benchmark results for
[`PrimeIntellect-ai/prime-flash-moe`](https://github.com/PrimeIntellect-ai/prime-flash-moe).

## Reproduce

```bash
env -u http_proxy -u https_proxy \
  uvx modal run prime-flash-moe/modal_benchmark.py
```

The runner pins upstream commit `1820183d63eed79fd166fbf4b81cae2b27b326c2`,
builds the CUDA extension for SM100, runs `benchmark/benchmark.py` with its
default benchmark parameters, and downloads the generated artifacts.

## Results

| Run | GPU | Result |
| --- | --- | --- |
| 2026-08-26 16:11 CST | NVIDIA B200 | [20260826-081142](results/20260826-081142/README.md) |
