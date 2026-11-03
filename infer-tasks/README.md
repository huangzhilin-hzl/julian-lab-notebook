# Inference benchmark tasks

Harbor 0.25.0 two-step tasks: start sglang, then measure a fixed-load concurrency sweep.
Verifiers check health, result completeness and load; reward is 0/1 without performance thresholds.

- [Template](sglang-bench-template/README.md): configurable image, launch command, model, hardware and load placeholders.
- [Humming off](dsv41-flash-h200-fp4-lowlatency/README.md): DeepSeek-V4.1 Flash FP4, 8× H200, `flashinfer_mxfp4`.
- [Humming on](dsv41-flash-h200-fp4-lowlatency-humming/README.md): the same configuration with MoE runner `humming`.

Both concrete tasks pin image index digest `sha256:77ff63ee69fd7ebef20df3a6d036ee4b8a4ffe0734efb0028c3b7643386e3ccf`
and sglang build `38d9e4a3fe4a493b0ae468477a274e83defda81a`. Exact launch commands are in their READMEs
and `task.toml`; load is 8192/1024, ratio 0.8, concurrency 1/2/4/8/16, 10× concurrency prompts,
2× concurrency warmups, seed 42, rate `inf`, ignore EOS and cache flush after warmup.
Measurements include throughput, latency percentiles and per-GPU throughput.

Run `harbor run -c jobs/modal-dsv41.yaml -y` from this directory for the baseline task,
or use [the paired experiment](../infer-experiments/dsv41-humming-ab/README.md).
See [Harbor notes](HARBOR_NOTES.md) for credentials, cache setup and provider constraints.
