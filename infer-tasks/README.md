# infer-tasks

Harbor benchmark tasks for inference. See [HARBOR_NOTES.md](HARBOR_NOTES.md) for the task format.

- `sglang-bench-template`: minimal two-step template. Step `serve` launches sglang; step `bench` runs one `bench_serving` pass per concurrency in `BENCH_CONC_LIST` against it. The bench verifier checks that the result JSON is complete and applies no performance thresholds.
- `dsv41-flash-h200-fp4-lowlatency`: DeepSeek-V4.1 Flash from the sglang cookbook (H200, FP4, low-latency, single node), humming-off with DSPARK and a pinned dev image; with InferenceX's fixed-seq-len load (8k/1k, concurrency 1-16).

Each new task goes in its own subdirectory. `jobs/` holds Harbor job configs for running the tasks on Modal (HARBOR_NOTES.md section 9).

- `dsv41-flash-h200-fp4-lowlatency-humming`: paired humming-on arm with the same image, DSPARK and load.

Paired job and comparison protocol: [dsv41-humming-ab](../infer-experiments/dsv41-humming-ab/README.md).
