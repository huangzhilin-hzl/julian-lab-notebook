# infer-tasks

Harbor benchmark tasks for inference. See [HARBOR_NOTES.md](HARBOR_NOTES.md) for the task format.

- [`sglang-bench-template`](sglang-bench-template/): minimal template that launches sglang and runs one `bench_serving` pass; the verifier checks that the result JSON is complete and applies no performance thresholds.
- [`dsv41-flash-h200-fp4-lowlatency`](dsv41-flash-h200-fp4-lowlatency/): DeepSeek-V4.1 Flash from the sglang cookbook (H200, FP4, low-latency, single node).

Each new task goes in its own subdirectory.
