# FlashMLA DSV4 Workload Benchmark on H20

## Run Info

- GPU: `NVIDIA H20`
- Raw CSV: [`flash_mla_workload_perf.csv`](./flash_mla_workload_perf.csv)
- Run log: [`run.log`](./run.log)

Command:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark/bench_flash_mla_workloads.py \
  --output flash_mla_workload_perf.csv
```

Default matrix:

- Workloads: `dsv4-pro`, `dsv4-flash`
- Scenarios: `swa`, `c4`, `c128`
- Batch sizes: `1`, `32`, `128`
- Target sequence lengths: `4096`, `32768`
- `s_q=1`, `runs=20`, `warmup=3`
- Varlen sampling enabled

All 36 cases finished with `correct=1`.

## Key Results

The most useful high-throughput comparison is `batch=128, seq_len_target=32768`:

| workload | h_q | scenario | time_us | TFLOPS | GB/s |
|---|---:|---|---:|---:|---:|
| dsv4-pro | 128 | swa | 66.5 | 63.1 | 645 |
| dsv4-pro | 128 | c4 | 326.4 | 111.7 | 352 |
| dsv4-pro | 128 | c128 | 400.2 | 31.1 | 153 |
| dsv4-flash | 64 | swa | 39.3 | 53.4 | 665 |
| dsv4-flash | 64 | c4 | 122.1 | 85.2 | 517 |
| dsv4-flash | 64 | c128 | 101.2 | 65.4 | 457 |

Main observations:

- `swa` is bandwidth-heavy and scales well at large batch: Pro reaches `645 GB/s`, DSV4 Flash reaches `665 GB/s`.
- `c4` adds a much larger extra sparse set. Pro `c4` hits the highest reported compute rate in this run (`111.7 TFLOPS`) but with much higher latency than SWA.
- `c128` behavior differs by workload: Pro `c128` is the slowest large-batch case (`400.2 us`), while DSV4 Flash `c128` is faster than DSV4 Flash `c4` at `batch=128, seq_len=32768`.
- Batch-1 numbers have fixed overhead noise and should not be used alone for throughput conclusions.

## Time Matrix

`time_us` for each workload/scenario:

| workload | scenario | b1/4k | b1/32k | b32/4k | b32/32k | b128/4k | b128/32k |
|---|---|---:|---:|---:|---:|---:|---:|
| dsv4-pro | swa | 18.2 | 18.3 | 22.0 | 22.1 | 66.5 | 66.5 |
| dsv4-pro | c4 | 22.4 | 22.4 | 98.6 | 104.8 | 299.8 | 326.4 |
| dsv4-pro | c128 | 18.6 | 140.9 | 27.2 | 61.6 | 86.9 | 400.2 |
| dsv4-flash | swa | 137.2 | 19.0 | 22.1 | 22.2 | 39.1 | 39.3 |
| dsv4-flash | c4 | 21.6 | 18.5 | 46.9 | 47.5 | 245.0 | 122.1 |
| dsv4-flash | c128 | 21.5 | 21.0 | 27.1 | 42.4 | 49.1 | 101.2 |

## Caveats

- This is a FlashMLA microbenchmark, not an end-to-end SGLang server benchmark.
- The `c4` and `c128` indices in the script are synthetic. They exercise the extra-k-cache path but do not exactly reproduce SGLang's real C4 indexer distribution or C128 page-table generation.
- The script was run without `--kineto`, so `splitkv_us` and `combine_us` are `0.0` placeholders in the CSV.
