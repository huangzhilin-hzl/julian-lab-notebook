# mKernel

This directory stores sanitized mKernel benchmark notes and result tables.

## Documents

| File | Content |
| --- | --- |
| [h20_two_node_gemm_ar_20260526.md](h20_two_node_gemm_ar_20260526.md) | Sanitized two-node H20 mKernel benchmark note for the completed `gemm_ar` sweep and the CUDA 13 compile/runtime findings. |
| [h20_two_node_release_cumemhost0_20260601.md](h20_two_node_release_cumemhost0_20260601.md) | Sanitized two-node H20 release-shape comparison with CuBLAS+NCCL using `NCCL_CUMEM_HOST_ENABLE=0`. |
| [images/gemm_ar_h20_n2_homepage.svg](images/gemm_ar_h20_n2_homepage.svg) | Homepage-style chart for the two-node H20 `gemm_ar` sweep. |
| [images/h20_roce_cumemhost0_20260601/](images/h20_roce_cumemhost0_20260601/) | Official-style per-kernel SVG comparison charts for the two-node H20 RoCE release-shape run. |
| [images/h20_roce_peermem_20260602/](images/h20_roce_peermem_20260602/) | Official-style per-kernel SVG comparison charts for the two-node H20 RoCE peermem-compatible rerun. |
| [results/h20_two_node_gemm_ar_20260526.csv](results/h20_two_node_gemm_ar_20260526.csv) | Structured result table for mKernel `gemm_ar` vs CuBLAS+NCCL baseline. |
| [results/h20_two_node_release_cumemhost0_20260601.csv](results/h20_two_node_release_cumemhost0_20260601.csv) | Structured result table for the two-node H20 release-shape comparison. |
