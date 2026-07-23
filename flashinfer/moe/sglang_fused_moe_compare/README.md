# Fused MoE single-backend benchmark

这个脚本一次只运行：

```text
一个 workload JSON × 一个 backend
```

它不遍历 workload 列表，不自动启动另一个 backend，也不聚合或计算 speedup。调用者逐次选择 `humming`、`flashinfer`、`sglang_cutlass` 或 `sglang_marlin`。

测试范围是单个 local-rank fused MoE，不包含 router/top-k、A2A、all-reduce 或模型其他层。模型 shape 和量化参数全部来自 workload。

## 逐个运行 backend

Humming：

```bash
python flashinfer/moe/sglang_fused_moe_compare/bench_fused_moe.py \
  --workload flashinfer/moe/sglang_fused_moe_compare/workload.wint4.example.json \
  --backend humming \
  --output results/wint4.humming.json
```

FlashInfer：

```bash
python flashinfer/moe/sglang_fused_moe_compare/bench_fused_moe.py \
  --workload flashinfer/moe/sglang_fused_moe_compare/workload.wint4.example.json \
  --backend flashinfer \
  --output results/wint4.flashinfer.json
```

SGLang 原生 CUTLASS W4A8（同一份 `wint4_afp8` group-128 workload）：

```bash
python flashinfer/moe/sglang_fused_moe_compare/bench_fused_moe.py \
  --workload flashinfer/moe/sglang_fused_moe_compare/workload.wint4.example.json \
  --backend sglang_cutlass \
  --output results/wint4.sglang-cutlass.json
```

SGLang Marlin W4A16：

```bash
python flashinfer/moe/sglang_fused_moe_compare/bench_fused_moe.py \
  --workload flashinfer/moe/sglang_fused_moe_compare/workload.wint4-a16.example.json \
  --backend sglang_marlin \
  --output results/wint4-a16.sglang-marlin.json
```

比较相同 quant contract 时，各次调用应使用同一 workload、seed、`num_tokens` 和 routing。调用者可按相同 `M` 读取 JSON 的 `rows[].gpu_p50_ms`。例如 Humming 和 FlashInfer：

```text
humming_speedup = flashinfer_gpu_p50_ms / humming_gpu_p50_ms
```

大于 1 表示 Humming 更快。

脚本直接使用当前 Python 环境中的 `sglang`、`flashinfer` 和 `humming`。使用源码 checkout 时，通过 editable install 或统一设置 `PYTHONPATH`；脚本没有任何 `--sglang-root`、`--humming-root` 或 `--flashinfer-root` 参数。

## 输入和输出

`--workload` 只接受一个 JSON object，不接受数组或 JSONL。提供了三个 smoke
示例：

- `workload.mxfp4.example.json`
- `workload.wint4.example.json`
- `workload.wint4-a16.example.json`

以及八个模型 workload：

- `workload.dsv4-pro.mxfp4-fp8.json`
- `workload.dsv4-flash.mxfp4-fp8.json`
- `workload.dsv4-pro.wint4-a16-g128.json`
- `workload.dsv4-flash.wint4-a16-g128.json`
- `workload.kimi-k2.6.wint4-afp8.json`
- `workload.kimi-k2.6.wint4-afp8-g128.json`
- `workload.kimi-k2.6.wint4-a16-g128.json`
- `workload.glm-5.2-w4afp8.json`

只验证 workload，不加载 CUDA：

```bash
python flashinfer/moe/sglang_fused_moe_compare/bench_fused_moe.py \
  --workload flashinfer/moe/sglang_fused_moe_compare/workload.mxfp4.example.json \
  --backend humming \
  --dry-run
```

每次调用只产生一个 `--output` JSON，包含：

- `workload`：规范化后的 shape 和量化参数；
- `backend` 与 `environment`：当前 backend、GPU、CUDA 和包版本；
- `run_args`：warmup、repeat、设备和 autotune 配置；
- `rows`：每个 `num_tokens=M` 的 latency、吞吐、路由统计和输出 probe。

## 模型 workload 口径

模型结构字段来自对应仓库的配置；TP 是公开部署配置，`num_tokens` 是 benchmark
sweep，不是模型结构。这些 workload 都使用 `ep_size=1`，因此模型 global top-k
就是当前脚本的 local top-k；`num_experts` 只包含 routed experts，不包含单独执行的
shared expert。

| workload | K | routed N | TP | local N | routed E | top-k | quant contract |
|---|---:|---:|---:|---:|---:|---:|---|
| [DSV4 Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json) | 7168 | 3072 | 8 | 384 | 384 | 6 | MXFP4 E2M1 / E8M0-g32 × FP8 E4M3 |
| DSV4 Pro shape variant | 7168 | 3072 | 8 | 384 | 384 | 6 | signed INT4/BF16-g128 × BF16 (Marlin W4A16) |
| [DSV4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json) | 4096 | 2048 | 4 | 512 | 256 | 6 | MXFP4 E2M1 / E8M0-g32 × FP8 E4M3 |
| DSV4 Flash shape variant | 4096 | 2048 | 4 | 512 | 256 | 6 | signed INT4/BF16-g128 × BF16 (Marlin W4A16) |
| [Kimi K2.6](https://huggingface.co/moonshotai/Kimi-K2.6/blob/main/config.json) | 7168 | 2048 | 8 | 256 | 384 | 8 | INT4/BF16-g32 × FP8 E4M3 |
| Kimi K2.6 shape variant | 7168 | 2048 | 8 | 256 | 384 | 8 | signed INT4/BF16-g128 × FP8 E4M3 (CUTLASS W4A8) |
| Kimi K2.6 shape variant | 7168 | 2048 | 8 | 256 | 384 | 8 | signed INT4/BF16-g128 × BF16 (Marlin W4A16) |
| [GLM-5.2-W4AFP8](https://huggingface.co/PhalaCloud/GLM-5.2-W4AFP8/blob/main/config.json) | 6144 | 2048 | 8 | 256 | 256 | 8 | INT4/BF16-g128 × FP8 E4M3 |

所有 workload 都复用对应模型的 problem shape；原始 DSV4、Kimi 与 GLM workload
还复用目标量化 layout，而标为 shape variant 的 group-128 workload 只复用 shape。
权重、scale 与输入均由脚本合成；`weight_scale_amplitude`、`fc*_input_scale` 和
`fc*_prequant_scale` 是可复现的 benchmark 参数，不是从 checkpoint 抽取的实际数值。

DSV4 的 TP4/TP8 分别跟随 [SGLang Day-0](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
公开的 Flash/Pro 部署形状。其 expert FP4 scale 是 K 方向 group-32 E8M0；
模型顶层 FP8 配置中的 `weight_block_size=[128,128]` 不是 FP4 expert 的
group size。

Kimi K2.6 的原始 workload 使用官方 checkpoint 的 symmetric INT4 group-32
weight layout，并为 fused-MoE benchmark 增加运行时 FP8 E4M3 activation。当前
Humming 路径支持这个 group size；FlashInfer 和 SGLang CUTLASS 的 SM90
packed-INT4 fused-MoE kernel 固定为 group-128，因此不能用原始 workload 运行
这两个 backend。新增的两份 Kimi group-128 workload 只复用 Kimi problem shape，
分别覆盖 CUTLASS W4A8 和 Marlin W4A16，不代表实际 checkpoint layout。TP8 来自其
[公开部署指南](https://huggingface.co/moonshotai/Kimi-K2.6/blob/main/docs/deploy_guidance.md)。
GLM-5.2-W4AFP8 的 routed experts 则原生使用 INT4 group-128 和运行时 FP8
E4M3；TP8 来自其[模型卡 serving 配置](https://huggingface.co/PhalaCloud/GLM-5.2-W4AFP8#serving)。

DSV4 的两份 W4A16-g128 workload 同样是 shape-matched Marlin coverage 变体；
DSV4 原生量化 workload 仍是 MXFP4AFP8。不同 activation/weight contract 的结果
不会在报告中计算 speedup。

## Quant provider 与 backend

| `quant_mode` | `humming`（经 SGLang） | `flashinfer` | `sglang_cutlass` | `sglang_marlin` |
|---|---|---|---|---|
| `mxfp4_fp8` | MXFP4 E2M1 + E8M0 group-32，运行时 FP8 E4M3 | `use_wfp4afp8_humming=True` | 不支持 | 不支持 |
| `wint4_afp8` | centered UINT4 表示的对称 INT4 + BF16 group scale，运行时 FP8 E4M3 | packed signed INT4、8-slot W4A8 scales | signed INT4 group-128，静态 tensorwise FP8 | 不支持 |
| `wint4_a16` | 不支持 | 不支持 | 不支持 | symmetric INT4 group-32/64/128 × BF16 |

`wint4afp8`、`wint4xafp8`、`int4_fp8` 和 `w4a8` 会规范化为 `wint4_afp8`；`wint4a16`、`wint4xa16`、`int4_bf16` 和 `w4a16` 会规范化为 `wint4_a16`。`quant_params` 由 provider 补全、校验并原样记录到结果中。不支持的 quant/backend 组合会在加载 workload 后直接失败，并列出该 contract 可用的 backend。

WINT4 使用 BF16 group scale。`wint4_afp8` group-128 workload 可运行 Humming、
FlashInfer Hopper W4A8 和 SGLang CUTLASS W4A8；group-32 workload 当前只能运行
Humming。可共同运行时，各端从同一组逻辑 signed-INT4 gate/up/down 权重构建：

- Humming 使用 centered-uint4 nibble 和 `[gate, up]` W13；
- FlashInfer 使用 two's-complement nibble 和 `[up, gate]` W13；
- SGLang CUTLASS 使用 two's-complement nibble 和 `[gate, up]` W13；
- scalar prequant 的倒数会折入 FlashInfer weight scales，保持相同逻辑权重函数。

但 activation quant policy 不完全相同：SGLang-Humming standard runner 使用动态 per-token/per-group FP8 scale；FlashInfer Hopper W4A8 使用静态 input/prequant scales 和 per-expert alpha；SGLang CUTLASS 使用 workload 的 `fc1_input_scale/fc2_input_scale` 做静态 tensorwise FP8。因此 WINT4 输出会记录 `cross_backend_contract_matched=false`。这些结果可以比较各自原生集成路径性能，但不能把输出差异表述成严格逐元素 cross-backend correctness。

Marlin 是独立的 `wint4_a16` contract：激活保留 BF16，权重先从 GPTQ packed layout repack 成 Marlin layout，计时调用走 `MoeRunner(MARLIN)` 的 `none -> marlin` fused path。当前实现支持 symmetric、无 zero point、无 act-order的 group-32/64/128 INT4；要求 `K % 128 == 0`、`N_local % max(64, group_size) == 0` 和 SM80+。它不能与 `wint4_afp8` 的 latency 当成同一 activation contract 下的直接 backend 替换。

SGLang CUTLASS W4A8 固定 group-128、要求 SM90、`K/N_local` 为 128 的倍数且不支持 `swiglu_limit`。它的 callable 包含 routing reorder、静态 FP8 量化、两次 grouped GEMM、SwiGLU 和最终 weighted combine。

## 计时边界

主指标复用 FlashInfer `bench_gpu_time` 的 CUDA Event latency并关闭 CUDA Graph。输入构造、weight transform/repack、编译和 FlashInfer autotune 在计时外；route-dependent adapter 工作、SGLang CUTLASS 的 FP8 量化和 Marlin runner 的 per-call workspace 保留在 backend callable 内。脚本另报逐次 synchronize 的 wall latency。

原生 NVFP4 是另一套 contract，不能通过修改 workload 名称套用上述两个 provider；需要新增独立 provider。

## GLM-5.2 W4AFP8 实测结果（H20，2026-07-22）

测试运行在 `molou/flashinfer-h20-sglang-dev-07221751` 的 NVIDIA H20
`cuda:1` 上。workload 为 `K=6144`、`N_global=2048`、`N_local=256`、
`E=256`、`top_k=8`、`TP=8`、`EP=1`，使用 balanced routing。两端均为
warmup 20、repeat 100，主指标是 CUDA Event p50；FlashInfer 逐 shape
autotune，Humming 使用 SGLang indexed fused-MoE 路径。

`FI/Humming` 表示 FlashInfer latency / Humming latency；大于 1 表示
Humming 更快。

| M | Active experts | Humming p50 (ms) | FlashInfer p50 (ms) | FI/Humming | 更快的 backend |
|---:|---:|---:|---:|---:|---|
| 1 | 8 | 0.4693 | 0.1368 | 0.292 | FlashInfer 3.43x |
| 8 | 64 | 0.4664 | 0.1901 | 0.408 | FlashInfer 2.45x |
| 32 | 256 | 0.4817 | 0.5656 | 1.174 | Humming 1.17x |
| 128 | 256 | 0.4652 | 0.5767 | 1.240 | Humming 1.24x |
| 512 | 256 | 0.4947 | 0.7095 | 1.434 | Humming 1.43x |
| 1024 | 256 | 0.6059 | 0.9017 | 1.488 | Humming 1.49x |
| 2048 | 256 | 1.1329 | 1.3952 | 1.232 | Humming 1.23x |
| 4096 | 256 | 1.5721 | 2.1747 | 1.383 | Humming 1.38x |
| 8192 | 256 | 3.0734 | 4.0592 | 1.321 | Humming 1.32x |

交叉点位于 `M=8` 和 `M=32` 之间：FlashInfer 在 `M=1/8` 更快，
Humming 在 `M=32–8192` 的七个点全部更快。

该结果只比较 local-rank fused MoE，不包含 router 和通信。两端共享逻辑权重、
输入和 routing seed，但 activation quantization policy 不完全相同，因此这是原生
集成路径的性能对比，不是严格的逐元素正确性对比。

- [详细测试报告](results/glm52-h20-20260722-07221751/README.md)
- [汇总 CSV](results/glm52-h20-20260722-07221751/comparison.csv)
- [Humming 原始结果](results/glm52-h20-20260722-07221751/glm52.humming.json)
- [FlashInfer 原始结果](results/glm52-h20-20260722-07221751/glm52.flashinfer.json)

## Kimi K2.6 与 DSV4 实测结果（H20，2026-07-22）

同一 Pod、GPU 和计时参数下完成原始量化与新增 backend-coverage workload：

- Kimi K2.6 使用真实的 INT4 `group_size=32`。Humming 完成全部 9 个点，
  p50 为 `0.4672–4.2984 ms`；FlashInfer 和 SGLang CUTLASS 的 SM90
  packed-INT4 fused-MoE 固定 group-128，因此没有这两个 backend 的 g32 结果。
- Kimi group-128 shape variant：SGLang CUTLASS W4A8 与 Marlin W4A16 均完成
  9 个点，p50 分别为 `0.2417–4.7172 ms` 和 `0.1519–12.1904 ms`。两者
  activation contract 不同，不报告 cross-backend speedup。
- DSV4 Flash：FlashInfer 在 `M=1/8` 更快；`M=32` 本次 p50 近似持平
  （FlashInfer 低 1.8%）；Humming 在 `M=128–8192` 更快。
- DSV4 Pro：FlashInfer 在 `M=1/8` 更快，Humming 在 `M=32–8192` 更快。
- DSV4 Flash/Pro 的 Marlin W4A16-g128 shape variant 也各完成 9 个点，p50
  分别为 `0.1851–8.8467 ms`、`0.1842–11.7063 ms`；它们与原生 MXFP4
  contract 分表记录。

- [Kimi K2.6 完整结果、运行环境和原始数据](results/kimi26-h20-20260722-07221751/README.md)
- [DSV4 Flash/Pro 完整结果、运行环境和原始数据](results/dsv4-h20-20260722-07221751/README.md)
