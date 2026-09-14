# GLM-5.3 / B300 / 128k prefill：跨引擎优化调查

日期：2026-09-14。对照 vLLM、TensorRT-LLM、TokenSpeed、SGLang 的源码、官方博客和 PR；目标是加速这次单请求、冷缓存的 128k prefill。没有运行新的 GPU 性能实验，也没有修改引擎代码、权重或原始 trace。

**仍有可优化空间。优先顺序是：先验证 NCCL/NVLS 与 host 发射问题，再开发 CP 兼容的 prefill Top-K，随后评估 MoE combine 的通信实现和大 M 的 BF16 投影。** 四个引擎大量复用 FlashInfer、TRTLLM-Gen、DeepGEMM；更换引擎名称本身不能保证换掉热点 kernel。

## 1. 比较基线和证据口径

| 项目 | 本次实际配置 / 证据 |
|---|---|
| 硬件 | 8 × B300，SM103 |
| 模型 / 精度 | GLM-5.3，`modelopt_fp4`；routed experts NVFP4，所列 attention/shared-expert GEMM 为 BF16；KV FP8 |
| 并行 | TP8 / EP8 / prefill CP8，interleave，DP1，DCP1；attention TP 实际为 1 |
| 请求 | 131072 input，OSL=1，BS=1，无 prefix cache hit；4 × 32768 chunk，每卡 4096 query tokens |
| Backend | DSA prefill/decode 为 TRTLLM，MoE 为 FlashInfer TRTLLM；prefill graph 关闭，EAGLE 开启 |
| 测量 | 未采集 TTFT 1547.02 ms；采集时 1828.59 ms。均为单次样本，不能据此量化稳定收益 |
| 当前 trace | Rank 0 GPU 窗口 1798.60 ms；kernel sum 1735.90 ms，四个主模型 chunk kernel sum 1659.75 ms |

原分析：[report.md:1](/Users/huangzhilin/security_inference/profile/b300/glm53/prefill_trace_128k_20260914/analysis_baseline_1789358763/report.md:1)。逐 chunk 数据：[chunk_kernel_breakdown.md:1](/Users/huangzhilin/security_inference/profile/b300/glm53/prefill_trace_128k_20260914/analysis_baseline_1789358763/chunk_kernel_breakdown.md:1)。完整 kernel/module 对照：[kernel_module_mapping.md:1](/Users/huangzhilin/security_inference/profile/b300/glm53/prefill_trace_128k_20260914/analysis_baseline_1789358763/kernel_module_mapping.md:1)。

此次读取的本地版本不同，**不是同版本的四引擎实测横评**：

| 引擎 | 本地 HEAD | 提交日期 | 用途 |
|---|---|---|---|
| SGLang | `8ba15052e9` | 2026-09-10 | 判断当前接入分支和限制 |
| vLLM | `553fcb82d5` | 2026-07-31 | 看图优化、通信和 MLA 实现；较新进展通过线上 PR 核对 |
| TensorRT-LLM | `9769ff1378` | 2026-08-27 | 看 DSA、MoE one-sided/FP8 combine；9 月 GVR prefill 通过 PR diff 核对 |
| TokenSpeed | `5354e72a` | 2026-09-08 | 看 GLM DSA、metadata 复用、RSAG 和 MLA kernel |

Trace 内记录的代码行号与本地 HEAD 不完全一致。**是否已启用以 trace 和运行配置为准；本地源码只用于解释和寻找候选路径。**

## 2. 四个引擎各有什么值得借鉴

| 引擎 | 有价值的实现 / PR / 博客 | 对本 trace 的判断 |
|---|---|---|
| vLLM | [#46635](https://github.com/vllm-project/vllm/pull/46635) 用 RS 取代 AllReduce 后再切片；[#46876](https://github.com/vllm-project/vllm/pull/46876) GLM/DSV3.2 norm、RoPE、collective fusion；[Wide-EP 博客](https://vllm.ai/blog/2025-12-17-large-scale-serving) 的 microbatch overlap | 本 trace 的 MoE 已有 RS，不能再计算一次“AR→RS”收益。值得借鉴的是通信与计算重叠、缩小 eager 区域及 fused epilogue |
| TensorRT-LLM | [One-sided AlltoAll 博客](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog18_Optimizing_MoE_Communication_with_One_Sided_AlltoAll_Over_NVLink.html)、[#11844](https://github.com/NVIDIA/TensorRT-LLM/pull/11844) FP8 combine、[#18702](https://github.com/NVIDIA/TensorRT-LLM/pull/18702) self-sampling GVR prefill Top-K | 最有针对性的两个移植方向：CP→EP 通信改造和新的 prefill selection kernel |
| TokenSpeed | [官方介绍](https://lightseek.org/blog/lightseek-tokenspeed.html)、[#1240](https://github.com/lightseekorg/tokenspeed/pull/1240) DSA prefill plan 复用、[#634](https://github.com/lightseekorg/tokenspeed/pull/634) TRTLLM CuTeDSL Top-K、Triton RSAG | plan 生命周期和融合通信值得参考；它的 Blackwell DSA 也复用 TRTLLM sparse MLA，普通 MLA 的宣传数字不能套到本 trace |
| SGLang | [GLM5.2 优化博客](https://www.lmsys.org/blog/2026-07-13-glm52-optimization)、[#35175](https://github.com/sgl-project/sglang/pull/35175) ragged prefill Top-K V2、[#30117](https://github.com/sgl-project/sglang/pull/30117) BF16 TGV | 已有不少优化；但 CP + PAGED prefill、较大 M 和 NVFP4 的组合没有覆盖所有快路径 |

## 3. 优先级及可落地程度

下面的“时间预算”是 trace 中被该方案覆盖的原始工作量，不是承诺能节省的时间。不同方案会作用于相同路径，收益不能直接相加。

| 优先级 | 方案 | 涉及时间预算 | 当前能否直接做 |
|---|---|---:|---|
| P0 | 无 profiler 重复基线；确认 rank 2/7 launch 空隙是否仍存在 | 跨 rank host/collective 链；不能拿 317 ms 到达偏差当纯等待 | 可直接采集验证 |
| P1 | `--enable-nccl-nvls`；检查实际 collective 算法、消息尺寸和 buffer 注册 | 全窗口通信 588.14 ms，其中 MoE AG+RS 427.55 ms | 可做独立 A/B；收益未知 |
| P1 | CP/PAGED prefill Top-K：SGLang V2 adapter 或 TRTLLM GVR V2 adapter | 四个主 chunk Top-K 57.75 ms；第 4 chunk 25.58 ms | 需要适配 CP row 和物理 KV 索引 |
| P1–P2 | 复用剩余 metadata、缩小 Python/eager 区域 | host launch、部分小 kernel 和放大的 peer 等待 | 先测实际重复项；CP8 不能直接开现有 prefill graph |
| P2 | CP→EP one-sided dispatch/combine；之后评估 FP8 combine | MoE dispatch 184.90 ms + combine 242.65 ms | 需要代码接入与精度验证 |
| P2 | `o_proj` 等大 M BF16 GEMM 专项调优，或校准 FP8 | 主模型 `o_proj` 146.76 ms，`q_b_proj` 45.46 ms | 可做 kernel microbenchmark；量化要另做质量评估 |
| P2 | NVFP4 MoE finalize/通信融合与 microbatch overlap | 主模型 MoE GEMM 215.37 ms、finalize/routing 72.24 ms，加暴露的通信 | 工程量较大；先保留现有量化 recipe |
| P3 | Indexer logits 与 selection 的分块流水/更深融合 | 主模型 logits 75.55 ms + Top-K 57.75 ms | 研究方案，尚无本形状收益证据 |

另做两项配置对照：OSL=1 关闭 MTP；chunk 16k/32k/64k。它们有明确验证价值，但目前不能给出收益数字。

## 4. 通信：先 NVLS，再 one-sided，最后考虑低精度 combine

### 4.1 最小成本实验：NVLS

当前主要通信 kernel 为 `ncclDevKernel_AllGather_RING_LL` 和 `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL`；日志明确 `NCCL_NVLS_ENABLE=0`。SGLang 已有面向 prefill heavy workload 的 `--enable-nccl-nvls`，在启动时设置 NCCL 环境变量：

- [server_args.py:1954](/Users/huangzhilin/security_inference/sglang/python/sglang/srt/server_args.py:1954)
- [engine.py:1661](/Users/huangzhilin/security_inference/sglang/python/sglang/srt/entrypoints/engine.py:1661)

建议先只改这个开关，保持 CP8、EP8、chunk、模型和请求相同。观察实际 RS/AG kernel、collective 时长与 TTFT 是否一起改善。NVLS 能否参与每一种 collective 由 NCCL 版本、拓扑和配置共同决定；**打开开关不等于所有 AG/RS 都自动加速**。[NCCL 官方配置说明](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-nvls-enable)

先保留 `NCCL_ALGO`、`NCCL_PROTO` 自动选择；核实是否有外部强制设置。若需调参，先用本机对应消息大小的 collective benchmark 验证，不能仅凭 `RING_LL` 名称断言选错算法。不要把本次 `P2P/IPC` 数据通路误判成 Socket。

### 4.2 TRTLLM 的 one-sided AlltoAll 是结构性候选

当前 MoE 路径可概括为：

```text
CP-local hidden [4096,6144]
  → FP4 quant + routing metadata
  → grouped AllGather，收集全局 token
  → 本 rank 的 expert GEMM / finalize
  → BF16 全局结果 [32768,6144]
  → ReduceScatter
  → CP-local output [4096,6144]
```

单次 BF16 RS 的逻辑输入 384 MiB，输出 48 MiB。TRTLLM 的 one-sided 方案提供另一条路径：按目的 rank 去重 token，直接读写对称内存，MoE 消费 rank-major 收件区，再从相关 peer 收回结果。它可以减少中间复制和无关 rank 的数据搬运。[设计与 benchmark](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog18_Optimizing_MoE_Communication_with_One_Sided_AlltoAll_Over_NVLink.html)

源码入口：[nvlink_one_sided.py:139](/Users/huangzhilin/security_inference/TensorRT-LLM/tensorrt_llm/_torch/modules/fused_moe/communication/nvlink_one_sided.py:139)。TokenSpeed 另一条可借鉴路线是保持 AG/RS 语义，替换 collective 实现并融合 residual/RMSNorm：[triton_rsag.py:106](/Users/huangzhilin/security_inference/tokenspeed/python/tokenspeed/runtime/distributed/comm_backend/triton_rsag.py:106)、[comm_ops.py:324](/Users/huangzhilin/security_inference/tokenspeed/python/tokenspeed/runtime/distributed/comm_ops.py:324)。

**当前 SGLang 的 `--moe-a2a-backend flashinfer` 不可直接解决本案例：**

1. 配置要求 `enable_dp_attention` 且 `dp_size == tp_size`，本次是 CP8 / DP1：[moe_hook.py:263](/Users/huangzhilin/security_inference/sglang/python/sglang/srt/arg_groups/moe_hook.py:263)。
2. 即使改为受支持的 DP 配置，当前 extend 分支仍回到 `_dispatch_prefill_allgather`：[flashinfer.py:300](/Users/huangzhilin/security_inference/sglang/python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py:300)。
3. 单个长请求不能通过改成 DP8 自动分摊其 prefill；必须保留或另行实现长请求 token 分片。

因此要实现的是 **CP-local token 与 EP dispatcher 的接入**，不是删除配置断言。需要保留 interleave token 顺序、expert ID、source-rank 归属和 shared-expert 结果的正确归并。EP8 的收益也不能按 EP64/72 的通信冗余比例外推。

### 4.3 FP8 combine 比“再压缩 dispatch”更有增量价值

Dispatch 已经 FP4；更明显的增量在 BF16 combine。TRTLLM [#11844](https://github.com/NVIDIA/TensorRT-LLM/pull/11844) 于 2026-03-16 合入：专家输出在传输前转为 FP8，接收侧恢复工作 dtype 并归并。其报告的微基准 combine 约从 15.7 μs 降至 7.7 μs，**这不是本次 384 MiB / CP8 的实测**。

本地接口：[llm_args.py:1463](/Users/huangzhilin/security_inference/TensorRT-LLM/tensorrt_llm/llmapi/llm_args.py:1463)、[nvlink_one_sided.py:279](/Users/huangzhilin/security_inference/TensorRT-LLM/tensorrt_llm/_torch/modules/fused_moe/communication/nvlink_one_sided.py:279)。

这属于有损传输，恢复 BF16 dtype 不会恢复被舍入的信息。建议先完成 BF16 one-sided 对照，再单独启用 FP8 combine，比较各层输出、长上下文任务质量及 TTFT。不能把“传输字节减半”直接写成“242.65 ms 减半”，因为原时长还包含 peer 到达不齐和同步。

## 5. Top-K：最清晰的 kernel 移植机会

### 5.1 为什么开 SGLang Top-K V2 仍可能没变化

SGLang [#35175](https://github.com/sgl-project/sglang/pull/35175) 于 2026-08-21 合入，报告 B200 上 ragged prefill Top-K 为旧实现的 1.3–1.8 倍速度；PR 明确保留 CP 路径的旧 kernel。

当前 dispatch 的两个 V2 分支也有实际限制：PAGED 快路径要求 row 数与 request/page-table row 数匹配、`row_starts is None` 等；RAGGED 快路径不接收 CP 的 `batch_idx_list`。见 [dsa_topk_backend.py:120](/Users/huangzhilin/security_inference/sglang/python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py:120)。本 trace 发射的是 `topk_transform_prefill_kernel`，不是这些 V2 kernel。

本次 TRTLLM DSA 需要 **物理 paged KV 索引**。移植不能仅把旧调用替换成返回普通逻辑 Top-K 的 kernel；还要处理每个 CP query 的 causal 边界、row-to-request 关系、prefix 偏移和 page-table transform。接口对此有明确说明：[dsa_indexer_metadata.py:83](/Users/huangzhilin/security_inference/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer_metadata.py:83)。

### 5.2 更近期的候选：TRTLLM self-sampling GVR V2 prefill

[#18702](https://github.com/NVIDIA/TensorRT-LLM/pull/18702) 已于 **2026-09-09** 合入，晚于本地 TRTLLM checkout。这是 prefill 专用支持，和依赖上一 decode step hint 的 GVR V1 不同。

已核对 PR diff：它提供 `run_prefill(logits, row_starts, row_ends, indices, max_row_len=None)`；按行的有效窗口执行 selection，输出相对该窗口的 local index，短行走 identity/padding；prefill 用单 CTA/row，不使用 decode 的 cluster split。配置基于 `enable_heuristic_topk` 和 `use_self_sampling_topk`，前置重构为 [#18446](https://github.com/NVIDIA/TensorRT-LLM/pull/18446)。**旧 checkout 的 `TRTLLM_GVR_SELF_SAMPLING` 环境变量不是新 PR 的接入方式。**

PR 的 prefill 实际流水测试在 DSV4 上报告 selection kernel 1.84–2.61×；K=2048 的 DSV3.2 也列入正确性验证，但没有本案例的 GLM-5.3 / CP8 / 128k TTFT 证据。其 BS=1 实验报告 TTFT 基本不变，不能据此预告本次端到端收益。

建议同一份真实 logits 比较三个候选：旧 kernel、SGLang V2 + CP/PAGED adapter、GVR V2 + CP/PAGED adapter。时间包含索引转换和额外 launch。先实现独立转换验证语义，再判断是否值得融合回 selection epilogue。

正确性应覆盖：四个 chunk 的不同 prefix 长度、interleave causal mask、非连续 KV page、短行、并列分数、`-1` padding；还要验证下游 sparse MLA 读到的 KV token 集合。SGLang 存在仍开放的 Top-K overflow 报告 [#35257](https://github.com/sgl-project/sglang/issues/35257)、[#36807](https://github.com/sgl-project/sglang/issues/36807)，这里是需要加入回归样本的线索，**不是本 trace 已产生错误的证据**。

### 5.3 增益尺度

四个主 chunk 的 logits+Top-K 为 **10.53 / 24.55 / 39.84 / 58.37 ms**，占各 chunk kernel 时间 **2.44% / 5.84% / 10.09% / 14.12%**。其中 Top-K 共 **57.75 ms**。

假设 Top-K 实际减半且完全在关键路径，节省的 kernel 工作量为 **28.87 ms**，约为 1798.60 ms GPU 窗口的 **1.6%**。这是条件估算，非 TTFT 预测。若要明显提高端到端收益，需要同时优化 logits、通信或 host 链。

## 6. TokenSpeed 的 prefill plan 复用，以及 host/graph 优化

TokenSpeed [#1240](https://github.com/lightseekorg/tokenspeed/pull/1240) 于 2026-08-27 合入，复用请求不变量：page table、packed KV slots、row boundaries 和 CPU candidate lengths，移除动态 shape 和 GPU `.item()` 引起的同步。其 GLM-5.2-FP8 / B200 / 8k 实验报告 TTFT 降 5.69%，是值得借鉴的真实 prefill 优化。

本地实现：[glm5.py:609](/Users/huangzhilin/security_inference/tokenspeed/python/tokenspeed/runtime/models/glm5.py:609)。它仍重新计算每层 query/weights 对应的 scores；**不能把复用 metadata 理解成任意复用跨层 Top-K 结果**。

SGLang 已经将若干字段保存在 `DSAMetadata`，`seq_len_sum`/`max_seq_len` 也取自 CPU length mirror，而不是无条件读取 GPU。见 [dsa_indexer_metadata.py:115](/Users/huangzhilin/security_inference/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer_metadata.py:115)、[dsa_indexer.py:1089](/Users/huangzhilin/security_inference/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py:1089)。所以不能照搬 TokenSpeed 的同步次数和百分比。应检查 CP adapter 是否重复构建 row 映射、张量转换、分配，以及 metadata/kernel wrapper 的逐层 Python 成本。

Trace 已观察到 rank 2/7 在 81% 的 MoE AG 中最后进入；rank 2 有 173.70 ms 的 GPU 空隙发生在后续 launch 尚未开始时。先用低开销 timeline 和 CPU scheduling 采样确认该现象，而后决定是否处理 CPU affinity、线程竞争或 Python 代码。当前 trace 无法证明具体 NUMA/绑核错误。

Prefill graph 需要专门适配 CP8：

- SGLang piecewise graph 对 CP 有禁用规则：[cuda_graph_hook.py:245](/Users/huangzhilin/security_inference/sglang/python/sglang/srt/arg_groups/cuda_graph_hook.py:245)。
- 当前 CP breakable graph 的白名单为 `zigzag + trtllm_mha`，不是这里的 `interleave + dsa`：[bcg.py:48](/Users/huangzhilin/security_inference/sglang/python/sglang/srt/layers/cp/bcg.py:48)。
- CP Indexer 对 graph 也有显式限制：[dsa_indexer.py:1330](/Users/huangzhilin/security_inference/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py:1330)。

因此建议的开发步骤是：先把可复用元数据和缓冲区固定下来，再对稳定的投影/norm/MoE 子段做 capture，保留确实动态的 CP/Indexer 边界。要核对最终 resolved config 和实际 `cudaGraphLaunch`，不能只记录传入了 graph 开关。

## 7. Dense GEMM：应瞄准大 M 的 o_proj

四个主 chunk 中的主要 BF16 GEMM：

| Module / 算子 | 实际输入乘法形状 | 累计 |
|---|---|---:|
| `self_attn.o_proj` | `[4096,16384] × [16384,6144]` | **146.76 ms** |
| `self_attn.q_b_proj` | `[4096,2048] × [2048,16384]` | **45.46 ms** |
| `mlp.shared_experts.gate_up_proj` | `[4096,6144] × [6144,4096]` | 36.83 ms |
| `mlp.shared_experts.down_proj` | `[4096,2048] × [2048,6144]` | 18.00 ms |
| `fused_qkv_a_proj_with_mqa` | `[4096,6144] × [6144,2624]` | 27.08 ms |
| MLA K BMM / V BMM | `[64,4096,192] × [64,192,512]` / `[64,4096,512] × [64,512,256]` | 20.87 / 21.88 ms |

`fetch_qkv_latent` 是 fused Q/KV 低秩投影结果的访问包装。相比它的 27.08 ms，`o_proj` 的绝对预算大得多。

SGLang GLM5.2 博客中 TGV 的主要测试范围为 decode M=1–32；当前 selector 对 M=4096 不走 TGV，`o_proj` 的 K=16384 也超出其允许范围。见 [cutedsl_bf16_gemm.py:1357](/Users/huangzhilin/security_inference/sglang/python/sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py:1357)。**`--bf16-gemm-backend cutedsl` 不是此次 prefill 的现成加速方案。**

建议做两层实验：

1. 保持 BF16，用实际 stride/layout、M=2048/4096/8192 比较 cuBLAS/cuBLASLt/CuTe 大 M kernel，检查 tactic、tile、2CTA 和 epilogue；把布局转换、workspace 和真实 layer launch 顺序计入。
2. 若愿意改变量化 recipe，先校准 `o_proj` 或 `q_b_proj` 的 FP8，再考虑 shared experts。vLLM 的 [Blackwell 博客](https://vllm-project.github.io/2026/02/03/dsr1-gb200-part1.html) 和 TRTLLM 的 [EP 优化博客](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog14_Scaling_Expert_Parallelism_in_TensorRT-LLM_part3.html) 提供投影低精度实现参考；它们对 DeepSeek 的质量结果不等于 GLM-5.3 可直接复用。

这里先不建议 FP4 全量化 attention。当前 checkpoint 刻意保留的 BF16 路径，需要用本模型评估证明可以降低精度。

## 8. MoE finalize、通信融合和 overlap

本次主模型 `moe_other` 72.24 ms，大头是 `finalizeKernelVecLoad`。可借鉴 TRTLLM 将 FC2/finalize、shared-expert add、combine 的相邻操作整合，减少全局结果写出和再次读入。[EP kernel fusion 说明](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog14_Scaling_Expert_Parallelism_in_TensorRT-LLM_part3.html)

SGLang 有 `SGLANG_ENABLE_MOE_DEFERRED_FINALIZE`，但实际调用还要求 bypassed top-k 等条件，所在路径不是本 trace 的 `forward_normal`。见 [deepseek_v2.py:1077](/Users/huangzhilin/security_inference/sglang/python/sglang/srt/models/deepseek_v2.py:1077) 和 [deepseek_v2.py:1169](/Users/huangzhilin/security_inference/sglang/python/sglang/srt/models/deepseek_v2.py:1169)。不能只打开 env 就把 72 ms 记为可消除。

vLLM 有 `GEMM → ReduceScatter` 图替换：[collective_fusion.py:341](/Users/huangzhilin/security_inference/vllm/vllm/compilation/passes/fusion/collective_fusion.py:341)。但本次 attention TP=1，`o_proj` 后没有同样的 TP reduce-scatter；真正的 RS 在 MoE 汇合处。可借鉴融合方式，不能把该 pass 直接套到 `o_proj`。

另一条路线是把一个 chunk 内的 query/token 分为两个 microbatch，重叠一组的 MoE dispatch/combine 与另一组的计算，参考 [vLLM DBO](https://vllm.ai/blog/2025-12-17-large-scale-serving) 和 [ubatch_utils.py:38](/Users/huangzhilin/security_inference/vllm/vllm/v1/worker/ubatch_utils.py:38)。本次通信与计算重叠仅约 0.96 ms，理论上有空间；但必须保证同层 KV 写入、causal attention 和 collectives 的一致顺序。不能把依赖前缀 KV 的两个 chunk 当独立请求并发。

先比较 `M=4096 × 1` 与 `M=2048 × 2`：小 microbatch 会增加 launch、通信轮数，降低 GEMM 效率，重叠收益要覆盖这些损失。NVFP4+CP 的 dispatcher 能否支持异步 completion 也要验证。

MegaMoE 暂列探索项。SGLang [#38563](https://github.com/sgl-project/sglang/pull/38563) 在查询时仍 OPEN，提供保留 ModelOpt NVFP4 scales 的 FlashInfer mega kernel 接入，但 PR 自己报告的 Qwen3.5 测试比 unfused baseline 慢 13%–22%。这不能用作“融合成一个 kernel 就一定更快”的依据。现有 DeepEP V2 adapter 也要求 128×128 blockwise FP8，明确拒绝 FP4 experts：[layer.py:240](/Users/huangzhilin/security_inference/sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py:240)。

## 9. Attention 和 Indexer logits：不要混淆普通 MLA 与 DSA

TokenSpeed 的 [MLA README](https://github.com/lightseekorg/tokenspeed/tree/main/tokenspeed-mla) 比较的是普通 ragged MLA prefill；其 AOT binary 和开源版本表现也不同。GLM DSA 的本地实现调用 `trtllm_batch_decode_with_kv_cache_mla(..., sparse_mla_top_k=..., backend="trtllm-gen")`，见 [__init__.py:312](/Users/huangzhilin/security_inference/tokenspeed/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/flashinfer/__init__.py:312)。调用名中的 decode 不代表只处理 decode；这里把每个 sparse query 当一个单位处理。

本 trace 已是 Q/KV E4M3 的 TRTLLM sparse FMHA。换成普通 `tokenspeed_mla` 既没有证明能保留 DSA 的索引语义，也不能借用其普通 MLA 数字。

对 logits 的增量工作建议是：

- 在 B300 的真实 `4096 query × 32k/64k/96k/128k KV` 形状下对 DeepGEMM tile/scheduler 做 benchmark。已有 DeepGEMM、FP8 和 `clean_logits=False` 不算新增方案。
- 分开调 **scheduler chunk** 与 **Indexer 内部 query tile**。最后一块若物化完整 FP32 score 矩阵，逻辑尺寸是 `4096 × 131072 × 4 = 2 GiB`，实际分配还依赖实现的裁剪/对齐。较小 query tile 可能改善 workspace/L2，但也增加 launch，并不是越小越好。
- 后续可以研究 logits producer 与 Top-K consumer 的 tile 流水，或带正确候选归并的融合 selection，以减少大 score 矩阵的读写。它不能跳过模型规定的有效分数计算；K=2048 的候选存储与合并也有明显成本。

TRTLLM 的 [DSV3.2 技术博客](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.html) 提供 sparse MLA、DeepGEMM tile 和 Top-K 优化背景；TokenSpeed 的 query tiling 实现在 [__init__.py:524](/Users/huangzhilin/security_inference/tokenspeed/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/deep_gemm/__init__.py:524)。这些是实验设计依据，目前没有本 trace 上的新增实测收益。

## 10. 建议执行的实验顺序

每组先预热编译和 autotune，再测冷 prefix 的同一请求。每个配置至少重复 5–10 次，交错执行 A/B，报告 median 和波动；不在重 profiler 下给最终 TTFT 排名。当前配置已允许 FlashInfer autotune，要检查实际命中形状与缓存，不能把“开启 autotune”列成新增功能。

| 实验 | 改动 | 必须记录 |
|---|---|---|
| E0 | 原配置重复运行，另采低开销多 rank timeline | TTFT、每 chunk span、rank 2/7 launch gap、collective arrival spread |
| E1 | 仅开 `--enable-nccl-nvls` | 是否实际生效、AG/RS kernel、collective 时长、TTFT |
| E2 | 在胜出的通信配置上，OSL=1 关闭 EAGLE/MTP | TTFT 和 draft extend/verify 是否消失；同时保留原始 EAGLE 结果 |
| E3 | chunk=16384 / 32768 / 65536，同时匹配 prefill token budget | 每卡 M、显存峰值、logits tiling、chunk 数、TTFT |
| E4 | 从四个 chunk 提取真实 logits，比较三套 Top-K + transform | 正确索引集合、真实有效长度、单 kernel 和整个 selection 链时间 |
| E5 | BF16 o_proj/q_b_proj 大 M 微基准 | 完整 layout 转换成本、实际 tactic、同精度数值误差 |
| E6 | CP→EP one-sided BF16 adapter，然后单独 FP8 combine | 与原 CP/EP 输出一致性、通信字节和时间、端到端质量/TTFT |
| E7 | metadata/局部 capture，再做 microbatch overlap | 实际 graph/stream overlap、launch 数和跨 rank 空隙、显存与 TTFT |

E2 只针对本次 OSL=1 目标。Trace 中四次 draft extend GPU span 共约 62.55 ms，但这些活动与端到端边界并非简单可减关系；最终以 A/B 为准。长输出 agentic 场景应独立评价 MTP。

不建议把 DCP8、PD、prefix cache 命中率的改善当成本次冷 prefill 的直接优化证据：DCP8 的已有单次 TTFT 是 1632.30 ms，高于 baseline 1547.02 ms；PD 面向资源隔离与吞吐；prefix cache 需要存在可复用前缀。另需注意当前 DSA CP helper 断言 attention TP=1，`CP4 × attention TP2` 不是直接可用的扫参组合：[communicator_dsa_cp.py:83](/Users/huangzhilin/security_inference/sglang/python/sglang/srt/layers/communicator_dsa_cp.py:83)。

## 11. 可核查的 PR 清单

状态为 2026-09-14 查询结果。MERGED 表示上游已合入，不表示本次运行环境已包含或启用。

| 项目 | PR | 状态 / 日期 | 本次作用 |
|---|---|---|---|
| vLLM | [#46635](https://github.com/vllm-project/vllm/pull/46635) | MERGED，06-28 | AR→RS 设计参考；本 trace 已 RS |
| vLLM | [#46876](https://github.com/vllm-project/vllm/pull/46876) | MERGED，06-28 | GLM/DSV3.2 op fusion；部分后续融合仍为 TODO |
| TRTLLM | [#11844](https://github.com/NVIDIA/TensorRT-LLM/pull/11844) | MERGED，03-16 | FP8 MoE combine |
| TRTLLM | [#18446](https://github.com/NVIDIA/TensorRT-LLM/pull/18446) | MERGED，09-08 | GVR 配置与 dispatch 重构 |
| TRTLLM | [#18702](https://github.com/NVIDIA/TensorRT-LLM/pull/18702) | MERGED，09-09 | 真正的 self-sampling prefill Top-K；本地旧 HEAD 未包含 |
| TokenSpeed | [#634](https://github.com/lightseekorg/tokenspeed/pull/634) | MERGED，07-13 | CuTeDSL decode Top-K；不当作 prefill 收益 |
| TokenSpeed | [#1240](https://github.com/lightseekorg/tokenspeed/pull/1240) | MERGED，08-27 | GLM DSA prefill plan 复用 |
| SGLang | [#30117](https://github.com/sgl-project/sglang/pull/30117) | MERGED，07-07 | 小 M BF16 TGV；不覆盖本次 M=4096 |
| SGLang | [#35175](https://github.com/sgl-project/sglang/pull/35175) | MERGED，08-21 | ragged prefill Top-K V2；CP 仍需适配 |
| SGLang | [#38563](https://github.com/sgl-project/sglang/pull/38563) | OPEN | NVFP4 MegaMoE 探索项，不能假定更快 |

部分 PR 的描述及查询状态保存在 [cross_engine_research_sources.json:1](cross_engine_research_sources.json)。其中的外部 benchmark 是上游作者报告，本调查没有在用户的 B300 上复现。
