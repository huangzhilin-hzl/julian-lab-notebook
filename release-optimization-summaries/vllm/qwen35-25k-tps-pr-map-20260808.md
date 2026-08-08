# Qwen3.5 25K TPS/GPU：相关 PR 与性能链路

- 更新时间：2026-08-08
- 原文：[vLLM Reaches 25K Total TPS/GPU on Qwen3.5](https://vllm.ai/blog/2026-08-06-qwen35-25k-tps)
- 模型：[nvidia/Qwen3.5-397B-A17B-NVFP4](https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4)
- 复现配方：[NVIDIA/srt-slurm-recipes](https://github.com/NVIDIA/srt-slurm-recipes/tree/main/recipes/multi-node/Qwen3.5/GB200/8k1k/vllm/disagg)
- 核对范围：博客直接引用的 10 个实现 PR，以及 vLLM 接入明确依赖的 1 个 FlashInfer 修复 PR

## 结论

25K total TPS/GPU 不是某一个 kernel PR 的单点收益，而是三条链路同时成立后的系统结果：

1. **Blackwell GDN prefill**：FlashInfer 提供 CuTe-DSL kernel，vLLM 在满足条件时自动路由到该实现。
2. **HMA + NIXL 混合状态传输**：P/D 分离不仅传 full-attention KV，还要正确传 GDN/SSM 的 conv/recurrent state，并处理逻辑块、物理块和异构 TP 映射。
3. **无竞态 async scheduling**：解决 RDMA 收到的新状态被延迟 zeroing 覆盖，以及已释放 block 被仍在执行的旧 step 回写污染这两类竞态。

截至 2026-08-08，博客直接引用的 10 个 PR 均已合入；此外，FlashInfer #3155 是 vLLM #40717 PR 描述中明确要求的正确性前置修复，也已合入。

```text
Qwen3.5 prompt
  -> vLLM GDN backend dispatch
  -> FlashInfer Blackwell GDN prefill kernel
  -> HMA 将不同 layer type 映射到对应物理 cache 区域
  -> NIXL 传输 FA KV + GDN/SSM state
  -> decode 侧 async scheduler 安全接管收到的 blocks
  -> GB200 NVL72 上的 P/D + DEP serving
```

## PR 总表

### 1. Blackwell GDN prefill

| PR | 合入时间 / merge commit | 核心作用 | 关键代码路径 | 公开验证 |
| --- | --- | --- | --- | --- |
| [FlashInfer #3001](https://github.com/flashinfer-ai/flashinfer/pull/3001) | 2026-04-13 / `7c562d50` | 新增 SM100/SM100A CuTe-DSL chunked GDN prefill kernel、tile scheduler、Python API、benchmark 和测试 | `flashinfer/gdn_kernels/blackwell/gdn_prefill.py`、`flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py`、`flashinfer/gdn_kernels/blackwell/gated_delta_net_tile_scheduler.py` | PR 给出的 Qwen3.5 多尺寸/TP/shape microbenchmark 相对 FLA/Triton 为约 **1.02x–5.78x** |
| [FlashInfer #3155](https://github.com/flashinfer-ai/flashinfer/pull/3155) | 2026-04-25 / `5e1318cb` | 修复 persistent prefill kernel 在 spawned worker 中取得错误/陈旧 SM 数，避免 CTA 无工作并在首次调用死锁 | Blackwell GDN tile scheduler | 这是 vLLM #40717 描述中明确要求先合入的正确性依赖，不是独立性能来源 |
| [vLLM #40717](https://github.com/vllm-project/vllm/pull/40717) | 2026-05-20 / `1cb22443` | 在 Blackwell 条件满足时，让 `auto`/`flashinfer` GDN prefill backend 路由到 FlashInfer；不满足时保留 Triton/FLA fallback | `vllm/model_executor/layers/mamba/gdn_linear_attn.py`、`vllm/platforms/cuda.py` | 8xB200、Qwen3.5-397B-A17B-NVFP4：kernel 最高 **5.92x**；prefill-only throughput **1.13x**；mean TTFT **-12%** |

vLLM #40717 的自动选择条件不是“所有 Blackwell 都无条件启用”，而是至少包括：CUDA 平台、SM10.x、`head_k_dim == 128`、CUDA runtime 13+，并安装相应 CuTe-DSL/CUTLASS 依赖。不满足约束时仍走 Triton/FLA。

### 2. HMA + NIXL 混合状态传输

| PR | 合入时间 / merge commit | 核心作用 | 关键代码路径 | 正确性/性能边界 |
| --- | --- | --- | --- | --- |
| [vLLM #35758](https://github.com/vllm-project/vllm/pull/35758) | 2026-03-06 / `5b3ba94a` | 让 `NixlConnector` 理解 Hybrid KV Cache Manager 的逻辑块与物理 cache 区域，只传相应 layer type 的有效区域 | `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py`、`vllm/v1/core/kv_cache_manager.py` | descriptor 数从 **4,284 降到 1,650**；小规模单机 8xH100 实验报告最高约 **7% throughput**，不能外推为 25K 结果的独立增益 |
| [vLLM #36687](https://github.com/vllm-project/vllm/pull/36687) | 2026-03-16 / `f5c081d4` | 建立 hybrid SSM-FA P/D 主干：NIXL 同时传 FA KV 和 Mamba/SSM state，使用双 descriptor view，初始支持 P/D 同构 TP | `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py`、`vllm/distributed/kv_transfer/kv_connector/utils.py` | PR 的 Nemotron GSM8K 验证证明主干可工作；当时 TP>1 仍要求关闭 async scheduling，后续由竞态修复链解决 |
| [vLLM #37310](https://github.com/vllm-project/vllm/pull/37310) | 2026-03-19 / `d49f2731` | 对 Mamba/SSM 使用 N-1 prefill，避免 decode 侧用已完成 state 重算最后一个 prompt token 并把多余 step 烧进状态 | `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py` | 修复生成重复/状态污染；GSM8K 5-shot 多次运行均值为 0.8416 |
| [vLLM #37416](https://github.com/vllm-project/vllm/pull/37416) | 2026-04-02 / `66e86f1d` | 新增 `VLLM_SSM_CONV_STATE_LAYOUT=DS`，把 conv state 从 `(state_len, dim)` 调整为 `(dim, state_len)`，便于连续切分 TP shard | `vllm/envs.py`、`vllm/model_executor/layers/mamba/gdn_linear_attn.py`、`vllm/model_executor/layers/mamba/mamba_mixer2.py`、`vllm/model_executor/layers/kda.py` | 为异构 TP 的 3-read 传输铺路；PR 的 colocated benchmark 报告 DS layout 的 TTFT 最高约 **1.5x** 更好 |
| [vLLM #37635](https://github.com/vllm-project/vllm/pull/37635) | 2026-04-06 / `bfdc0a3a` | 实现 hybrid SSM-FA 的异构 TP 传输，覆盖 `P_TP > D_TP`、`D_TP > P_TP`、KV head replication，并修复 GQA head-to-rank 映射 | `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py`、`vllm/distributed/kv_transfer/kv_connector/v1/ssm_conv_transfer_utils.py` | 重点是跨 TP 配置的正确状态切片与读取；不能把它单独视为吞吐 PR |
| [vLLM #41869](https://github.com/vllm-project/vllm/pull/41869) | 2026-05-14 / `24337fb8` | 把上述混合 SSM 传输链扩展到 Qwen3.5 GDN：conv-state split、异构 TP kernel block 匹配、P/D 物理块倍率不一致处理 | `vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py`、`vllm/distributed/kv_transfer/kv_connector/v1/ssm_conv_transfer_utils.py` | Qwen3.5-0.8B 在 9 组 P_TP/D_TP 组合上 GSM8K 分数均落在 standalone baseline `0.323 +/- 0.03` 范围 |

这组 PR 的实际数据流可以概括为：

```text
HMA logical block
  -> 按 attention / SSM(GDN) layer type 定位 physical page
  -> 生成 FA KV descriptor view + SSM state descriptor view
  -> 根据 P_TP / D_TP 与 KV head replication 计算 shard/offset
  -> NIXL 将两类状态写入 decode 侧对应 blocks
```

其中 `VLLM_SSM_CONV_STATE_LAYOUT=DS` 是博客配方明确要求的设置；没有它，SSM/GDN conv-state transfer 不能按该路径正确工作。

### 3. Async scheduling 的两类竞态

| PR | 合入时间 / merge commit | 修复的竞态 | 实现方式 | 关键代码路径 |
| --- | --- | --- | --- | --- |
| [vLLM #45357](https://github.com/vllm-project/vllm/pull/45357) | 2026-06-15 / `d467a2a7` | request 已结束，但下一 async step 仍在飞；block 被立即复用并接收 RDMA 状态后，又被旧 step 的 GPU 写污染 | 用 scheduler sequence fence 延迟真正归还 block pool；请求 bookkeeping 仍立即清理 | `vllm/v1/core/sched/scheduler.py`、`vllm/v1/core/kv_cache_manager.py`、`vllm/v1/core/single_type_kv_cache_manager.py`、`vllm/v1/core/kv_cache_coordinator.py` |
| [vLLM #48481](https://github.com/vllm-project/vllm/pull/48481) | 2026-07-16 / `530852f9` | NIXL 已把 remote prefill KV 写进新 block，但异步排队的 allocation zeroing 随后把收到的 attention KV 清零 | NIXL metadata 标记即将被 remote load 覆盖的 attention blocks，scheduler 对这些 blocks 跳过 zeroing，而不是插入全局 CUDA 同步 | `vllm/v1/core/kv_cache_manager.py`、`vllm/v1/core/sched/scheduler.py`、`vllm/v1/core/single_type_kv_cache_manager.py` |

两者处理的是不同 race，缺一不可：

```text
#45357: old GPU write -> prematurely reused block -> RDMA new state -> old GPU write corrupts it
#48481: RDMA new KV -> delayed zeroing kernel -> new KV is erased
```

博客明确说明，在这两类 race 修复前启用 async scheduling 会让 accuracy 降到 0；而 `--async-scheduling` 又是跨过 25K tok/s/GPU 的关键功能之一。

## 25K 数字到底证明了什么

博客的完整 serving 测试合同如下：

| 维度 | 设置 |
| --- | --- |
| 硬件 | GB200 cluster，NVLink72 |
| 模型 | Qwen3.5-397B-A17B-NVFP4 |
| 请求长度 | ISL/OSL = 8192/1024 |
| Decode | 固定 1 个 DEP8 endpoint，即 8 GPU |
| Prefill | 4–8 个 DEP2 endpoints，每个 2 GPU |
| 并发扫描 | 64–5120 |
| 软件 | `vllm/vllm-openai:nightly-d223c90`、Dynamo `1.2.0.dev20260526`、srt-slurm `1.0.32` |
| 准确率 | 五组拓扑的 GSM8K 均为 88%，与 aggregated run 一致 |
| 峰值结论 | final Pareto frontier 达到约 **25,000 total TPS/GPU** |

这个结果重点优化 Pareto 曲线左侧，即总吞吐，而不是低并发下的单用户 Gen TPS、TPOT 或 ITL。尤其要注意：

- `--stream-interval 100` 会把流式输出按 100 token 缓冲，降低 frontend overhead，但也会影响逐 token latency 的测量解释。
- prefill 侧 `--max-num-batched-tokens 16384` 相当于 2x ISL；博客报告它在 4/5/6xDEP2 高并发配置中带来约 **+8% total TPS/GPU**。
- decode 侧在并发 4096/5120 时把 `--max-cudagraph-capture-size` 提到 640/768；作者明确表示不确定这是否是 Pareto 数字的必要条件，因此不能把它写成已证明的独立优化收益。
- `--mamba-ssm-cache-dtype bfloat16` 用于提高 decode endpoint 的有效 cache 容量。
- `--language-model-only` 除关闭多模态输入外，还解锁文本 attention 层的 fused QK-norm + RoPE + gate 路径。
- prefix caching 在随机数据集上关闭；这不是对真实重复前缀流量的通用建议。

## PR 关系与不要混淆的支线

```text
FlashInfer #3001 -> FlashInfer #3155 -> vLLM #40717

vLLM #35758 -> #36687 -> #37310
                         -> #37416 -> #37635 -> #41869

vLLM #45357 --------------------------+
                                      +-> race-free async scheduling
vLLM #48481 --------------------------+
```

| PR | 状态 | 为什么不列入最终主表 |
| --- | --- | --- |
| [vLLM #45096](https://github.com/vllm-project/vllm/pull/45096) | Closed, not merged | 早期把两类 async race 合在一起处理；维护者后来拆分为 #45357 与最终的 #48481 |
| [vLLM #47373](https://github.com/vllm-project/vllm/pull/47373) | Closed, not merged | allocation zeroing race 的前一版；最终实现由 #48481 重新集成并合入 |

## 证据边界

- **已核对**：博客正文、PR 状态、merge commit、变更文件、PR 描述中的正确性/性能数据，以及公开复现配方。
- **未执行**：本地或集群上的 GB200 NVL72 serving 复现、逐 PR A/B、GSM8K 复跑、NIXL/RDMA trace 和 Nsight profiling。
- **不能推导**：不能把 kernel microbenchmark 的 5.92x、prefill-only 的 1.13x、HMA 小规模测试的 7% 和 recipe 的 8% 直接相乘；它们的工作负载、基线和瓶颈不同。
- **准确表述**：25K 是指定 nightly、模型、拓扑、请求长度和高并发范围下的公开系统结果，不是所有 Qwen3.5 尺寸、GPU、latency SLA 或默认 vLLM 配置的通用吞吐承诺。

## 来源

- [vLLM Qwen3.5 25K TPS/GPU blog](https://vllm.ai/blog/2026-08-06-qwen35-25k-tps)
- [vLLM hybrid SSM disaggregation blog](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg)
- [NIXL disaggregation roadmap #33702](https://github.com/vllm-project/vllm/issues/33702)
- [Qwen3.5 GB200 P/D recipes](https://github.com/NVIDIA/srt-slurm-recipes/tree/main/recipes/multi-node/Qwen3.5/GB200/8k1k/vllm/disagg)
- [FlashInfer #3001](https://github.com/flashinfer-ai/flashinfer/pull/3001)、[FlashInfer #3155](https://github.com/flashinfer-ai/flashinfer/pull/3155)
- [vLLM #35758](https://github.com/vllm-project/vllm/pull/35758)、[#36687](https://github.com/vllm-project/vllm/pull/36687)、[#37310](https://github.com/vllm-project/vllm/pull/37310)、[#37416](https://github.com/vllm-project/vllm/pull/37416)、[#37635](https://github.com/vllm-project/vllm/pull/37635)
- [vLLM #40717](https://github.com/vllm-project/vllm/pull/40717)、[#41869](https://github.com/vllm-project/vllm/pull/41869)、[#45357](https://github.com/vllm-project/vllm/pull/45357)、[#48481](https://github.com/vllm-project/vllm/pull/48481)
