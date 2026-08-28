# Qwen3.8-Flash-Next：SGLang 技术点与代码 PR 映射

来源：[Qwen3.8-Flash-Next: Day-0 Support in SGLang](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next)

GitHub 状态检查时间：2026-08-28（Asia/Shanghai）。

## 结论

文章中的 SGLang 实现并不是“每个技术点对应一个独立 PR”。模型定义、Qwen Sparse Attention（QSA）、IndexShare MTP、Gated Residual / HyperConnection、Per-Layer Embedding（PLE）、Pinned-Host Offload 和相关测试，主要集中在一个大型模型支持 PR：

- [sglang#36497: Introduce Qwen 3.8 Flash Next](https://github.com/sgl-project/sglang/pull/36497)

HyperConnection Mix 使用的 Blackwell Split-K BF16 GEMM 基础算子来自：

- [flashinfer#4266: Add Blackwell CuTeDSL BF16 Split-K dense GEMM](https://github.com/flashinfer-ai/flashinfer/pull/4266)

截至检查时间，#36497 仍是 Open，且 GitHub 显示与 `main` 冲突。因此“Day-0 Support”当前主要指官方镜像、Cookbook 和该 PR 分支上的可用实现，并不表示所有代码已经进入 SGLang `main`。

## 模型架构概览

Qwen3.8-Flash-Next 是 Qwen4 架构的早期预览版本，主体是 125B 参数 MoE，另有 51.2B 参数的 N-gram Embedding，每个 token 激活约 6B 参数。

| 组成 | 配置 |
| --- | --- |
| Transformer 层数 | 48 |
| GDN 线性注意力层 | 36 |
| QSA 稀疏注意力层 | 12 |
| 排布 | 每 4 层包含 3 个 GDN 层和 1 个 QSA 层 |
| MoE experts | 512 |
| 每 token 路由 | Top-10 experts |
| Residual 分支 | 4 个 HyperConnection 分支 |
| PLE | 8 个 bigram heads + 8 个 trigram heads |

SGLang 复用已有的 Qwen3.5 Gated DeltaNet 实现，在其上增加 Qwen4-Exp 模型、QSA 全注意力层、PLE 和四分支 Gated Residual：

- [`qwen4_exp.py:1388`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/models/qwen4_exp.py#L1388-L1584)：GDN 和 QSA decoder layer 注册。
- [`qwen4_exp.py:1221`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/models/qwen4_exp.py#L1221-L1274)：PLE 与 Attention/MoE 两组 HyperConnection 的组装。
- [`qwen4_exp.py:1284`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/models/qwen4_exp.py#L1284-L1319)：单路 hidden state 扩展为四路 residual，以及 Mix/Combine 调用顺序。

## 技术点与代码映射

### 1. GDN + QSA 混合注意力

36 个 Gated DeltaNet 层使用固定大小的 recurrent state 压缩历史；12 个 QSA 层保留随序列增长的原始 Attention K/V。因此模型级 KV 节省主要来自“只有 1/4 的层保存完整 K/V”，不是来自在 QSA 层内部丢弃原始 K/V。

对应实现：

- [`qwen4_exp.py:1388`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/models/qwen4_exp.py#L1388-L1454)：线性注意力层和全注意力层分别继承 Qwen3.5 对应实现。
- [`qwen4_exp.py:1576`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/models/qwen4_exp.py#L1576-L1584)：`linear_attention` 和 `full_attention` 的 layer type 映射。
- 主 PR：[sglang#36497](https://github.com/sgl-project/sglang/pull/36497)。

### 2. QSA：Retrieve Coarsely, Attend Precisely

QSA 有两条数据路径：

1. 轻量 indexer 在压缩索引 Key 上定位重要上下文。
2. Sparse GQA 根据逻辑 token index，从原始 Attention K/V cache 中读取数据并完成最终 softmax/value aggregation。

当前模型配置的关键参数为：

| 参数 | 数值 | 含义 |
| --- | ---: | --- |
| Index Query heads | 4 | 四个索引 Query head |
| Index KV heads | 1 | 一个共享 Key head，形成 MQA scorer |
| Index head dim | 128 | 索引向量维度 |
| Compression ratio | 4 | 每 4 个原始 index keys 压缩成一个 block key |
| Block top-k | 512 | 选择 512 个压缩 block |
| Token budget | 2048 | 展开后选择 512 × 4 个原始 token |
| Final maximum | 2051 | 2048 加当前未完成 block 的最多 3 个 token |

索引分数对应：

\[
s_{t,b}=\frac{1}{\sqrt{128}}\sum_{h=1}^{4}
\operatorname{ReLU}(\langle q^I_{t,h},\bar{k}^I_b\rangle)
\]

代码入口：

- [`qsa_indexer.py:43`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/layers/attention/qsa/qsa_indexer.py#L43-L119)：indexer 配置、Query/Key projection、block top-k 计算。
- [`mqa.py:39`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/layers/attention/qsa/mqa.py#L39-L58)：FP32 dot、ReLU、跨四个 Query heads 求和。
- [`qsa_indexer.py:226`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/layers/attention/qsa/qsa_indexer.py#L226-L263)：mean、Gemma RMSNorm、MRoPE 和 compressed-cache store 的融合路径。
- [`kernel.py:95`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/layers/attention/qsa/kernel.py#L95-L149)：压缩 block index 展开为原始 token index，并补齐未完成 block 的 tail。
- [`qwen_sparse_attn_backend.py:1397`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py#L1397-L1499)：Prefill sparse GQA。
- [`qwen_sparse_attn_backend.py:1552`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py#L1552-L1752)：Decode 阶段的 K/V compaction、TRTLLM-Gen 和 FlashAttention fallback。
- 主 PR：[sglang#36497](https://github.com/sgl-project/sglang/pull/36497)。

### 3. QSA Cache 与 Radix Cache

QSA 为每 4 个 token 额外保存一个 BF16 compressed index key。原始 Attention K/V 仍保存在普通 paged KV pool 中。

缓存管理的关键设计是：

```text
compressed_slot = full_slot // compress_ratio
```

只要 full-KV page size 是 compression ratio 的整数倍，每个四-token compression group 就不会跨 page。Compressed cache 因而能直接跟随 full-KV allocator 和 Radix Cache 的 page ownership，无需第二套生命周期管理。

未凑满四个 token 的原始 index keys 不按 token 长期保存，只进入每请求四槽 ring：

```text
ring_slot = req_pool_idx * compress_ratio + position % compress_ratio
```

对应实现：

- [`qsa_kv_pool.py:22`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/mem_cache/qsa_kv_pool.py#L22-L53)：compressed slot 与 Radix Cache ownership 设计。
- [`qsa_kv_pool.py:128`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/mem_cache/qsa_kv_pool.py#L128-L181)：per-request pending ring 和 compressed cache allocation。
- [`overrides.py:1348`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/arg_groups/overrides.py#L1348-L1398)：compressed QSA 强制 `page_size=64`，并自动处理 PLE offload 默认值。
- 主 PR：[sglang#36497](https://github.com/sgl-project/sglang/pull/36497)。

### 4. IndexShare MTP

普通 MTP speculative decoding 的每轮迭代包含一次 draft-extend 和多个 draft-decode forward。如果每个 draft step 都重新运行 QSA indexer，长上下文下 indexer 会成为主要成本。

IndexShare 的处理流程是：

```text
target verify / accepted tokens
        ↓
draft-extend：运行一次 QSA indexer
        ↓
保存每个 request 最后一个 accepted row 的 sparse indices
        ↓
draft decode step 1..N：跳过 indexer
        ↓
复用 frozen indices，并追加本轮已经 draft 出来的 token positions
```

代码入口：

- [`qwen_sparse_attn_backend.py:113`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py#L113-L187)：持久化共享 buffer、capture 和 lookup。
- [`qwen_sparse_attn_backend.py:1243`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py#L1243-L1369)：draft-extend 捕获最后 accepted row，decode lookup。
- [`qwen4_exp.py:1456`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/models/qwen4_exp.py#L1456-L1492)：决定运行、捕获或复用 indexer。
- [`eagle_worker_v2.py:358`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/speculative/eagle_worker_v2.py#L358-L423)：IndexShare buffer 初始化和 backend 注入。
- 主 PR：[sglang#36497](https://github.com/sgl-project/sglang/pull/36497)。

当前实现限制：

- 只支持 chain speculation，即 `speculative_eagle_topk=1`。
- `speculative_num_steps` 必须大于 1。
- Adaptive speculative decoding 会关闭该共享路径。

### 5. Gated Residual / HyperConnection

传统单路 residual 被扩展为四路：

```text
4 × hidden residual
        │
        ├─ Mix：动态读四路 → 1 × hidden → Attention / MoE
        │
        └─ Combine：Attention / MoE 输出按四个动态系数写回四路
```

参考实现中的 Mix：

```python
input_mix_weight = F.silu(
    F.linear(hyper_input_normed, input_mix_weight_down) / hc
)
input_mix_weight = F.linear(input_mix_weight, input_mix_weight_up)
input_mix_weight = torch.sigmoid(input_mix_weight)
output = (
    input_mix_weight.unflatten(-1, (hc, hs))
    * hyper_input_normed.unflatten(-1, (hc, hs))
).mean(dim=-2)
```

Combine：

```python
inject = 2 * torch.sigmoid(
    F.linear(normed_residual, block_inject_weight) / hc
)
output = residual + block_output.unsqueeze(-2) * inject.unsqueeze(-1)
```

代码入口：

- [`hyperconnection.py:117`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/layers/hyperconnection.py#L117-L231)：Mix 和 Combine 数学参考实现。
- [`hyperconnection.py:233`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/layers/hyperconnection.py#L233-L338)：按 shape 和设备选择具体 kernel。
- 主 PR：[sglang#36497](https://github.com/sgl-project/sglang/pull/36497)。

### 6. HyperConnection Mix Kernel

低 M 场景的瓶颈是 GEMM 的 M 维并行度不足。SGLang 使用 Blackwell CuTe Split-K，将 K 维分给多个 CTA，并把 SiLU、Sigmoid、gate 和四分支 reduction 放进两个 GEMM epilogue。

up-projection 权重会离线重排，让同一个 hidden element 的四个 branch gate 在 tile 内相邻：

- [`hc_mix.py:24`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/kernels/ops/elementwise/hc_mix.py#L24-L34)：权重重排和 padding。
- [`hc_mix.py:72`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/kernels/ops/elementwise/hc_mix.py#L72-L120)：两个 fused Split-K GEMM。
- [flashinfer#4266](https://github.com/flashinfer-ai/flashinfer/pull/4266)：通用 Blackwell BF16 Split-K GEMM，已于 2026-08-06 合并。
- [sglang#36497](https://github.com/sgl-project/sglang/pull/36497)：HC 专用集成和 fused epilogue。

博客称 `M ≤ 16` 使用 Split-K，但 #36497 当前代码的 dispatch 条件是 `M ≤ 24`：

- [`hyperconnection.py:247`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/layers/hyperconnection.py#L247-L268)

部署和复现实验应以所使用 commit 的代码为准。

### 7. HyperConnection Combine Kernel

Combine 要计算四个 injection coefficient，再更新四路 residual：

- 大 M：一个 fused kernel 完成每个 token row。
- `M ≤ 32`：沿 hidden dimension 将一行切成多个 CTA；第一个 kernel 计算 FP32 partial gate，第二个 kernel 完成归约和 residual update。

对应实现：

- [`hc_combine.py:49`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/kernels/ops/elementwise/hc_combine.py#L49-L93)：单 kernel 路径。
- [`hc_combine.py:96`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/kernels/ops/elementwise/hc_combine.py#L96-L136)：小 M split 路径。
- [`hyperconnection.py:291`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/layers/hyperconnection.py#L291-L338)：`M ≤ 32` dispatch。
- 主 PR：[sglang#36497](https://github.com/sgl-project/sglang/pull/36497)。

### 8. PLE / N-gram Embedding

模型在配置编号 2、即零基第 1 个 decoder block 中放置一个 PLE：

- 8 个 bigram hash heads 使用 `(x[t-1], x[t])`。
- 8 个 trigram hash heads 使用 `(x[t-2], x[t-1], x[t])`。
- 每个 head 返回 160 个值。
- 16 × 160 拼接为 2560 维 embedding。

代码由配置自动推出这些维度：

```text
ngram_heads = (ngram_size - 1) * heads_per_ngram
            = (3 - 1) * 8
            = 16

head_dim_per_ngram = ple_embed_dim / ngram_heads
                   = 2560 / 16
                   = 160
```

Hash embedding 经过 K/V projection 后，以四路 residual hidden state 为 Query 计算门控；gated value 再经过 grouped norm、dilated depthwise convolution 和 SiLU，最终作为 PLE delta 写入四路 residual。

对应实现：

- [`qwen4_exp.py:422`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/models/qwen4_exp.py#L422-L506)：N-gram heads、hash vocabulary 和 embedding table。
- [`qwen4_exp.py:603`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/models/qwen4_exp.py#L603-L656)：bigram/trigram hash。
- [`qwen4_exp.py:872`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/models/qwen4_exp.py#L872-L946)：PLE projection、norm 和 depthwise convolution。
- [`qwen4_exp.py:1162`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/models/qwen4_exp.py#L1162-L1218)：PLE gate、short conv 和 delta。
- 主 PR：[sglang#36497](https://github.com/sgl-project/sglang/pull/36497)。

### 9. PLE Sparse Pinned-Host Offload

51.2B PLE 参数在 BF16 下约占 95.4 GiB，但每个 token 只访问 16 行。因此 SGLang 将每个 TP rank 的 vocabulary shard 放在 pinned host memory，只把命中的行 gather 到一个很小的 BF16 GPU buffer。

数据流：

```text
生成 16 个 hash row IDs
        ↓
Triton UVA kernel 直接读取 pinned CPU table
        ↓
得到 BF16 GPU rows
        ↓
TP reduce / DP gather-scatter
        ↓
PLE projection、gate、short conv
```

一个独立 CUDA stream 在前一个 decoder block 执行时启动 gather，从而将 Host-to-GPU 访问与模型计算重叠。

代码入口：

- [`qwen4_exp.py:718`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/models/qwen4_exp.py#L718-L869)：Pinned-host table、UVA gather 和 TP reduce。
- [`qwen4_exp.py:1106`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/models/qwen4_exp.py#L1106-L1160)：独立 stream 上的异步 prefetch 和消费。
- [`qwen4_exp.py:1621`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/models/qwen4_exp.py#L1621-L1665)：在 PLE 前一个 decoder layer 执行前启动 prefetch。
- [`qwen4_exp_mtp.py:41`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/models/qwen4_exp_mtp.py#L41-L48)：单层 MTP draft model 明确关闭 PLE。
- 主 PR：[sglang#36497](https://github.com/sgl-project/sglang/pull/36497)。

## 文章中的性能数据

以下数字是文章给出的特定硬件和配置结果，不能直接外推到其他 GPU、上下文长度或并发度。

| 优化 | 结果 |
| --- | ---: |
| HC Mix，B300，`M=4` | 12.36 µs → 6.03 µs，2.05× kernel speedup |
| HC Mix 端到端 speculative decode | 吞吐提升 7.6% |
| HC Combine，`M=4` | 4.17 µs → 2.13 µs，1.96× kernel speedup |
| HC Combine 小 M 端到端 | 吞吐提升 5.49% |
| HC Combine 大 M | 相对 cuBLAS baseline 最高 2.54×，有效带宽 6144 GB/s |
| PLE offload，H200 TP4 | 每 GPU target weights 83.91 → 60.45 GiB |
| PLE offload 后 KV capacity | 1.84M → 3.28M tokens，+78.54% |
| PLE offload matched throughput | 并发 1/2/4 的几何平均变化 -0.07% |
| NVFP4，B200 TP4，MTP | Batch 1 decode 540 tok/s，accept length 3.3（含 bonus token） |

## 核心 PR 状态

| PR | 状态（2026-08-28） | 作用 |
| --- | --- | --- |
| [sglang#36497](https://github.com/sgl-project/sglang/pull/36497) | Open；与 `main` 冲突 | Qwen3.8-Flash-Next 总模型支持 PR，包含本文绝大多数实现。 |
| [flashinfer#4266](https://github.com/flashinfer-ai/flashinfer/pull/4266) | Merged 2026-08-06 | Blackwell BF16 Split-K GEMM，HC Mix 的底层基础算子。 |
| [sglang#36496](https://github.com/sgl-project/sglang/pull/36496) | Merged 2026-08-26 | Qwen3.8-Flash-Next Cookbook。 |
| [sglang#36499](https://github.com/sgl-project/sglang/pull/36499) | Merged 2026-08-26 | Cookbook 源码安装路径指向 #36497。 |
| [sglang#36611](https://github.com/sgl-project/sglang/pull/36611) | Merged 2026-08-27 | 修复 H200 上 BF16 SSM state 与 MTP verify backend 的 Cookbook 配置。 |

## 发布后的后续 PR

这些 PR 不属于文章正文的原始核心映射，但决定实际硬件、KV dtype 和部署方式是否可用。

| PR | 状态（2026-08-28） | 内容与依赖关系 |
| --- | --- | --- |
| [sglang#36806](https://github.com/sgl-project/sglang/pull/36806) | Merged into `qwen4-main-squashed` | 精确识别 SM120，将 RTX 50 / RTX PRO 6000 路由到已验证的 FlashInfer sparse decode；尚未因 #36497 未合入而进入 `main`。 |
| [sglang#36845](https://github.com/sgl-project/sglang/pull/36845) | Open；stacked on #36497 | SM121 / GB10 使用独立 Triton packed-varlen fallback，避免 FlashInfer 长上下文静默错误和 FA4 编译失败。 |
| [sglang#36644](https://github.com/sgl-project/sglang/pull/36644) | Open；stacked on #36497 | QSA 的 FP8 KV cache prefill、chunked prefill、decode dequant/scale 支持。 |
| [sglang#36651](https://github.com/sgl-project/sglang/pull/36651) | Open；stacked on #36497 | PD disaggregation 传输 PLE short-conv/N-gram state、QSA pending ring 和 compressed index K。 |
| [sglang#36567](https://github.com/sgl-project/sglang/pull/36567) | Open；stacked on #36497 | 从 NVMe 按需流式读取 PLE rows，目标是 128 GiB unified-memory 系统；当前初始路径为 TP1 + FP8 PLE。 |
| [sglang#36601](https://github.com/sgl-project/sglang/pull/36601) | Open；依赖 #36497 | ROCm gfx950/gfx942 的 QSA graph replay、AITER 和 BF16/FP8 验证。 |
| [sglang#36786](https://github.com/sgl-project/sglang/pull/36786) | Open | MXFP4/NVFP4 compressed-tensors checkpoint 中，target 量化但 MTP 保持 BF16 时的权重加载修复。 |

早期的合并式 SM120/SM121 方案 [sglang#36556](https://github.com/sgl-project/sglang/pull/36556) 仍是 Open 且冲突；后续证据表明 SM120 与 SM121 不能共用同一 decode 路径，因此目前更应关注按架构拆分的 #36806 和 #36845。

## 建议阅读顺序

1. 先读 [`qwen4_exp.py:1388`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/models/qwen4_exp.py#L1388-L1584)，建立 GDN/QSA 和四路 residual 的整体执行结构。
2. 再读 [`qsa_indexer.py:43`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/layers/attention/qsa/qsa_indexer.py#L43-L119) 与 [`kernel.py:95`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/layers/attention/qsa/kernel.py#L95-L149)，理解 QSA 如何从 block selection 变成 token selection。
3. 读 [`qwen_sparse_attn_backend.py:1397`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py#L1397-L1752)，看 sparse selection 如何真正驱动原始 K/V attention。
4. 读 [`qsa_kv_pool.py:22`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/mem_cache/qsa_kv_pool.py#L22-L181)，理解 QSA 与 Radix Cache 的生命周期耦合。
5. 读 [`eagle_worker_v2.py:358`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/speculative/eagle_worker_v2.py#L358-L423)，串起 IndexShare MTP。
6. 最后阅读 [`hyperconnection.py:117`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/layers/hyperconnection.py#L117-L338) 和 [`qwen4_exp.py:422`](https://github.com/sgl-project/sglang/blob/73a255206f916366c8d26d4022f82ddfb0ab558d/python/sglang/srt/models/qwen4_exp.py#L422-L1218)，分别深入 HyperConnection 与 PLE。

## 参考链接

- [LMSYS 原文](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next)
- [Qwen3.8-Flash-Next 官方仓库](https://github.com/QwenLM/Qwen3.8-Flash-Next)
- [Qwen3.8-Flash-Next SGLang Cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)
- [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
