# SGLang、vLLM、TensorRT-LLM 的 MLA / DSA API、实现与 PR 脉络

> 快照日期：2026-09-01
>
> 代码基线：SGLang `8a191554`、vLLM `d0e695a9`、TensorRT-LLM `f04859d0`。
>
> 算子库基线：FlashInfer `0b79dc1b`、FlashMLA `15f13e50`、DeepGEMM `559d79fb`。
>
> 范围：以 DeepSeek-V2/V3/R1 的 MLA，以及 DeepSeek-V3.2/GLM 系列采用的 DeepSeek Sparse Attention（DSA）为主。DeepSeek-V4 的稀疏注意力在三家代码中通常有独立后端或元数据路径，本文只在接口边界处提及，不把它和 V3.2 DSA 混为一谈。

## 结论先行

MLA 和 DSA 不是同一层面的能力：

- **MLA（Multi-head Latent Attention）** 是注意力与 KV Cache 的基础结构。它把 K/V 压缩为低维 latent，在 decode 阶段通过 weight absorption 避免显式还原完整多头 K/V，核心收益是减少 KV Cache 容量和显存带宽。
- **DSA（DeepSeek Sparse Attention）** 构建在 MLA 之上。它增加一个轻量 indexer，对历史 token 打分并选出 Top-K，再只对这些 token 执行 sparse MLA。典型 `index_topk` 是 2048；短序列或 Top-K 接近总长度时通常回退到 dense/full MHA。
- 三家的主流程相同：`MLA 投影/压缩 KV → indexer Q/K → paged MQA logits → Top-K → 逻辑位置映射到 paged KV 地址 → sparse MLA → 输出投影`。区别主要在于配置入口、后端选择、调度与 CUDA Graph 集成方式。
- 上游算子库不是三选一：DeepGEMM 主要计算 indexer logits，FlashMLA 主要计算最终 sparse/dense MLA，FlashInfer 则同时覆盖 MLA、paged logits、Top-K 和地址 remap，正在形成更完整的一站式算子面。
- backend 选择必须把 **SM100/103 与 SM120/121 分开**：它们虽然都属于 Blackwell，但 FA4、TRTLLM-gen、CUTLASS/CuTeDSL MLA 等路径通常以 SM100/103 为目标；SM120/121 在 dense MLA 上更多依赖 Triton/FlashInfer fallback，在 DSA 上则依赖模型、KV format 和私有算子的专门闭环。

| 维度 | SGLang | vLLM | TensorRT-LLM |
| --- | --- | --- | --- |
| 用户入口 | CLI 为主；`--attention-backend`，DSA 另有 prefill/decode/indexer/topk 子后端 | `AttentionConfig`、`--attention-backend`、`-ac.*`；DSA 通常由模型配置自动启用 | `LLM(..., sparse_attention_config=DeepSeekSparseAttentionConfig(...))` |
| MLA 后端组织 | `flashinfer`、`flashmla`、`trtllm_mla`、`cutedsl_mla`、`cutlass_mla`、FA3/FA4 等 | 统一 `AttentionBackendEnum`，后端声明 `use_mla` / `use_sparse` 能力 | 从模型结构自动构造 `MLAParams`，用户选择总体 `attn_backend` 与 KV dtype |
| DSA 后端组织 | `dsa` 是一级后端，下面分别选择 sparse prefill、decode、paged-MQA-logits、Top-K | sparse MLA 是统一 attention backend 的一种；没有独立公开的 `--dsa` 开关 | 显式、可序列化的 DSA 参数对象，下沉为 `DSAParams` 和 sparse-attention metadata |
| 可调粒度 | 三家中最容易显式混搭不同 kernel | 统一 selector 和 capability validation 最完整 | indexer/topk/短序列/分块等算法参数最丰富 |
| 当前边界 | 后端组合多，兼容性依赖 GPU、dtype、page size | 很多 DSA 参数来自模型 HF config，用户侧更偏自动选择 | DSA 公共配置仍标记为 prototype，并且只支持 PyTorch backend |
| 适合场景 | 快速接入/比较 kernel，跨 NVIDIA/AMD 后端实验 | 标准化 serving、统一调度与多后端自动选择 | NVIDIA 平台深度优化、piecewise CUDA Graph、量化 indexer/cache |

## 1. 公共数据流与实现边界

### 1.1 Dense MLA

以 decode 为例，MLA 不为每个 attention head 保存完整 K/V，而是保存压缩后的 latent KV 和单独的 RoPE key。运行时常见两条路径：

1. **Absorbed/MQA-like decode**：把 K/V 的升维权重吸收到 Q 和输出投影中，attention kernel 直接读取压缩 KV Cache；这是 MLA 降低 decode 带宽的关键。
2. **Unabsorbed/dense MHA prefill**：prefill 的计算形态更适合一次还原/计算完整 K/V，再走 FlashAttention 类 kernel。因此三家都逐渐形成“prefill 与 decode 后端分开选择”的设计。

工程上需同时处理 latent KV、RoPE、page/block layout、FP8/FP4 cache、chunked prefill、context/decode parallel 和 CUDA Graph；所以“支持 DeepSeek 模型”不等于已经覆盖所有 MLA 优化路径。

### 1.2 DSA on MLA

DSA 在原 MLA KV Cache 之外维护一份更窄、通常量化的 **indexer K cache**：

```text
hidden states
  ├─ MLA Q / compressed-KV / RoPE projections ───────────────┐
  └─ indexer Q/K/head-weight                                  │
       └─ paged/ragged MQA logits                             │
            └─ Top-K token positions                          │
                 └─ request-local → global/page remap         │
                      └─ sparse MLA reads selected latent KV ─┘
                           └─ up/output projection
```

性能热点因此从一个 attention kernel 扩展为四部分：

- indexer Q/K projection 和量化 index-K 写 cache；
- paged/ragged MQA logits，即当前 query 对全部历史 index-K 的扫描；
- Top-K 选择及跨层复用、跨 TP/CP rank 的索引处理；
- sparse MLA 对选中 token 的 gather 和 attention。

短序列走 dense/full MHA 不是语义回退，而是性能优化：当 `kv_len <= topk` 或 gather/indexer 的固定开销更高时，稀疏路径没有收益。

## 2. SGLang

### 2.1 用户 API

SGLang 的 [attention backend guide](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/attention_backend.mdx) 把 MLA 与 DSA 的选择集中在 server CLI。未显式指定时，框架会按 GPU 架构、KV dtype、模型和 phase 自动选择。

MLA 的主要入口：

| 参数 | 功能 |
| --- | --- |
| `--attention-backend` | 全局后端；MLA 可选项包括 `flashinfer`、`flashmla`、`trtllm_mla`、`cutedsl_mla`、`tokenspeed_mla`、`cutlass_mla`、`fa3`、`fa4` 等 |
| `--prefill-attention-backend` | 只覆盖 prefill，可和 decode 后端混搭 |
| `--decode-attention-backend` | 只覆盖 decode |
| `--kv-cache-dtype` | 控制 latent KV Cache 的 BF16/FP8 等格式；会影响可用 kernel |

例如在支持的 NVIDIA GPU 上可显式组合 dense MLA 后端：

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-R1 \
  --attention-backend trtllm_mla \
  --prefill-attention-backend fa4 \
  --kv-cache-dtype fp8_e4m3
```

DSA 的一级入口是 `--attention-backend dsa`。历史名称 `nsa` 仍保留兼容，但已经 deprecated：

| 参数 | 功能 |
| --- | --- |
| `--dsa-prefill-backend` | sparse prefill kernel，如 `flashmla_sparse`、`flashmla_sparse_q8`、`flashmla_kv`、`flashmla_auto`、`flashinfer_sparse_mla`、`fa3`、`trtllm`、`tilelang`、`aiter` |
| `--dsa-decode-backend` | sparse decode kernel，候选集合与硬件支持相关 |
| `--dsa-paged-mqa-logits-backend` | indexer 扫描后端：`auto`、`deepgemm`、`cutedsl`、`aiter` |
| `--dsa-topk-backend` | Top-K 实现：`sgl-kernel`、`torch`、`flashinfer` |
| `--enable-dsa-cache-layer-split` | 在 prefill CP 场景将 DSA GPU KV/indexer cache layer 分散到不同 rank；当前有 transfer/PP 约束 |

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.2 \
  --attention-backend dsa \
  --dsa-prefill-backend flashmla_sparse \
  --dsa-decode-backend fa3 \
  --dsa-paged-mqa-logits-backend deepgemm \
  --dsa-topk-backend sgl-kernel
```

生产部署通常应先让 auto policy 选择；只有做基准测试、固定 kernel，或明确知道 GPU/dtype/page-size 兼容矩阵时再手工覆盖。

### 2.2 内部实现

关键代码入口：

| 层次 | 代码 | 大致功能 |
| --- | --- | --- |
| 参数层 | [`server_args.py`](https://github.com/sgl-project/sglang/blob/8a191554e379741048ecfac02cea334eb2b883e0/python/sglang/srt/server_args.py#L1657) | 定义全局、phase-specific 和 DSA 子后端参数 |
| 自动策略 | [`overrides.py`](https://github.com/sgl-project/sglang/blob/8a191554e379741048ecfac02cea334eb2b883e0/python/sglang/srt/arg_groups/overrides.py#L675) | 按 Hopper/Blackwell/AMD、BF16/FP8 和模型选择默认后端 |
| 注册分发 | [`attention_registry.py`](https://github.com/sgl-project/sglang/blob/8a191554e379741048ecfac02cea334eb2b883e0/python/sglang/srt/layers/attention/attention_registry.py#L50) | 将 backend 名称映射到 MLA、DSA 或普通 attention backend 类 |
| MLA 模型层 | [`deepseek_v2.py`](https://github.com/sgl-project/sglang/blob/8a191554e379741048ecfac02cea334eb2b883e0/python/sglang/srt/models/deepseek_v2.py#L1734) | `DeepseekV2AttentionMLA`，负责 MLA 投影、cache 与 backend 调用 |
| DSA 总后端 | [`dsa_backend.py`](https://github.com/sgl-project/sglang/blob/8a191554e379741048ecfac02cea334eb2b883e0/python/sglang/srt/layers/attention/dsa_backend.py#L289) | 构造 prefill/decode phase implementation、metadata 和 Top-K backend |
| DSA indexer | [`dsa_indexer.py`](https://github.com/sgl-project/sglang/blob/8a191554e379741048ecfac02cea334eb2b883e0/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L202) | indexer 投影、paged-MQA logits、Top-K、索引 remap/cache |

SGLang 的特点是“**一级 DSA backend + 多个可替换子 backend**”。`DeepseekSparseAttnBackend` 负责统一 phase metadata 和 dispatch；FlashMLA/FA3/TRTLLM/TileLang/AITER 等只实现具体 sparse attention 接口。这样便于快速接 kernel，但组合数量多，正确性和性能都依赖显式的 capability check。

### 2.3 里程碑 PR

下表是功能演进主线，不是全部相关 PR：

| 时间 | PR | 功能描述 |
| --- | --- | --- |
| 2024-08 | [#905](https://github.com/sgl-project/sglang/pull/905) | DeepSeek-V2 MLA 初始 Triton 实现，建立 compressed latent attention 路径 |
| 2024-12 | [#2349](https://github.com/sgl-project/sglang/pull/2349) | MLA prefill 不做 weight absorption；形成 dense prefill / absorbed decode 分工 |
| 2025-02～04 | [#3550](https://github.com/sgl-project/sglang/pull/3550)、[#4831](https://github.com/sgl-project/sglang/pull/4831)、[#5390](https://github.com/sgl-project/sglang/pull/5390) | 接入 FlashInfer MLA、FA3、Blackwell CUTLASS MLA |
| 2025-04 | [#5052](https://github.com/sgl-project/sglang/pull/5052) | 用统一 `attention_backend` 替代单独的 FlashInfer-MLA 开关 |
| 2025-07～08 | [#8632](https://github.com/sgl-project/sglang/pull/8632)、[#8638](https://github.com/sgl-project/sglang/pull/8638) | TRTLLM-gen MLA decode 与 FP8 路径 |
| 2025-10 | [#11061](https://github.com/sgl-project/sglang/pull/11061)、[#11194](https://github.com/sgl-project/sglang/pull/11194) | DeepSeek-V3.2/DSA 初始支持和高性能 Top-K |
| 2025-11 | [#11892](https://github.com/sgl-project/sglang/pull/11892)、[#12065](https://github.com/sgl-project/sglang/pull/12065) | 短序列自适应 MHA；DSA context parallel 初始实现 |
| 2026-01～02 | [#13959](https://github.com/sgl-project/sglang/pull/13959)、[#16758](https://github.com/sgl-project/sglang/pull/16758)、[#18389](https://github.com/sgl-project/sglang/pull/18389) | CP/FP8 cache 优化；TRTLLM sparse BF16 与 FP8/NVFP4 路径 |
| 2026-04～05 | [#21502](https://github.com/sgl-project/sglang/pull/21502)、[#21783](https://github.com/sgl-project/sglang/pull/21783)、[#22851](https://github.com/sgl-project/sglang/pull/22851) | IndexCache、TRTLLM sparse prefill、可配置 Top-K backend |
| 2026-05 | [#25821](https://github.com/sgl-project/sglang/pull/25821) | 用户与代码术语从 NSA 统一重命名为 DSA，保留旧别名兼容 |
| 2026-06～07 | [#27705](https://github.com/sgl-project/sglang/pull/27705)、[#29421](https://github.com/sgl-project/sglang/pull/29421) | 融合 indexer Q/K 路径；DSA cache layer split |
| 2026-07 | [#30514](https://github.com/sgl-project/sglang/pull/30514)、[#31888](https://github.com/sgl-project/sglang/pull/31888) | Q8KV8/FP8 sparse prefill 及 GLM/DeepSeek 共享路径优化 |

## 3. vLLM

### 3.1 用户 API

vLLM 的 [attention backend design](https://github.com/vllm-project/vllm/blob/main/docs/design/attention_backends.md) 使用统一 backend selector。CLI 和 Python 都能选择后端：

```bash
vllm serve deepseek-ai/DeepSeek-V3.2 \
  --attention-backend FLASHMLA_SPARSE \
  -ac.mla_prefill_backend=FLASH_ATTN \
  -ac.indexer_kv_dtype=fp8
```

```python
from vllm import LLM
from vllm.config import AttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

llm = LLM(
    model="deepseek-ai/DeepSeek-V3.2",
    attention_config=AttentionConfig(
        backend=AttentionBackendEnum.FLASHMLA_SPARSE,
        mla_prefill_backend="FLASH_ATTN",
        indexer_kv_dtype="fp8",
    ),
)
```

核心参数：

| 参数 | 功能 |
| --- | --- |
| `AttentionConfig.backend` / `--attention-backend` | 统一选择 dense MLA 或 sparse MLA 后端 |
| `backend_per_kind` | 按 attention kind 覆盖后端，用于混合模型 |
| `mla_prefill_backend` / `-ac.mla_prefill_backend` | MLA prefill 单独选择；decode 仍使用主 backend |
| `indexer_kv_dtype` | DSA index-K cache 的 `auto`/`bf16`/`fp8`/`mxfp4`/`nvfp4` |
| `sparse_mla_force_mqa` | 调试/兼容用，强制 sparse MLA 的 MQA 路径 |

与 SGLang 不同，vLLM 没有独立的公开 `--dsa` 开关。DeepSeek-V3.2 模型层根据 HF config 构造 `MLAAttention(use_sparse=True, indexer=...)`，selector 再从声明支持 `use_mla=True`、`use_sparse=True` 的 backend 中选择。也就是说：**是否是 DSA 主要由模型决定，具体 sparse kernel 才由 backend 配置决定**。

vLLM 还提供独立的 [IndexCache](https://github.com/vllm-project/vllm/blob/main/docs/features/index_cache.md) 模型覆盖项，让部分 DSA 层复用之前计算的 Top-K：

```bash
vllm serve deepseek-ai/DeepSeek-V3.2 \
  --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}'
```

### 3.2 内部实现

| 层次 | 代码 | 大致功能 |
| --- | --- | --- |
| 公共配置 | [`attention.py`](https://github.com/vllm-project/vllm/blob/d0e695a91b67a8214c2e9ed77595186d2f2844b6/vllm/config/attention.py#L21) | `AttentionConfig`，统一 dense/sparse MLA 和 indexer cache 参数 |
| 后端枚举 | [`registry.py`](https://github.com/vllm-project/vllm/blob/d0e695a91b67a8214c2e9ed77595186d2f2844b6/vllm/v1/attention/backends/registry.py#L36) | 注册 `FLASHMLA`、`FLASHINFER_MLA`、`FLASHMLA_SPARSE`、`FLASH_ATTN_MLA_SPARSE` 等 |
| 选择器 | [`selector.py`](https://github.com/vllm-project/vllm/blob/d0e695a91b67a8214c2e9ed77595186d2f2844b6/vllm/v1/attention/selector.py#L20) | 用 `use_mla`、`use_sparse`、dtype、head size、block size、compute capability 验证候选 |
| MLA 公共层 | [`mla_attention.py`](https://github.com/vllm-project/vllm/blob/d0e695a91b67a8214c2e9ed77595186d2f2844b6/vllm/model_executor/layers/attention/mla_attention.py#L392) | compressed cache、absorbed decode、dense prefill、DCP/chunk 等统一逻辑 |
| DSA indexer | [`sparse_attn_indexer.py`](https://github.com/vllm-project/vllm/blob/d0e695a91b67a8214c2e9ed77595186d2f2844b6/vllm/model_executor/layers/sparse_attn_indexer.py#L729) | 自定义 op，负责 paged/ragged MQA logits、Top-K 与 buffer 管理 |
| 模型接线 | [`deepseek_v2.py`](https://github.com/vllm-project/vllm/blob/d0e695a91b67a8214c2e9ed77595186d2f2844b6/vllm/model_executor/models/deepseek_v2.py#L633) | `DeepseekV32IndexerCache`、`Indexer` 和 MLA attention 在同一模型文件接线，并按 pattern 决定哪些层运行或复用 Top-K |
| sparse kernel | [`flashmla_sparse.py`](https://github.com/vllm-project/vllm/blob/d0e695a91b67a8214c2e9ed77595186d2f2844b6/vllm/v1/attention/backends/mla/flashmla_sparse.py#L1) | FlashMLA sparse backend 的格式、GPU、block/head-size 约束及调用 |

vLLM 的实现重点是“**一个 attention layer 抽象 + capability-driven selector**”。同一个 `MLAAttention` 同时覆盖 dense/sparse、prefill/decode、DCP、量化 cache；不同 kernel backend 只暴露它能处理的组合。CUDA 平台再给 Hopper、Blackwell、SM120 分别维护候选优先级。

### 3.3 里程碑 PR

| 时间 | PR | 功能描述 |
| --- | --- | --- |
| 2024-06 | [#4650](https://github.com/vllm-project/vllm/pull/4650) | DeepSeek-V2 初始模型和 MLA 支持 |
| 2025-01～02 | [#12528](https://github.com/vllm-project/vllm/pull/12528)、[#12639](https://github.com/vllm-project/vllm/pull/12639)、[#13789](https://github.com/vllm-project/vllm/pull/13789) | absorbed decode 优化、chunked prefill、MLA 迁入 V1 engine |
| 2025-02～03 | [#13867](https://github.com/vllm-project/vllm/pull/13867)、[#14770](https://github.com/vllm-project/vllm/pull/14770) | FlashMLA V1；避免完整 K/V materialization |
| 2025-04～07 | [#16032](https://github.com/vllm-project/vllm/pull/16032)、[#17625](https://github.com/vllm-project/vllm/pull/17625)、[#20034](https://github.com/vllm-project/vllm/pull/20034) | Blackwell CUTLASS MLA、后端接线、FlashInfer ragged prefill |
| 2025-09 | [#21078](https://github.com/vllm-project/vllm/pull/21078)、[#23734](https://github.com/vllm-project/vllm/pull/23734) | TRTLLM-gen MLA decode kernel；MLA decode context parallel |
| 2025-09～10 | [#25896](https://github.com/vllm-project/vllm/pull/25896)、[#25999](https://github.com/vllm-project/vllm/pull/25999)、[#26763](https://github.com/vllm-project/vllm/pull/26763) | DeepSeek-V3.2/DSA 初始实现、indexer prefill chunk、Top-K 优化 |
| 2025-12～2026-02 | [#27532](https://github.com/vllm-project/vllm/pull/27532)、[#33451](https://github.com/vllm-project/vllm/pull/33451)、[#33680](https://github.com/vllm-project/vllm/pull/33680) | FP8 sparse prefill；FlashInfer sparse MLA；decode Top-K 加速 |
| 2026-03～04 | [#37252](https://github.com/vllm-project/vllm/pull/37252)、[#37421](https://github.com/vllm-project/vllm/pull/37421)、[#37735](https://github.com/vllm-project/vllm/pull/37735) | FP8 默认 sparse backend、CUDAGraph-safe Top-K、IndexCache |
| 2026-06～07 | [#46189](https://github.com/vllm-project/vllm/pull/46189)、[#47327](https://github.com/vllm-project/vllm/pull/47327)、[#48770](https://github.com/vllm-project/vllm/pull/48770) | Hopper FA sparse MLA、短序列 dense MHA、masked-MHA sparse prefill |
| 2026-09 | [#51724](https://github.com/vllm-project/vllm/pull/51724) | W4A16 DSA 路径，继续扩大量化模型覆盖 |

## 4. TensorRT-LLM

### 4.1 用户 API

TensorRT-LLM 的 dense MLA 参数主要从模型 checkpoint/config 推导，例如 `q_lora_rank`、`kv_lora_rank`、qk RoPE/nope dim、v head dim；用户通常只选择总体 `attn_backend`、KV Cache dtype、chunked prefill 和 graph policy。

DSA 则提供显式配置对象。官方 [sparse attention guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/sparse-attention.md) 的最小形式是：

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import DeepSeekSparseAttentionConfig

llm = LLM(
    model="deepseek-ai/DeepSeek-V3.2",
    backend="pytorch",
    attn_backend="TRTLLM",
    sparse_attention_config=DeepSeekSparseAttentionConfig(
        index_topk=2048,
        indexer_k_dtype="fp8",
        skip_indexer_for_short_seqs=True,
    ),
)
```

也可以在 YAML 中使用带判别字段的配置：

```yaml
sparse_attention_config:
  algorithm: dsa
  index_topk: 2048
  indexer_k_dtype: fp8
  skip_indexer_for_short_seqs: true
```

`DeepSeekSparseAttentionConfig` 的主要控制项：

| 参数 | 功能 |
| --- | --- |
| `index_n_heads`、`index_head_dim`、`index_topk` | indexer 结构和稀疏预算 |
| `indexer_max_chunk_size` | 长 prefill 的 indexer chunk 上限 |
| `skip_indexer_for_short_seqs` | 短序列跳过 indexer，直接 full attention |
| `use_cute_dsl_paged_mqa_logits` | 使用 CuTe DSL 实现 indexer paged MQA logits |
| `use_cute_dsl_topk` | 使用 CuTe DSL Top-K |
| `q_split_threshold` | 大 query 数量时切分计算的阈值 |
| `enable_heuristic_topk` | 启用 GVR/heuristic Top-K；当前代码校验 `{512, 1024, 2048}` |
| `indexer_k_dtype` | index-K cache 的 `fp8` 或 `fp4` |
| `index_share_for_mtp_iteration` | MTP iteration 间复用 index 结果 |

需要注意两层“backend”：外层 `backend="pytorch"` 是 TensorRT-LLM LLM API 的执行后端；内层 `attn_backend="TRTLLM"` 是 PyTorch runtime 里采用 TRTLLM attention kernels。当前 DSA config 的 `supports_backend()` 只接受前者为 PyTorch，尚不是 TensorRT engine backend 的通用能力。

### 4.2 内部实现

| 层次 | 代码 | 大致功能 |
| --- | --- | --- |
| 公共配置 | [`llm_args.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/f04859d04de68b255343a6eb4a7d9d0bbaf82ca8/tensorrt_llm/llmapi/llm_args.py#L944) | `DeepSeekSparseAttentionConfig` 验证参数，并下沉为 `DSAParams` |
| MLA 参数 | [`interface.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/f04859d04de68b255343a6eb4a7d9d0bbaf82ca8/tensorrt_llm/_torch/attention_backend/interface.py#L1104) | `MLAParams` 描述 latent ranks、RoPE/nope/v dims、投影与量化信息 |
| DSA attention | [`backend.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/f04859d04de68b255343a6eb4a7d9d0bbaf82ca8/tensorrt_llm/_torch/attention_backend/sparse/dsa/backend.py#L44) | `DSATrtllmAttention` 管理完整 indexer 层与共享 Top-K 层，输出 global topk |
| DSA indexer | [`indexer.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/f04859d04de68b255343a6eb4a7d9d0bbaf82ca8/tensorrt_llm/_torch/attention_backend/sparse/dsa/indexer.py#L596) | Q/K projection、FP8/FP4 index cache、DeepGEMM/CuTeDSL logits 和 Top-K |
| 稀疏模块 | [`module.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/f04859d04de68b255343a6eb4a7d9d0bbaf82ca8/tensorrt_llm/_torch/attention_backend/sparse/dsa/module.py#L68) | context/generation 分流、短序列 MHA、Hopper/Blackwell sparse MLA 和 piecewise graph 拆分 |
| 稀疏框架 | [`registry.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/f04859d04de68b255343a6eb4a7d9d0bbaf82ca8/tensorrt_llm/_torch/attention_backend/sparse/registry.py#L1) | 将 DSA、DSV4 等算法接入共享 sparse-attention lifecycle/metadata 接口 |

TensorRT-LLM 的特点是“**配置对象 → sparse framework → NVIDIA 专用 kernel/graph 路径**”。它不强调让用户自由组合每个 attention 子后端，而是把 indexer、cache manager、metadata、hook、piecewise CUDA Graph 和具体 GPU kernel 组合成受控实现。

### 4.3 里程碑 PR

| 时间 | PR | 功能描述 |
| --- | --- | --- |
| 2024-12 | [#2629](https://github.com/NVIDIA/TensorRT-LLM/pull/2629) | DeepSeek-V3 初始支持，建立 MLA 路径 |
| 2025-04～05 | [#3190](https://github.com/NVIDIA/TensorRT-LLM/pull/3190)、[#3571](https://github.com/NVIDIA/TensorRT-LLM/pull/3571)、[#3752](https://github.com/NVIDIA/TensorRT-LLM/pull/3752) | Hopper/Blackwell FP8 MLA、KV reuse、QMMA MLA kernel |
| 2025-06～07 | [#4467](https://github.com/NVIDIA/TensorRT-LLM/pull/4467)、[#4651](https://github.com/NVIDIA/TensorRT-LLM/pull/4651)、[#5713](https://github.com/NVIDIA/TensorRT-LLM/pull/5713) | piecewise CUDA Graph、Blackwell chunked prefill、Hopper context MLA |
| 2025-10 | [#8405](https://github.com/NVIDIA/TensorRT-LLM/pull/8405)、[#8692](https://github.com/NVIDIA/TensorRT-LLM/pull/8692) | DeepSeek-V3.2 BF16/FP8/NVFP4 和初始 DSA；TRTLLM-gen sparse MLA kernel |
| 2025-11 | [#8882](https://github.com/NVIDIA/TensorRT-LLM/pull/8882)、[#9376](https://github.com/NVIDIA/TensorRT-LLM/pull/9376) | 自定义 DSA Top-K；V3.2 MLA chunked prefill |
| 2026-03 | [#11677](https://github.com/NVIDIA/TensorRT-LLM/pull/11677)、[#11871](https://github.com/NVIDIA/TensorRT-LLM/pull/11871)、[#12503](https://github.com/NVIDIA/TensorRT-LLM/pull/12503) | 短序列 MHA、长序列 token-parallel indexer、DSA op 为 piecewise graph 拆分 |
| 2026-05～07 | [#13340](https://github.com/NVIDIA/TensorRT-LLM/pull/13340)、[#15574](https://github.com/NVIDIA/TensorRT-LLM/pull/15574)、[#16420](https://github.com/NVIDIA/TensorRT-LLM/pull/16420) | Blackwell FP4 indexer、跨层 Top-K 共享、CuTe DSL GVR Top-K E2E |
| 2026-08 | [#12733](https://github.com/NVIDIA/TensorRT-LLM/pull/12733)、[#16224](https://github.com/NVIDIA/TensorRT-LLM/pull/16224)、[#16666](https://github.com/NVIDIA/TensorRT-LLM/pull/16666) | 统一 sparse-attention framework、SM120 FlashInfer sparse MLA、GVR writeback overlap |
| 2026-08～09 | [#18383](https://github.com/NVIDIA/TensorRT-LLM/pull/18383)、[#18391](https://github.com/NVIDIA/TensorRT-LLM/pull/18391) | native next-N DSA paged MQA；batch cross-layer index remap |

## 5. 不同 SM 上 prefill / decode MLA backend 的可选组合

本节只描述上述固定 commit 已经进入主干的能力。记号 `P → D` 表示“prefill backend → decode backend”。“可选”表示框架 selector/实现已经有对应路径，不表示任意模型 shape、dtype、page size、TP/DCP、speculative decoding 组合都经过生产验证。

硬件分组不能只写 Hopper/Blackwell：

| SM | 典型 GPU | 需要单列的原因 |
| --- | --- | --- |
| SM80/86 | A100/A30、A10 | 没有 FA3/Blackwell TRTLLM-gen；dense MLA 主要依靠 FlashInfer/Triton/FA2 fallback |
| SM89 | L40S、RTX 4090 | 同属 8.x，但不是 Ampere；当前 sparse MLA 专用 kernel 覆盖仍和 SM80 类似 |
| SM90 | H100/H200/H20 | FA3、FlashMLA 和 Hopper DSA 的主阵地 |
| SM100/103 | B200/GB200、B300/GB300 | datacenter Blackwell；TRTLLM-gen、FA4、CUTLASS/CuTeDSL MLA 的主要目标 |
| SM120/121 | RTX 50/RTX PRO 6000 Blackwell、GB10 | consumer/workstation Blackwell；不能假定 SM100 kernel 可直接运行，三框架都有单独 fallback/实现 |

### 5.1 SGLang：dense MLA

SGLang 的全局 backend 会同时供两个 phase 使用；设置 `--prefill-attention-backend` 或 `--decode-attention-backend` 后，未设置的一侧继承 `--attention-backend`。当前 [MLA support matrix](https://github.com/sgl-project/sglang/blob/8a191554e379741048ecfac02cea334eb2b883e0/docs/docs/advanced_features/attention_backend.mdx#L188) 与 [auto selector](https://github.com/sgl-project/sglang/blob/8a191554e379741048ecfac02cea334eb2b883e0/python/sglang/srt/arg_groups/model_override_base.py#L273) 对应关系如下：

| SM | auto（dense MLA） | 当前有实际意义的显式 `P → D` 组合 | 主要限制 |
| --- | --- | --- | --- |
| SM80/86、SM89 | `triton → triton` | `flashinfer → flashinfer`；`triton → triton` | FlashInfer MLA 为 page 1、当前不接 FP8 KV；Triton 是可移植 fallback，但不支持 chunked prefix cache |
| SM90 | `fa3 → fa3`（CUDA 12.3+） | `fa3 → flashmla`、`flashinfer → flashmla`、`flashinfer → flashinfer`、`triton → triton` | FlashMLA 强制 page 64；FA3 不接 FP8/FP4 MLA KV；FlashInfer 为 page 1 |
| SM100/103 | 普通 MLA 为 `flashinfer → flashinfer`；DeepSeek-V3/R1 模型覆盖为 `trtllm_mla → trtllm_mla` | `fa4 → trtllm_mla`、`trtllm_mla → cutedsl_mla`、`tokenspeed_mla → tokenspeed_mla`、`flashinfer → cutlass_mla`、`flashinfer → flashmla` | TRTLLM/CuTeDSL/TokenSpeed page 32/64；CuTeDSL 只做 decode，未给 prefill 时自动补 `trtllm_mla`；TokenSpeed 只接受 FP8 KV；CUTLASS page 128 |
| SM120/121 | `triton → triton` | 高置信通用路径是 `flashinfer → flashinfer` 或 `triton → triton`；R1 shape + FP8 可试 `tokenspeed_mla → tokenspeed_mla` | 不要照搬 SM100 的 FA4/CUTLASS/CuTeDSL 组合。`trtllm_mla` 的 CLI 校验按广义 Blackwell 接受 SM12x，但实际 TRTLLM-gen kernel/build 仍需单独确认，不列为默认推荐 |

SM100/103 最典型的 phase split 是：

```bash
python -m sglang.launch_server \
  --model-path nvidia/DeepSeek-R1-FP4 \
  --attention-backend trtllm_mla \
  --prefill-attention-backend fa4 \
  --page-size 64
```

这里 prefill 走未吸收的 FA4 dense MHA，decode 走读取 latent KV 的 TRTLLM MLA。若把 decode 改为 `cutedsl_mla`，prefill 应显式保留 `trtllm_mla`；源码也会在缺省时自动完成这一补齐，并拒绝把 CuTeDSL 用作 prefill。[page/dtype/SM 校验](https://github.com/sgl-project/sglang/blob/8a191554e379741048ecfac02cea334eb2b883e0/python/sglang/srt/arg_groups/overrides.py#L1127)

### 5.2 SGLang：DSA / sparse MLA

DSA 使用另一组 phase 参数 `--dsa-prefill-backend` / `--dsa-decode-backend`。下面的 auto 组合直接来自 [`_dsa_split_backend_resolution`](https://github.com/sgl-project/sglang/blob/8a191554e379741048ecfac02cea334eb2b883e0/python/sglang/srt/arg_groups/overrides.py#L669)：

| SM / KV dtype | auto `sparse P → sparse D` | 可选或动态分支 | 结论 |
| --- | --- | --- | --- |
| SM80/86、SM89 | auto policy 会落到 Hopper 风格名称，但主干没有相应 SM8x 专用 sparse MLA 闭环 | 无可靠生产组合 | 应视为未完整支持；vLLM 的 Triton sparse MLA 与 FlashMLA A100 支持也仍在 open PR，不要把 CLI choice 当作可运行能力 |
| SM90 + BF16 KV | `flashmla_sparse → fa3` | `flashmla_sparse → flashmla_sparse`；短 prefill 可动态走 FA3 dense MHA | 官方默认；page 64 |
| SM90 + FP8 KV | `flashmla_kv → flashmla_kv` | `flashmla_sparse_q8 → flashmla_kv`；`flashmla_auto` 会按 KV dtype/shape选 BF16 gather 或 FP8 path | `flashmla_sparse_q8` 是 SM90、prefill-only；不能放到 decode 一侧 |
| SM100/103 + BF16 KV | `flashmla_sparse → trtllm` | sparse 两侧都可用 `flashmla_sparse`；短 prefill 可动态走 TRTLLM ragged dense MHA | decode 默认 TRTLLM sparse MLA |
| SM100/103 + FP8 KV | `trtllm → trtllm` | `flashmla_kv` 可用于支持的 head/shape；短 prefill仍可转 TRTLLM ragged MHA | TRTLLM 是默认完整 prefill/decode 组合 |
| SM120/121 + GLM DSA + FP8 KV | `flashinfer_sparse_mla → flashinfer_sparse_mla` | 无通用交叉组合；构造时强制两侧都是该 backend | 这是模型、dtype、SM 三重特化；当前只接受 GLM DSA，不等同于通用 DeepSeek-V3.2 SM120 支持 |
| SM120/121 + 通用 DeepSeek-V3.2 | 无本文认可的已验证默认组合 | open PR 正在补 SM120 sparse prefill/fallback | 不能把 SM100 的 `trtllm` 默认机械外推到 SM120；需要按具体 wheel/kernel 做启动与精度验证 |

DSA 还有一个容易被忽略的“第三条 prefill 路径”：当序列较短、未启用相冲突的 CP/graph 模式时，SGLang 会绕过 Top-K sparse MLA，SM90 用 FA3 dense MHA，SM100/103 用 TRTLLM ragged dense MHA；SM120 不在该 one-shot allowlist 中。[短序列分流代码](https://github.com/sgl-project/sglang/blob/8a191554e379741048ecfac02cea334eb2b883e0/python/sglang/srt/layers/attention/dsa_backend.py#L3301)

### 5.3 vLLM：dense MLA

vLLM 的两侧名称来自不同枚举：`AttentionConfig.backend`/`--attention-backend` 主要决定 decode/paged backend；`mla_prefill_backend` 只决定 dense MLA prefill 子后端。两者可以自由配对，但各自先经过 capability validation。[CUDA decode priority](https://github.com/vllm-project/vllm/blob/d0e695a91b67a8214c2e9ed77595186d2f2844b6/vllm/platforms/cuda.py#L82)；[prefill selector](https://github.com/vllm-project/vllm/blob/d0e695a91b67a8214c2e9ed77595186d2f2844b6/vllm/v1/attention/backends/mla/prefill/selector.py#L48)

| SM | auto `dense P → dense D` | 可选 decode backend | prefill 可选项与限制 |
| --- | --- | --- | --- |
| SM80/86、SM89 | `FLASH_ATTN(FA2) → TRITON_MLA` | `TRITON_MLA` | prefill 只有 `FLASH_ATTN` selector 路径；专用 FlashMLA/FlashInfer decode capability check 不接受 SM8x |
| SM90 | `FLASH_ATTN(FA3) → FLASH_ATTN_MLA` | `FLASHMLA`、`TRITON_MLA` | prefill 为 `FLASH_ATTN`；`FLASH_ATTN_MLA` 只接受 BF16/FP16 KV，FlashMLA 还可覆盖 FP8 KV、page 64 |
| SM100/103，DeepSeek R1/V3 shape | `FLASH_ATTN(FA4) → FLASHINFER_MLA` | `TOKENSPEED_MLA`、`CUTLASS_MLA`、`FLASHMLA`、`TRITON_MLA` | prefill 还可选 `TRTLLM_RAGGED`、`FLASHINFER`、`TOKENSPEED_MLA`；FlashInfer prefill只接受标准 DeepSeek `(128,64,128)` 维度 |
| SM100/103，Kimi-K3 类 `(192,64,256)` | `TRTLLM_RAGGED → FLASHINFER_MLA` | `CUTLASS_MLA`、`FLASHMLA`、`TRITON_MLA` | selector 对该 shape 把 `TRTLLM_RAGGED` 放在 FA4 前；TokenSpeed 与 FlashInfer prefill不支持该 prefill shape |
| SM120/121 | `FLASH_ATTN(FA2 fallback) → TRITON_MLA` | `TRITON_MLA` | 当前 FA4 capability gate 不含 12.x，dense decode priority 也只列 Triton；不要将 SM100 的 FlashInfer/CUTLASS decode 外推到 SM120 |

例如固定 SM100 的 FA4 prefill、FlashInfer/TRTLLM-gen decode：

```bash
vllm serve deepseek-ai/DeepSeek-R1 \
  --attention-backend FLASHINFER_MLA \
  -ac.mla_prefill_backend=FLASH_ATTN
```

若把 decode 换成 `CUTLASS_MLA`，block size 必须为 128；`FLASHINFER_MLA`、`TOKENSPEED_MLA` 为 32/64，`FLASHMLA` 为 64。相关限制在各 backend 的 `supports_compute_capability`、KV dtype 和 block-size 声明中统一验证。[backend registry](https://github.com/vllm-project/vllm/blob/d0e695a91b67a8214c2e9ed77595186d2f2844b6/vllm/v1/attention/backends/registry.py#L65)

### 5.4 vLLM：DSA / sparse MLA

DSA 层仍由主 `backend` 同时拥有 sparse prefill 与 decode；没有 SGLang 式公开 `sparse prefill backend A → sparse decode backend B`。`mla_prefill_backend` 只控制 backend 决定走 dense-MHA prefill 时的子路径，不替换 Top-K sparse MLA kernel。

| SM / KV dtype | sparse prefill + decode backend | dense/短 prefill 分支 | 主要限制 |
| --- | --- | --- | --- |
| SM80/86、SM89 | 主干无可用 sparse MLA backend | 无 | `TRITON_MLA_SPARSE` 仍是 open PR，当前应判定 DSA 未完整支持 |
| SM90 + BF16/FP16 KV | auto `FLASH_ATTN_MLA_SPARSE`；可选 `FLASHMLA_SPARSE` | `FLASH_ATTN`/FA3 dense MHA | FA sparse 不接 FP8 KV，也暂不支持 DCP |
| SM90 + FP8/packed KV | `FLASHMLA_SPARSE` | 支持条件满足时仍可用 FA3 dense MHA | FlashMLA 同时实现 sparse prefill/decode，page 64 |
| SM100/103 + BF16 KV | `FLASHINFER_MLA_SPARSE` 与 `FLASHMLA_SPARSE` 之间按每 rank Q-head 数选择：`<=16` 优先 FlashInfer，否则优先 FlashMLA | R1 shape 通常 FA4；Kimi shape 优先 TRTLLM ragged | FlashMLA 为 page 64；FlashInfer 为 32/64 |
| SM100/103 + FP8 KV | auto 优先 `FLASHINFER_MLA_SPARSE`，不支持其 cache 变体时回落 `FLASHMLA_SPARSE` | 同上 | `FLASHINFER_MLA_SPARSE` 不接受 `fp8_ds_mla`，FlashMLA 接受 packed `fp8_ds_mla`/NVFP4 组合 |
| SM120/121 | `FLASHINFER_MLA_SPARSE_SM120` 同时处理 prefill/decode | **没有** dense-MHA prefill 路径，始终走 Top-K MQA | BF16 query、FP8/`fp8_ds_mla` KV、`index_topk=2048`，block 64/256，并要求安装的 FlashInfer 含 SM120 private op |

上述 SM100 稀疏优先级可直接在 [`cuda.py`](https://github.com/vllm-project/vllm/blob/d0e695a91b67a8214c2e9ed77595186d2f2844b6/vllm/platforms/cuda.py#L93) 看到；SM120 的严格 dtype/topk gate 在 [`flashinfer_mla_sparse.py`](https://github.com/vllm-project/vllm/blob/d0e695a91b67a8214c2e9ed77595186d2f2844b6/vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py#L132)。DeepSeek-V4 使用 `*_DSV4` 的模型专用 backend，不应并入这张 V3.2 DSA 表。

### 5.5 TensorRT-LLM：dense MLA

TensorRT-LLM 的稳定用户面没有公开、独立的 prefill/decode backend 字段。`attn_backend="TRTLLM"` 进入一个有序 FMHA library registry，运行时按 phase/shape/dtype 选择；`TLLM_FMHA_LIBS` 可改变库顺序，但属于开发/调试级 override。所以下表写的是**实际内部组合**，不是两个可自由组合的 LLM API 参数。[FMHA registry](https://github.com/NVIDIA/TensorRT-LLM/blob/f04859d04de68b255343a6eb4a7d9d0bbaf82ca8/tensorrt_llm/_torch/attention_backend/fmha/registry.py#L28)；[phase selector](https://github.com/NVIDIA/TensorRT-LLM/blob/f04859d04de68b255343a6eb4a7d9d0bbaf82ca8/tensorrt_llm/_torch/attention_backend/trtllm.py#L1825)

| SM | `attn_backend="TRTLLM"` 的实际 `P → D` | 可替代的公开外层 backend | 说明 |
| --- | --- | --- | --- |
| SM80/86、SM89 | TRTLLM fused/thop fallback → 同一 fallback | `FLASHINFER` | 可运行性更依赖具体 DeepSeek 配置与构建，缺少 SM90/SM100 专用优化 |
| SM90 | TRTLLM context MLA → TRTLLM generation；FP8 generation 内部优先 FlashMLA kernel | `FLASHINFER` | 用户不直接选择 FlashMLA；它是 TRTLLM 内部 generation kernel |
| SM100/103 | TRTLLM context MLA fallback → CuTeDSL（命中性能区间时）/TRTLLM-gen/THOP fallback | `FLASHINFER` | FlashInfer TRTLLM-gen 明确拒绝 MLA context，因此 prefill 不走该库；CuTeDSL 也只接受 decode-only batch |
| SM120/121 | TRTLLM fused/thop fallback → 同一 fallback | `FLASHINFER` | SM100/103 TRTLLM-gen 与 CuTeDSL registry gate 都拒绝 SM12x；若选择外层 FlashInfer，则使用其 ragged/paged prefill 和 planned MLA decode |

因此 TensorRT-LLM 不能用类似 SGLang 的参数直接表达“FA4 prefill + CuteDSL decode”。SM100/103 上接近的行为由 registry 自动形成：context 走 TRTLLM fallback，decode 在满足 head、batch、dtype、page 和非 tree-spec 条件时先试 CuTeDSL，否则试 TRTLLM-gen，再回落 THOP。[TRTLLM-gen gate](https://github.com/NVIDIA/TensorRT-LLM/blob/f04859d04de68b255343a6eb4a7d9d0bbaf82ca8/tensorrt_llm/_torch/attention_backend/fmha/flashinfer_trtllm_gen.py#L493)；[CuTeDSL gate](https://github.com/NVIDIA/TensorRT-LLM/blob/f04859d04de68b255343a6eb4a7d9d0bbaf82ca8/tensorrt_llm/_torch/attention_backend/fmha/cute_dsl_mla.py#L43)

### 5.6 TensorRT-LLM：DSA / sparse MLA

DSA 只支持外层 `attn_backend="TRTLLM"`；`attn_backend="FLASHINFER"` 的 sparse registry 当前不注册 DSA。最终 sparse MLA phase 组合由 SM 和 KV format 固定：

| SM / KV dtype | 实际 `sparse P → sparse D` | 短序列分支 | 结论 |
| --- | --- | --- | --- |
| SM80/86、SM89 | 无主干生产组合 | 无 | 当前 DSA 高性能路径从 SM90 起 |
| SM90 + BF16 KV | `FlashMLA sparse prefill → FlashMLA sparse decode` | 可通过 `TRTLLM_MLA_SHORT_SEQ_MHA_THRESHOLD` 启用 dense MHA | 模块直接调用 bundled `flash_mla_sparse_fwd`，head 数补到 64 的倍数 |
| SM100/103 + BF16/FP8/NVFP4 路径 | `TRTLLM absorbed sparse context → TRTLLM sparse generation` | 可启用短序列 dense MHA | indexer 可用 DeepGEMM/CuTeDSL，最终 sparse attention 由 TRTLLM fused op 消费 Top-K；不是 FlashInfer TRTLLM-gen dense FMHA |
| SM120/121 + `fp8_ds_mla` | `FlashInfer SM120 sparse MLA → FlashInfer SM120 sparse MLA` | 不启用短序列 MHA | 同一个私有 SM120 op 服务 DSA 与 DSV4；缺少 `fp8_ds_mla` 或 pinned FlashInfer symbol 会启动失败 |

Hopper/Blackwell 分流可见 [`dsa/module.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/f04859d04de68b255343a6eb4a7d9d0bbaf82ca8/tensorrt_llm/_torch/attention_backend/sparse/dsa/module.py#L68)；SM120 专用库只接受 `dsa`/`deepseek_v4`、SM120/121 与 `fp8_ds_mla`。[`flashinfer_sparse_mla.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/f04859d04de68b255343a6eb4a7d9d0bbaf82ca8/tensorrt_llm/_torch/attention_backend/fmha/flashinfer_sparse_mla.py#L25)

### 5.7 横向选择速查

| 目标 | SGLang | vLLM | TensorRT-LLM |
| --- | --- | --- | --- |
| SM90 dense MLA | 默认 FA3；可显式 `FA3/FlashInfer → FlashMLA` | `FLASH_ATTN(FA3) → FLASH_ATTN_MLA/FLASHMLA` | TRTLLM 自动；FP8 decode 内部用 FlashMLA |
| SM100 dense MLA | 最多显式组合：FA4/TRTLLM/FlashInfer prefill × TRTLLM/CuTeDSL/TokenSpeed/CUTLASS/FlashMLA decode | prefill 子枚举 × 统一 decode enum，capability validation 最严格 | registry 自动组合，稳定 API 不暴露 phase split |
| SM90 DSA | BF16 默认 FlashMLA sparse → FA3；FP8 默认 FlashMLA KV 两侧 | BF16 FA sparse/FlashMLA，FP8 FlashMLA；一个 sparse backend 同时管两侧 | BF16 FlashMLA 两侧 |
| SM100 DSA | BF16 `FlashMLA sparse → TRTLLM`；FP8 TRTLLM 两侧 | FlashInfer sparse/FlashMLA sparse 自动竞争 | TRTLLM fused sparse 两侧 |
| SM120 DSA | 当前明确特化为 GLM FP8 + FlashInfer 两侧；通用 V3.2 仍需谨慎 | FlashInfer SM120 两侧，限制最清楚 | FlashInfer SM120 两侧，强制 `fp8_ds_mla` |
| SM8x DSA | 不完整 | 不完整；等待 Triton sparse backend | 不支持 |

当前 open PR 会改变表中两个明显空洞：SGLang [#32779](https://github.com/sgl-project/sglang/pull/32779) 在补 SM90/SM120 fused sparse prefill，vLLM [#38476](https://github.com/vllm-project/vllm/pull/38476) 在补跨 SM8x/11x/12x 的 `TRITON_MLA_SPARSE`；FlashMLA [#183](https://github.com/deepseek-ai/FlashMLA/pull/183) 则在补 A100 dense MLA decode。它们在合并并通过框架 capability gate 前，不计入上面的“当前可选”。

## 6. FlashInfer、FlashMLA、DeepGEMM 算子 API 与演进

三者不是同类替代品。按 DSA pipeline 拆开后，职责关系更清楚：

| DSA/MLA 阶段 | FlashInfer | FlashMLA | DeepGEMM |
| --- | --- | --- | --- |
| RoPE、latent/index cache 写入 | `mla_rope_quantize_fp8`、`concat_mla_k` 等辅助算子 | 约定输入 cache layout，本身不负责完整模型投影 | 通用 FP8/FP4 GEMM；`fp8_gemm_nt_skip_head_mid` 可把左右结果直接写入带 per-head 中间空段的投影布局 |
| Dense MLA | planned `BatchMLAPagedAttentionWrapper`；TRTLLM-gen/CuTeDSL/XQA functional decode | `flash_mla_with_kvcache`；dense varlen MHA prefill API | 不提供最终 attention |
| DSA indexer logits | 2026-08 新增 `fp8_paged_mqa_logits` / `fp4_paged_mqa_logits` | 不提供 indexer logits | `fp8_fp4_mqa_logits` 与 paged 版本是主要能力 |
| Top-K 与索引 remap | `top_k`、`top_k_page_table_transform`、`top_k_ragged_transform`、`top_k_varlen` | 不提供 | 不提供；只产出 logits |
| Sparse MLA | TRTLLM-gen sparse MLA、SM120 sparse MLA、DSV4 专用入口 | `flash_mla_with_kvcache(indices=...)` decode；`flash_mla_sparse_fwd` prefill | 不提供最终 sparse attention |

因此常见组合不是三选一，而是：

```text
DeepGEMM/FlashInfer paged-MQA logits
  → FlashInfer/sgl-kernel/TRTLLM Top-K + page remap
  → FlashMLA/FlashInfer/TRTLLM sparse MLA
```

### 6.1 FlashInfer

#### 公开算子 API

FlashInfer 的 MLA/DSA API 已经从单个 decode wrapper 扩展为 attention、indexer logits、Top-K/remap 和 cache 辅助算子的组合面：

| API | 所在层 | 功能与当前边界 |
| --- | --- | --- |
| [`BatchMLAPagedAttentionWrapper`](https://github.com/flashinfer-ai/flashinfer/blob/0b79dc1b09f24632179e0d4d231c3a981d5556d5/flashinfer/mla/_batch_mla/_wrapper.py#L104) | Dense MLA | `plan()` + `run()` 生命周期，面向 absorbed decode/incremental prefill；统一 FA2、FA3、CUTLASS backend。当前推荐 `MLAPlanMetadata` 和结构化 `query=`/`kv_cache=`，旧 flat metadata、位置参数和 split tensor 调用已 deprecated |
| [`trtllm_batch_decode_with_kv_cache_mla`](https://github.com/flashinfer-ai/flashinfer/blob/0b79dc1b09f24632179e0d4d231c3a981d5556d5/flashinfer/mla/_core.py#L2872) | Dense/sparse decode | TRTLLM-gen 风格 functional API；`sparse_mla_top_k > 0` 时接收选中 KV 索引并进入 sparse MLA，dense 时可在 TRTLLM-gen/CuTeDSL/XQA 等路径间选择 |
| [`trtllm_batch_decode_sparse_mla_dsv4`](https://github.com/flashinfer-ai/flashinfer/blob/0b79dc1b09f24632179e0d4d231c3a981d5556d5/flashinfer/mla/_core.py#L1454) | DSV4 sparse decode | 处理 DSV4 的 compressed/SWA 双 segment、NoPE 等独立语义；不应当直接当作 V3.2 DSA API |
| [`compute_paged_mqa_logits_schedule`](https://github.com/flashinfer-ai/flashinfer/blob/0b79dc1b09f24632179e0d4d231c3a981d5556d5/flashinfer/attn_scores/attn_scores.py#L866) | Indexer metadata | 在 GPU 上生成 paged-MQA-logits CTA schedule，可传预分配 `out` 以配合 CUDA Graph |
| [`fp8_paged_mqa_logits`](https://github.com/flashinfer-ai/flashinfer/blob/0b79dc1b09f24632179e0d4d231c3a981d5556d5/flashinfer/attn_scores/attn_scores.py#L1022) | DSA indexer | FP8 Q/index-K 的 weighted-ReLU MQA logits；当前公开实现面向 Blackwell SM100/SM103，支持显式 schedule/output buffer |
| [`fp4_paged_mqa_logits`](https://github.com/flashinfer-ai/flashinfer/blob/0b79dc1b09f24632179e0d4d231c3a981d5556d5/flashinfer/attn_scores/attn_scores.py#L1323) | DSA indexer | MXFP4 Q/index-K 版本；同样面向 SM100/SM103，head/scale/`next_n` 组合比 FP8 更受限 |
| `precompile_paged_mqa_logits` | JIT 生命周期 | 预编译常用 FP8/FP4 静态配置，避免服务首请求触发 CuTe DSL 编译 |
| [`top_k`](https://github.com/flashinfer-ai/flashinfer/blob/0b79dc1b09f24632179e0d4d231c3a981d5556d5/flashinfer/topk.py#L855) | Top-K | 在 radix、SM100 clusters 和 CUB backend 间自动选择；支持 deterministic、tie-break、`dsa_graph_safe` |
| [`top_k_page_table_transform`](https://github.com/flashinfer-ai/flashinfer/blob/0b79dc1b09f24632179e0d4d231c3a981d5556d5/flashinfer/topk.py#L1053) | Top-K + remap | 融合 Top-K 与 logical-index → physical-page/token 地址转换，直接生成 sparse MLA 可消费的 int32 索引 |
| [`top_k_ragged_transform`](https://github.com/flashinfer-ai/flashinfer/blob/0b79dc1b09f24632179e0d4d231c3a981d5556d5/flashinfer/topk.py#L1299) | Top-K + remap | ragged KV 的融合 Top-K 与 per-row offset 变换 |
| [`mla_rope_quantize_fp8`](https://github.com/flashinfer-ai/flashinfer/blob/0b79dc1b09f24632179e0d4d231c3a981d5556d5/flashinfer/rope.py#L1286) | Cache prepare | 融合 MLA RoPE 与 FP8 cache 量化，减少 cache 写入前后的独立 kernel |

`attn_scores` 的 logits 定义与 DeepGEMM 相同：对每个历史位置计算各 indexer head 的 `ReLU(q·k)`，乘 head weight 后求和。它只完成“全历史扫描”，不执行 Top-K 或 sparse attention。

API 成熟度上，FlashInfer 的 Top-K/remap 已经是顶层公开导出，且显式考虑 graph safety、determinism 和 page layout。MLA wrapper 正在迁移到 canonical metadata/structural input；兼容入口仍在，但新接入不应继续使用旧 positional API。

截至固定的三框架 commit，SGLang、vLLM 和 TensorRT-LLM 尚未切换到 2026-08 新增的 FlashInfer `attn_scores`：它们的 DSA indexer logits 仍主要调用 DeepGEMM、AITer 或各自的 CuTeDSL/Triton 实现。FlashInfer 新 API 更像是下一轮统一/替换入口。

#### 里程碑 PR

| 时间 | PR | API/算子进展 |
| --- | --- | --- |
| 2025-02 | [#765](https://github.com/flashinfer-ai/flashinfer/pull/765)、[#804](https://github.com/flashinfer-ai/flashinfer/pull/804) | 支持 DeepSeek prefill shape；引入 memory-efficient fused paged MLA attention |
| 2025-04 | [#1031](https://github.com/flashinfer-ai/flashinfer/pull/1031) | 增加 CUTLASS MLA backend |
| 2025-11～12 | [#2138](https://github.com/flashinfer-ai/flashinfer/pull/2138)、[#2163](https://github.com/flashinfer-ai/flashinfer/pull/2163) | TRTLLM-gen per-tensor sparse MLA；MLA API 从 `decode.py` 独立到 `flashinfer.mla` 并进入文档 |
| 2025-12 | [#2119](https://github.com/flashinfer-ai/flashinfer/pull/2119)、[#2215](https://github.com/flashinfer-ai/flashinfer/pull/2215) | 面向 sparse attention 优化 Top-K，并加入融合 page construction/remap |
| 2026-03 | [#2836](https://github.com/flashinfer-ai/flashinfer/pull/2836)、[#2743](https://github.com/flashinfer-ai/flashinfer/pull/2743) | sparse MLA decode kernel selection heuristic；CuTe DSL MLA decode op |
| 2026-04 | [#3009](https://github.com/flashinfer-ai/flashinfer/pull/3009)、[#3133](https://github.com/flashinfer-ai/flashinfer/pull/3133) | 新 Top-K 算法；增加 `row_starts` 和 `dsa_graph_safe` |
| 2026-05～06 | [#3269](https://github.com/flashinfer-ai/flashinfer/pull/3269)、[#3355](https://github.com/flashinfer-ai/flashinfer/pull/3355)、[#3395](https://github.com/flashinfer-ai/flashinfer/pull/3395) | DSV4 sparse MLA、TRTLLM-gen/CuTe decode autotune、SM120 sparse MLA |
| 2026-07～08 | [#4108](https://github.com/flashinfer-ai/flashinfer/pull/4108)、[#3901](https://github.com/flashinfer-ai/flashinfer/pull/3901) | NoPE sparse MLA decode；`top_k_varlen` 的 GVR/radix sparse-KV selection |
| 2026-08 | [#4365](https://github.com/flashinfer-ai/flashinfer/pull/4365) | 新增公开 `attn_scores`：FP8/FP4 paged-MQA logits 与 GPU schedule API |
| 2026-08 | [#4697](https://github.com/flashinfer-ai/flashinfer/pull/4697)、[#4719](https://github.com/flashinfer-ai/flashinfer/pull/4719) | 隔离 planned FA2/FA3/CUTLASS backend；CuTeDSL variable-Q decode + DCP |

当前值得跟踪的上游 open PR：

| PR | 方向 |
| --- | --- |
| [#2814](https://github.com/flashinfer-ai/flashinfer/pull/2814) | 融合完整 V3.2 decode indexer：paged logits + histogram/clusters Top-K，目标是消除大 logits 中间张量 |
| [#4031](https://github.com/flashinfer-ai/flashinfer/pull/4031) | 继续重构 `BatchMLAPagedAttentionWrapper` backend internals，同时承诺保持公开 API 兼容 |
| [#4551](https://github.com/flashinfer-ai/flashinfer/pull/4551) | 为 SM120 sparse MLA 增加受支持 `(heads, topk)` 查询，避免运行时才发现 dispatch miss |
| [#4737](https://github.com/flashinfer-ai/flashinfer/pull/4737) | paged-MQA logits 扩到 Rubin SM107，并补 FP4 `next_n=4` |
| [#4802](https://github.com/flashinfer-ai/flashinfer/pull/4802)、[#4842](https://github.com/flashinfer-ai/flashinfer/pull/4842) | 重构 SM120 sparse MLA；增加 GLM-5.3 NoPE sparse MLA |

### 6.2 FlashMLA

#### 公开算子 API

FlashMLA 的公开面非常小，当前 [`__all__`](https://github.com/deepseek-ai/FlashMLA/blob/15f13e5030374295491c5ce31b02d7e63a7772c6/flash_mla/__init__.py#L1) 只有六组函数，MLA/DSA 核心是三项：

| API | 功能与约束 |
| --- | --- |
| [`get_mla_metadata`](https://github.com/deepseek-ai/FlashMLA/blob/15f13e5030374295491c5ce31b02d7e63a7772c6/flash_mla/flash_mla_interface.py#L37) | 现在返回空的 `FlashMLASchedMeta`，第一次 attention 调用才按 shape、topk、cache length 初始化 scheduler；旧位置参数仍接受但被忽略 |
| [`flash_mla_with_kvcache`](https://github.com/deepseek-ai/FlashMLA/blob/15f13e5030374295491c5ce31b02d7e63a7772c6/flash_mla/flash_mla_interface.py#L53) | 统一 dense/sparse decode：`indices is None` 走 paged dense；传 `indices` 走 token-sparse decode。当前 sparse 路径要求 non-causal + FP8 KV cache，可接 `topk_length` 和第二份 `extra_k_cache`/indices |
| [`flash_mla_sparse_fwd`](https://github.com/deepseek-ai/FlashMLA/blob/15f13e5030374295491c5ce31b02d7e63a7772c6/flash_mla/flash_mla_interface.py#L176) | DSA sparse prefill forward；输入 BF16 `q/kv` 与 int32 indices，返回 `(out, max_logits, lse)`。当前不带 batch 维，需要框架自行 reshape/rebase indices |
| `flash_attn_varlen_func` | SM100 dense MHA varlen prefill，含 autograd forward/backward |
| `flash_attn_varlen_qkvpacked_func` / `flash_attn_varlen_kvpacked_func` | 上述 dense MHA prefill 的 packed-QKV/KV 兼容入口 |

当前官方 support matrix 的主线是：SM90 dense decode；SM90/SM100 FP8 sparse decode；SM90/SM100 sparse prefill；SM100 dense MHA prefill。V3/V3.2 的 MQA 形态通常是 `d_qk=576`、`d_v=512`，FP8 sparse decode cache 每 token 包含 512-byte FP8 NoPE、四个 FP32 scale 和 64 个 BF16 RoPE 元素。

两个 API 风险需要单独标记：

- `FlashMLASchedMeta` 可复用的前提是 shape、`cache_seqlens`、`topk_length` 等保持一致；框架通常为 graph bucket 分别维护 metadata。
- README decode 示例仍使用旧 positional 参数顺序，open [#214](https://github.com/deepseek-ai/FlashMLA/pull/214) 正在修复。当前 `softmax_scale` 位于 `num_splits` 和 `causal` 之间，新代码应全部使用 keyword argument，避免静默绑定错误。

#### 里程碑 PR

| 时间 | PR | API/算子进展 |
| --- | --- | --- |
| 2025-04 | [#71](https://github.com/deepseek-ai/FlashMLA/pull/71) | dense MLA decode kernel 更新，保持旧接口兼容并提升 compute-bound 性能 |
| 2025-08 | [#76](https://github.com/deepseek-ai/FlashMLA/pull/76) | NVIDIA 贡献 SM100 dense MHA forward/backward |
| 2025-09 | [#98](https://github.com/deepseek-ai/FlashMLA/pull/98) | 正式发布 Hopper DSA：sparse prefill + FP8 paged sparse decode |
| 2026-01 | [#150](https://github.com/deepseek-ai/FlashMLA/pull/150) | 大规模 API/kernel 重构，形成当前 lazy scheduler 和统一 dense/sparse decode 接口 |
| 2026-04 | [#178](https://github.com/deepseek-ai/FlashMLA/pull/178)、[#181](https://github.com/deepseek-ai/FlashMLA/pull/181) | 在 `nv_dev` 分支增加 DSA backward indexer LSE，以及 Top-K 2048/1024 instantiation；不等同于 main 已公开 sparse backward API |

当前上游 open PR 显示的演进方向：

| PR | 方向 |
| --- | --- |
| [#183](https://github.com/deepseek-ai/FlashMLA/pull/183) | 补 SM80/A100 dense MLA decode |
| [#198](https://github.com/deepseek-ai/FlashMLA/pull/198) | SM100 sparse MLA backward、sliding window/sink、block-sparse forward/backward，补训练闭环 |
| [#200](https://github.com/deepseek-ai/FlashMLA/pull/200) | SM90 将 selected KV 先 pack 再做 sparse decode，改善随机 gather 行为 |
| [#216](https://github.com/deepseek-ai/FlashMLA/pull/216) | V3.2 sparse decode 支持 per-request dynamic `topk_length` |
| [#184](https://github.com/deepseek-ai/FlashMLA/pull/184) | 修复 SM100 V3.2 sparse decode 的 FP8 handling；说明该组合仍在持续收敛 |

### 6.3 DeepGEMM

#### 公开算子 API

DeepGEMM 不是 MLA attention 库；在 DSA 中它处于 **indexer projection/scoring** 层。当前 Python 符号由 pybind 直接导出，核心定义在 [`attention.hpp`](https://github.com/deepseek-ai/DeepGEMM/blob/559d79fb6994a58b8a15b4b93bf13ccc16edf247/csrc/apis/attention.hpp#L19)：

| API | 阶段 | 功能与当前语义 |
| --- | --- | --- |
| `fp8_gemm_nt_skip_head_mid` | Projection | FP8 GEMM 的特殊 epilogue；每个 head 只把 left/right 写到扩展 output 的对应位置并跳过中间段，避免额外 layout-copy kernel；caller 需保证中间段已有预期值 |
| [`fp8_fp4_mqa_logits`](https://github.com/deepseek-ai/DeepGEMM/blob/559d79fb6994a58b8a15b4b93bf13ccc16edf247/csrc/apis/attention.hpp#L76) | Prefill indexer | 非 paged weighted-ReLU MQA logits；`q=(data, optional_scale)`、`kv=(data, scale)`，统一 FP8、MXFP8、MXFP4；用 `cu_seq_len_k_start/end` 描述每个 query 扫描范围 |
| [`get_paged_mqa_logits_metadata`](https://github.com/deepseek-ai/DeepGEMM/blob/559d79fb6994a58b8a15b4b93bf13ccc16edf247/csrc/apis/attention.hpp#L184) | Decode metadata | 根据二维 `context_lens`、KV block size 和 SM 数生成 schedule；metadata 必须和后续 paged op 使用相同配置 |
| [`fp8_fp4_paged_mqa_logits`](https://github.com/deepseek-ai/DeepGEMM/blob/559d79fb6994a58b8a15b4b93bf13ccc16edf247/csrc/apis/attention.hpp#L219) | Decode/MTP indexer | 从 fused paged index-K cache 扫描 logits；支持 FP8/MXFP8/MXFP4、FP32/BF16 logits、二维 `context_lens`，以及 SM100 的可选 varlen indices |
| [`fp8_mqa_logits`](https://github.com/deepseek-ai/DeepGEMM/blob/559d79fb6994a58b8a15b4b93bf13ccc16edf247/csrc/apis/attention.hpp#L365) | Legacy prefill | FP8-only wrapper，内部转调 `fp8_fp4_mqa_logits`；新接入不应再把它当作完整能力面 |
| `fp8_paged_mqa_logits` | Legacy decode | FP8-only wrapper，内部转调统一 paged API |

算子输出是形如 `[num_q, max_kv]` 的 logits 大矩阵。其核心评分为：

```text
logit(q, kv_position) = Σ_head weight(q, head) × ReLU(q_head · index_k)
```

`clean_logits=True` 会把扫描范围外位置写成 `-inf`；`max_seqlen_k` 可让 non-paged API 返回压缩宽度，减少 prefill 临时 logits 内存。框架随后必须再调用 Top-K，并把局部 token index 映射为物理 KV 地址。

当前 main 的主要硬件边界：

- SM90：FP8 路径，indexer heads 主要为 32/64；paged KV block size 为 64。
- SM100：FP8/MXFP8/MXFP4，heads 可为 8/16/32/64，head dim 可为 32/64/128（FP4 不含 32），paged block size 可为 32/64/128。
- [#324](https://github.com/deepseek-ai/DeepGEMM/pull/324) 的 SM120 支持合并到 `nv_dev`，不是当前 main 的保证；main README 和上述固定代码仍以 SM90/SM100 为公开主线。框架在 SM120 上因此通常选择 FlashInfer、Triton/CuTeDSL 或 torch fallback。

#### 里程碑 PR

| 时间 | PR | API/算子进展 |
| --- | --- | --- |
| 2025-09 | [#200](https://github.com/deepseek-ai/DeepGEMM/pull/200) | 首次发布 DeepSeek-V3.2/H800 indexer 的 non-paged + paged FP8 MQA-logits kernel |
| 2025-11 | [#227](https://github.com/deepseek-ai/DeepGEMM/pull/227)、[#229](https://github.com/deepseek-ai/DeepGEMM/pull/229) | SM100 MMA shape 优化；SM90 logits 正确性修复 |
| 2026-02 | [#285](https://github.com/deepseek-ai/DeepGEMM/pull/285) | 修复 SM100 MQA logits 同步问题 |
| 2026-04 | [#304](https://github.com/deepseek-ai/DeepGEMM/pull/304) | FP4 indexer、larger-MTP 和统一 `fp8_fp4_*` API；旧 `fp8_*` 降为 legacy wrapper |
| 2026-04～06 | [#314](https://github.com/deepseek-ai/DeepGEMM/pull/314)、[#353](https://github.com/deepseek-ai/DeepGEMM/pull/353) | SM90 next-N/metadata IMA 与空 workload OOB 修复；其中 #314 落在 `nv_dev` |
| 2026-06 | [#364](https://github.com/deepseek-ai/DeepGEMM/pull/364) | 合并 SM100 paged/non-paged、FP8/FP4 四套 kernel，扩展 heads/head-dim/reduce precision 并改善 load balance |
| 2026-07 | [#377](https://github.com/deepseek-ai/DeepGEMM/pull/377) | main public release 增加 MXFP8 indexer，并继续统一量化格式 |
| 2026-06 | [#324](https://github.com/deepseek-ai/DeepGEMM/pull/324)、[#369](https://github.com/deepseek-ai/DeepGEMM/pull/369) | `nv_dev` 的 SM120 与 16-head 演进线；需看后续 public-release PR 是否同步到 main，不能只看 merged 状态 |

当前 open PR：

| PR | 分支 | 方向 |
| --- | --- | --- |
| [#326](https://github.com/deepseek-ai/DeepGEMM/pull/326) | main | SM90 FP4 MQA logits，补 Hopper FP4 indexer |
| [#340](https://github.com/deepseek-ai/DeepGEMM/pull/340) | main | SM90 paged FP8 logits 支持 `next_n=3` |
| [#399](https://github.com/deepseek-ai/DeepGEMM/pull/399) | main | 修复 SM90 paged scheduler `prefix_sum` 越界，属于 decode 正确性/稳定性问题 |
| [#379](https://github.com/deepseek-ai/DeepGEMM/pull/379) | `nv_dev` | SM120 小 head-dim 的 FP8 MQA-logits swizzle 修复 |
| [#393](https://github.com/deepseek-ai/DeepGEMM/pull/393) | `nv_dev` | 从 pybind11 迁移到 `TORCH_LIBRARY`，改善框架注册、fake/meta op 和编译集成 |

### 6.4 三框架实际调用落点

| 算子库 | SGLang | vLLM | TensorRT-LLM |
| --- | --- | --- | --- |
| FlashInfer | [`flashinfer_mla_backend.py`](https://github.com/sgl-project/sglang/blob/8a191554e379741048ecfac02cea334eb2b883e0/python/sglang/srt/layers/attention/flashinfer_mla_backend.py#L73) 使用 planned MLA wrapper；[`dsa_topk_backend.py`](https://github.com/sgl-project/sglang/blob/8a191554e379741048ecfac02cea334eb2b883e0/python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py#L83) 使用 Top-K/remap | [`flashinfer_mla.py`](https://github.com/vllm-project/vllm/blob/d0e695a91b67a8214c2e9ed77595186d2f2844b6/vllm/v1/attention/backends/mla/flashinfer_mla.py#L1) 和 sparse/SM120 backend 调 TRTLLM-gen MLA | [`flashinfer.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/f04859d04de68b255343a6eb4a7d9d0bbaf82ca8/tensorrt_llm/_torch/attention_backend/flashinfer.py#L221) 使用 planned MLA；TRTLLM-gen backend 也反向通过 FlashInfer 暴露 kernel |
| FlashMLA | 通过 `sgl_kernel.flash_mla` 的 AOT/bundled wrapper 提供 dense/sparse backend | [`ops/flashmla.py`](https://github.com/vllm-project/vllm/blob/d0e695a91b67a8214c2e9ed77595186d2f2844b6/vllm/v1/attention/ops/flashmla.py#L92) 封装上游 decode/prefill API | 构建时可打包 `tensorrt_llm.flash_mla`，DSA module 在 Hopper sparse prefill 中调用 |
| DeepGEMM | [`dsa_indexer.py`](https://github.com/sgl-project/sglang/blob/8a191554e379741048ecfac02cea334eb2b883e0/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L880) 调 metadata、paged/non-paged FP8 logits | [`sparse_attn_indexer.py`](https://github.com/vllm-project/vllm/blob/d0e695a91b67a8214c2e9ed77595186d2f2844b6/vllm/model_executor/layers/sparse_attn_indexer.py#L500) 通过 vLLM wrapper 调统一 FP8/FP4 API | [`indexer.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/f04859d04de68b255343a6eb4a7d9d0bbaf82ca8/tensorrt_llm/_torch/attention_backend/sparse/dsa/indexer.py#L1322) 在 FP8 legacy、统一 FP8/FP4 与本地 CuTeDSL kernel 间 dispatch |

这也解释了三框架的配置差异：`dsa-paged-mqa-logits-backend` 选择的是 **indexer scoring**，`dsa-topk-backend` 选择的是 **selection/remap**，而 `dsa-prefill/decode-backend` 选择的才是 **最终 sparse MLA**；三者不能互相替代。

## 7. 三框架当前在途 PR 与尚未完全覆盖的方向

以下状态以 2026-09-01 为准；它们用于判断近期演进方向，不应当作当前已发布能力：

| 项目 | Open PR | 方向与意义 |
| --- | --- | --- |
| SGLang | [#32779](https://github.com/sgl-project/sglang/pull/32779) | 面向 SM90/SM120 的 fused Triton sparse-MLA prefill，减少多 kernel/gather 开销 |
| SGLang | [#31821](https://github.com/sgl-project/sglang/pull/31821) | DeepSeek-V3.2/GLM-5.x DSA 的 decode context parallel |
| SGLang | [#31480](https://github.com/sgl-project/sglang/pull/31480) | 架构无关 torch paged-MQA logits 与 Triton fast path，补齐可移植 fallback |
| vLLM | [#38476](https://github.com/vllm-project/vllm/pull/38476) | `TRITON_MLA_SPARSE`，覆盖 SM8x/11x/12x 等现有专用 kernel 较弱的平台 |
| vLLM | [#48726](https://github.com/vllm-project/vllm/pull/48726) | 融合 DSA indexer Top-K（LiteTopk），降低 logits→Top-K 的中间开销 |
| vLLM | [#54394](https://github.com/vllm-project/vllm/pull/54394) | prefill indexer rows 在 TP 间切分，降低长上下文 indexer 计算/显存压力 |
| TensorRT-LLM | [#17681](https://github.com/NVIDIA/TensorRT-LLM/pull/17681) | DSA NVFP4 KV Cache，继续压低 Blackwell 上 cache 带宽与容量 |
| TensorRT-LLM | [#18268](https://github.com/NVIDIA/TensorRT-LLM/pull/18268) | 将 DSA decode metadata 融合进单个 Triton kernel |
| TensorRT-LLM | [#16309](https://github.com/NVIDIA/TensorRT-LLM/pull/16309) | Vanilla sparse attention 路径，提供更通用/可比对的实现 |

共同的未完问题集中在：

- **老架构与非 NVIDIA 覆盖**：高性能 sparse MLA 首先集中在 Hopper/数据中心 Blackwell；A100、部分 SM120 模型，以及 AMD/XPU 通常依赖独立 fallback/backend。
- **prefill indexer 扩展性**：长 query × 长 KV 的 indexer 扫描仍可能成为瓶颈，正在通过 chunk、TP/CP、fused projection/logits/topk 优化。
- **量化格式组合爆炸**：模型权重、latent KV、index-K cache、query 的 BF16/FP8/FP4 组合，并非每个后端都支持。
- **跨层/跨 MTP 复用**：IndexCache、共享 Top-K 和 GVR 能省 indexer 成本，但引入 layer pattern、索引生命周期和 graph-safe buffer 管理。
- **调度与图捕获**：ragged batch、speculative/MTP、piecewise CUDA Graph、DCP/PP/CP 同时开启时，metadata 与 index remap 比 attention 数学本身更容易出问题。

## 8. 如何选

- 需要**快速试验不同 MLA/DSA kernel、分开控制 prefill/decode/indexer/topk**：SGLang 的参数面最直接。
- 需要**统一 serving 抽象、按硬件自动选择并验证 backend capability**：vLLM 的配置和 selector 更规整；接受 DSA 大部分结构参数由模型 config 驱动。
- 需要**在 Hopper/Blackwell 上深挖 FP8/FP4、piecewise CUDA Graph、GVR 与专用 NVIDIA kernel**：TensorRT-LLM 的实现最深入，但 DSA 目前要使用 PyTorch runtime，且配置仍带 prototype 属性。
- 做**算子级拆分 benchmark**：把 `indexer projection`、`paged-MQA logits`、`Top-K/remap`、`sparse MLA` 分开计时。Blackwell 上可额外比较 FlashInfer `attn_scores` 与 DeepGEMM，但现有三框架快照尚未自动采用前者。
- 做公平 benchmark 时，至少固定：模型 commit、GPU/SM、weight dtype、latent KV dtype、index-K dtype、page/block size、`index_topk`、prefill/decode 长度、是否 short-seq MHA、是否 IndexCache/GVR、TP/CP/DCP 和 CUDA Graph 模式。只写“FlashMLA vs TRTLLM”不足以复现实验。

## 9. 资料口径

- 用户 API 以三家框架及 FlashInfer、FlashMLA、DeepGEMM main 分支文档和上述固定 commit 的参数定义为准；`nv_dev` PR 单独标明，不把分支合并状态当作 main 已发布。
- PR 表只选择改变架构、公开入口、kernel family、并行/量化/图捕获能力的里程碑，未枚举 bugfix、测试和模型接线小改动。
- “功能描述”基于 PR 标题、合并代码与当前调用链归纳；同一个能力常由多个后续 PR 才完善，不能仅凭最早的 enablement PR 判断生产成熟度。
