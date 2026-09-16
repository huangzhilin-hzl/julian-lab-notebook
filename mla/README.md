# MLA 配置对照：DeepSeek R1 / V3.1 / V3.2 / V4 / V4.1 与 GLM-5.3

记录时间：2026-09-16；R1 / V3.1 / V3.2 对照补充于 2026-09-17。

范围：DeepSeek-R1、DeepSeek-V3.1、DeepSeek-V3.2、DeepSeek-V4-Flash、DeepSeek-V4-Pro、DeepSeek-V4.1-Flash、GLM-5.3、GLM-5.3-Flash 的语言主干 attention。基于记录时官方 Hugging Face `config.json` 与官方推理代码整理；链接指向 `main`，后续可能更新。R1、V3.1、V3.2 分别采用官方 `deepseek-ai/DeepSeek-R1`、`deepseek-ai/DeepSeek-V3.1`、`deepseek-ai/DeepSeek-V3.2` 仓库。

## 核心区别

这八个模型中，只有 **GLM-5.3-Flash 的 MLA 主注意力是 NoPE**；其余模型都保留 64 维 RoPE。R1、V3.1、V3.2 的 MLA 主维度相同；R1/V3.1 使用全量因果注意力，V3.2 在 MLA 上增加 DSA indexer 和 top-k 稀疏选择。

从 attention kernel 的维度看：

| 模型 | Q/K 非 RoPE 部分 | Q/K RoPE 部分 | Q/K 总维度 | V / attention 输出维度 |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek-R1，矩阵吸收后 | 512 | 64 | **576** | 512 |
| DeepSeek-V3.1，矩阵吸收后 | 512 | 64 | **576** | 512 |
| DeepSeek-V3.2，矩阵吸收后 | 512 | 64 | **576** | 512 |
| DeepSeek-V4-Flash | 448 | 64 | **512** | 512 |
| DeepSeek-V4-Pro | 448 | 64 | **512** | 512 |
| DeepSeek-V4.1-Flash | 448 | 64 | **512** | 512 |
| GLM-5.3，矩阵吸收后 | 512 | 64 | **576** | 512 |
| GLM-5.3-Flash，矩阵吸收后 | 512 | 0 | **512** | 512 |

DeepSeek V4/V4.1 的 `head_dim=512` **已经包含** 64 维 RoPE；R1/V3.1/V3.2 和 GLM 的 `kv_lora_rank=512` 是 latent 维度，RoPE key 需要额外拼接。GLM-5.3-Flash 与 DeepSeek V4/V4.1 虽然 kernel Q/K 宽度同为 512，内部划分和计算语义不同。

DeepSeek V4/V4.1 的 V 使用完整 512 维共享 KV，包含经过 RoPE 的通道；attention 输出的对应通道会做逆旋转，再经过分组输出投影。R1/V3.1/V3.2 和 GLM 吸收路径的 V 是 512 维 latent，不包含额外的 RoPE key 通道。

依据：[R1 官方 Attention 实现](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/modeling_deepseek.py)、[V3.2 官方 MLA 实现](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/inference/model.py)、[V4 官方 Attention 实现](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py)、[V4.1 官方 Attention 实现](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py)，以及下表链接的模型配置。

## 配置字段与推导维度

head 数均为 **TP 切分前**；主干层数不包含 MTP。GLM-5.3-Flash 的 MLA 参数仅适用于其稀疏注意力层。标记 `†` 的值根据官方实现推导或按共享 KV 宽度换算，并非同名的原始配置字段。

| 配置项 | [DS R1](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/config.json) | [DS V3.1](https://huggingface.co/deepseek-ai/DeepSeek-V3.1/blob/main/config.json) | [DS V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json) | [DS V4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json) | [DS V4 Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json) | [DS V4.1 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json) | [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json) | [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 主注意力组织 | 全量因果 MLA | 全量因果 MLA | MLA + DSA | SWA + 压缩 KV | SWA + 压缩 KV | SWA + CSA2 | MLA + DSA / IndexShare | KDA + NoPE MLA / DSA |
| `num_hidden_layers` | 61 | 61 | 61 | 43 | 61 | 40 | 78 | 45：11 MLA + 34 KDA |
| `hidden_size` | 7168 | 7168 | 7168 | 4096 | 7168 | 5120 | 6144 | 4096 |
| `num_heads_q`（=`num_attention_heads`，TP 前） | **128** | **128** | **128** | **64** | **128** | **64** | **64** | **64** |
| `num_key_value_heads` | 128 | 128 | 128 | 1 | 1 | 1 | 64 | 64 |
| `q_lora_rank` | 1536 | 1536 | 1536 | 1024 | 1536 | 1280 | 2048 | 1536 |
| `kv_lora_rank` / 等效共享 KV 宽度 | 512 | 512 | 512 | 512†（含 RoPE） | 512†（含 RoPE） | 512†（含 RoPE） | 512 | 512 |
| `head_dim`，原始字段 | — | — | — | 512 | 512 | 512 | 192 | 0 |
| `qk_head_dim` | 192† | 192† | 192† | 512† | 512† | 512† | 256 | 256 |
| `qk_nope_head_dim` | 128 | 128 | 128 | 448† | 448† | 448† | 192 | 256 |
| `qk_rope_head_dim` | 64 | 64 | 64 | 64 | 64 | 64 | 64 | **0** |
| `v_head_dim` | 128 | 128 | 128 | 512† | 512† | 512† | 256 | 256 |
| `mla_use_nope` | — | — | — | — | — | — | — | `true` |
| `o_groups` | — | — | — | 8 | 16 | 8 | — | — |
| `o_lora_rank` | — | — | — | 1024 | 1024 | 1024 | — | — |
| `index_n_heads` | 不适用 | 不适用 | 64 | 64 | 64 | 32 | 32 | 32 |
| `index_head_dim` | 不适用 | 不适用 | 128 | 128 | 128 | 128 | 128 | 128 |
| `index_topk` | 不适用 | 不适用 | 2048 | 512 | **1024** | 512 | 2048 | 2048 |
| `sliding_window` | — | — | — | 128 | 128 | 128 | — | — |
| `max_position_embeddings` | 163840 | 163840 | 163840 | 1048576 | 1048576 | 1048576 | 1048576 | 1048576 |

字段读取与解释：

- V4.1-Flash 和 GLM-5.3-Flash 的语言配置位于 `text_config`；其余六个模型的字段在配置根级。
- `num_heads_q` 是主 attention 的 Q head 数，对应官方配置中的 `num_attention_heads`，与 indexer 的 `index_n_heads` 分开统计；矩阵吸收改变每个 head 的维度，不改变 Q head 数。
- 按 Q heads 做标准 TP 切分时，kernel 的本地 `num_heads_q = num_attention_heads / attention_tp_size`。例如 attention TP=4 时，R1/V3.1/V3.2/V4-Pro 为 32，其余模型为 16；应使用实际 attention 并行度，而不是直接套用其他模块的 TP 大小。
- `†` 表示按官方实现补齐的维度；`—` 表示官方配置没有该字段，且此处未作等效换算，不表示值为零。
- R1/V3.1 的 indexer 字段标为“不适用”，因为它们使用全量因果 attention，没有 DSA top-k indexer；V3.2 才在这一组对比中引入稀疏选择。
- R1/V3.1/V3.2 的吸收前 `qk_head_dim=128+64=192`；该值由配置推导，吸收后的 kernel 宽度为 `512+64=576`。
- `head_dim` 在不同架构中含义不同。GLM-5.3 的原始值为 192，GLM-5.3-Flash 为 0；二者都不能直接作为 kernel Q/K 宽度。应使用 `qk_*` 和 `kv_lora_rank`。
- DeepSeek V4/V4.1 的 Q/K 总宽度为 `head_dim=512`，非 RoPE 宽度为 `512-64=448`；V 使用完整共享 KV，因此 V 宽度也是 `512`。
- DeepSeek V4/V4.1 行中的等效共享 KV 宽度 `512†` 包含 64 维 RoPE；它不是一个新增的 `kv_lora_rank=512` 配置项，不能再加 64 得到 576。R1/V3.1/V3.2 和 GLM 的 `kv_lora_rank=512` 则不包含额外 RoPE key，因此 R1/V3.1/V3.2 和 GLM-5.3 的吸收后 Q/K 宽度是 `512+64=576`。
- R1/V3.1/V3.2 的 `num_key_value_heads=128`、GLM 的 `num_key_value_heads=64`，都不代表吸收路径需要缓存对应份数的 latent KV。主 KV 缓存按共享 latent 和额外 RoPE key 表示。
- `max_position_embeddings` 按 HF 配置原值记录；R1/V3.1/V3.2 的 `163840` 不能直接当作具体服务端允许的最大请求长度。
- V4-Pro 的 `index_topk=1024` 采用根目录 HF `config.json` 的值；不要用旧版示例或推理配置的值覆盖它。

## R1 / V3.x 与 GLM 的矩阵吸收前后

| 模型 | 吸收前 Q/K | 吸收前 V | 吸收后 Q/K | 吸收后 V |
| --- | --- | ---: | --- | ---: |
| DeepSeek-R1 | `128 + 64 = 192` | 128 | `512 + 64 = 576` | 512 |
| DeepSeek-V3.1 | `128 + 64 = 192` | 128 | `512 + 64 = 576` | 512 |
| DeepSeek-V3.2 | `128 + 64 = 192` | 128 | `512 + 64 = 576` | 512 |
| GLM-5.3 | `192 + 64 = 256` | 256 | `512 + 64 = 576` | 512 |
| GLM-5.3-Flash | `256 + 0 = 256` | 256 | `512 + 0 = 512` | 512 |

吸收后，Q 的非 RoPE 部分映射到 `kv_lora_rank=512` 的空间，attention 输出也先保留在该 latent 空间。这里的 512 不能与配置中的 `qk_nope_head_dim` 或展开后的 `v_head_dim` 混为一谈：R1/V3.1/V3.2 的 `v_head_dim=128`，两个 GLM 模型为 256。

## 稀疏选择与层间组织

以下层号均从 0 开始，只描述主干；配置数组末尾用于 MTP 的条目另行注明。

### DeepSeek-R1 / V3.1 / V3.2

三者均为 61 层、128 个 Q heads，`q_lora_rank=1536`、`kv_lora_rank=512`、`qk_nope_head_dim=128`、`qk_rope_head_dim=64`、`v_head_dim=128`。主 MLA 的投影和缓存维度一致，主要区别在 token 选择方式。

| 对比项 | R1 | V3.1 | V3.2 |
| --- | --- | --- | --- |
| Attention token 范围 | 全部因果可见 token | 全部因果可见 token | DSA indexer 筛选的因果可见 token |
| DSA indexer | 无 | 无 | 每层一个 indexer |
| Indexer 配置 | 不适用 | 不适用 | `index_n_heads=64, index_head_dim=128, index_topk=2048` |
| 吸收后主 KV 表示 | 512 latent + 64 RoPE | 512 latent + 64 RoPE | 512 latent + 64 RoPE，另有 indexer cache |

V3.2 的 top-k 是对历史 token 的选择，不是 V4 的时间压缩 KV，也不是 V4.1 的跨层 KV 共享。官方实现中每个 MLA 层实例化自己的 indexer；历史 token 少于 2048 时，选择数量受可用历史长度限制。

R1/V3.1 的全量 attention 与 V3.2 的稀疏 attention 即使具有相同的吸收后 `Q/K=576, V=512`，也需要按不同的 token 访问模式选择 kernel。

依据：[R1 配置](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/config.json)、[R1 Attention 实现](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/modeling_deepseek.py)、[V3.1 配置](https://huggingface.co/deepseek-ai/DeepSeek-V3.1/blob/main/config.json)、[V3.1 Attention 实现](https://huggingface.co/deepseek-ai/DeepSeek-V3.1/blob/main/modeling_deepseek.py)、[V3.2 配置](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json)、[V3.2 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)、[V3.2 MLA / Indexer 实现](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/inference/model.py)。

### DeepSeek-V4-Flash / Pro

两者均采用 SWA 加可选压缩 KV。`compress_ratios` 中：

| 值 | 主注意力访问方式 |
| --- | --- |
| `0` | 仅 128-token 滑动窗口 |
| `4` | 128-token 窗口 + indexer 选出的压缩 KV；Flash 的 `index_topk=512`，Pro 为 `1024` |
| `128` | 128-token 窗口 + 全部因果可见的 128:1 压缩 KV，不通过 top-k indexer 筛选 |

Flash 主干前两层为 `0`，随后交替为 `4, 128, 4, 128, ...`；Pro 前两层为 `128`，随后同样交替为 `4, 128, ...`。两者配置数组末尾各有一个额外的 `0`，对应 MTP。

因此 `index_topk` 不是所有层统一的 attention KV 数，也不包含额外的 SWA 窗口。[官方实现](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py)

### DeepSeek-V4.1-Flash

主干采用 20 层 causal encoder + 20 层 decoder；CSA2 通过 Full / Reindex / Reuse 模式跨层共享主 KV、indexer K 和 top-k 索引。Decoder 的全局 KV 来自最终 encoder hidden states。[官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)

```python
# 主干 40 层；官方 compress_ratios 末尾另有 3 个 MTP 条目，均为 0。
compress_ratios_backbone = [0] * 2 + [2] * 18 + [1] * 20

kv_source_layer_ids = [2, 8, 14, 20]
index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]

candidate_source_layer_id = 20
candidate_topk_blocks = 2048
candidate_block_size = 8
```

`compress_ratio > 0` 不代表每层都生成自己的压缩 KV；只有 source layer 生成，其他层读取共享缓存。相对于 V4，kernel 主维度仍是 `448+64`，变化集中在压缩率、缓存/索引共享和分层候选筛选。[官方配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json)、[官方实现](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py)

### GLM-5.3

78 层均使用 MLA/DSA。Indexer 采用 IndexShare，`indexer_types` 中有 21 个 `full`、57 个 `shared`；共享的是稀疏选择索引，不能据此认为主 MLA KV 也被跨层共享。

```python
index_topk = 2048
index_topk_freq = 4
index_skip_topk_offset = 3
index_share_for_mtp_iteration = True

# 主干配置数组的等价表达：
indexer_types = ["full"] * 3 + ["shared"] * 3
indexer_types += (["full"] + ["shared"] * 3) * 18
```

[官方配置](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json)

### GLM-5.3-Flash

主干是 34 层 KDA 线性注意力与 11 层 NoPE 稀疏 MLA 的混合架构；MLA 位于 `[3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]`，其余层为 KDA。

```python
mla_use_nope = True
index_topk = 2048
index_kpool = 4
index_kpool_compress = True
index_kpool_always_select_tail = True
index_share_for_mtp_iteration = True
```

`indexer_types` 配置数组全为 `full`，应结合 `layer_types` 解读，不能理解为 45 层都运行 MLA indexer。KDA 的独立配置为 `num_heads=64`、`head_dim=128`、`short_conv_kernel_size=4`；这些维度不属于 MLA。

[官方配置](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json)

## FlashInfer kernel 对照

以下对应记录时本地 FlashInfer 工作树中的代码，用于解释 kernel 维度，不代表对所有模型、dtype 和 GPU 后端的支持承诺。

| 对象 | 配置或维度 |
| --- | --- |
| `flashinfer/mla/_core.py` 中的 `deepseek_mla_dimensions` | `qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128, kv_lora_rank=512`，与 R1/V3.1/V3.2 的 MLA 主维度一致 |
| `flashinfer/mla/_core.py` 中的 `nope_mla_dimensions` | `qk_nope_head_dim=256, qk_rope_head_dim=0, v_head_dim=256, kv_lora_rank=512`，与 GLM-5.3-Flash 的 MLA 配置一致 |
| SM120 `KVCacheTraits<ModelType::DSV3_2>` | `D_NOPE=512, D_ROPE=64, D_QK=576, D_V=512`，对应 V3.2 的稀疏 MLA 吸收路径 |
| SM120 `KVCacheTraits<ModelType::DSV4>` | `D_NOPE=448, D_ROPE=64, D_QK=512, D_V=512` |
| SM120 `KVCacheTraits<ModelType::GLM_NSA>` | `D_NOPE=512, D_ROPE=64, D_QK=576, D_V=512` |
| SM120 `KVCacheTraits<ModelType::GLM53_NOPE>` | `D_NOPE=512, D_ROPE=0, D_QK=512, D_V=512` |

`KVCacheTraits` 定义位于 `include/flashinfer/attention/sparse_mla_sm120/model/kv_cache_traits.cuh`。

模型的 `index_topk` 与 kernel 收到的索引宽度要分别记录：GLM-5.3-Flash 官方配置是 `2048`；本地 `flashinfer/mla/_sparse_mla_sm120_plan.py` 将 128-token indexer tail 合并后，以 `2176 = 2048 + 128` 作为校准宽度。`2176` 不是模型原始 `index_topk`，也不是该实现唯一允许的运行时宽度。
