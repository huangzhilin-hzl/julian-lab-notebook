# Blackwell MLA 后端选型与开源实现：SGLang、vLLM、TokenSpeed

核查日期：2026-09-17。本文汇总 FlashInfer 两类 MLA API 的区别、TRTLLM-gen cubin、CuTe DSL sparse 限制、FlashMLA 使用条件，以及 TokenSpeed 自身运行时和 `tokenspeed-mla` 库的接入情况。B 卡在本文指 B200/B300 数据中心 Blackwell（SM100/103），不把 SM120/121 工作站卡视作同一支持范围。

结论来自固定版本源码检查；没有执行 GPU benchmark。上游 `main` 的策略不保证已进入发行版，安装依赖、显式后端配置、模型 shape 和 KV 格式仍会过滤候选。性能方向是源码注释或作者报告，不是本次复测结果。

**1. 先区分算法、框架后端和底层 kernel。**

| 名称 | 本文含义 |
| --- | --- |
| Dense MLA decode | 读取请求历史的 compressed latent KV；DeepSeek V3/R1 常见 absorbed QK=576、V=512 |
| MLA prefill | 常见为展开后的多头 Q/K/V 计算；192/128 等 prefill shape 不能直接当作 576/512 decode |
| Sparse MLA / DSA | 按每 query 的 top-k token 索引访问 KV；还需区分 indexer/top-k 与最终 attention |
| `FLASHINFER_MLA` / `trtllm_mla` | 框架接入层名称；需继续追实际调用和 backend 参数 |
| `trtllm-gen` | FlashInfer 的 TRT-LLM generated-kernel 路径，主 attention kernel 从 cubin 加载 |
| `tokenspeed-mla` | 可独立安装的 CuTe DSL MLA 库；TokenSpeed、SGLang、vLLM 都有调用路径 |
| TokenSpeed 运行时 | 有统一 MLA/DSA dispatcher，也有显式 `tokenspeed_mla`、`trtllm_mla`、`flashmla` 等后端；不等于只用自家 MLA 库 |

**1.1 `BatchMLAPagedAttentionWrapper` 与 TRTLLM MLA API 的关系。**

两者都属于 `flashinfer.mla`，但在本文固定版本中是独立入口。`BatchMLAPagedAttentionWrapper` 负责矩阵吸收后的 paged MLA decode 和 incremental prefill；非吸收形式的 prefill 需要其他 prefill wrapper。它不封装 `trtllm_batch_decode_with_kv_cache_mla`，也不是所有 MLA 库的统一入口。范围说明见 [_wrapper.py:109](https://github.com/flashinfer-ai/flashinfer/blob/cbb79124f304d0fa82c8ad47e33c215f229e0f62/flashinfer/mla/_batch_mla/_wrapper.py#L109-L120)。

| API | 调用方式 | 本文版本的实现选择 |
| --- | --- | --- |
| `flashinfer.mla.BatchMLAPagedAttentionWrapper` | 创建 wrapper，先 `plan()`，再 `run()` | `fa2`、`fa3`、`cutlass`、`cutile`；`auto` 仅按架构在 FA2/FA3 间选择 |
| `flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla` | 直接函数调用 | B200/B300 上的 `trtllm-gen` / `cute-dsl` 路径；`auto` 的行为还受 sparse、DCP 和 shape 影响。SM120/121 另有 XQA / sparse 分支 |

Wrapper 注册表见 [_wrapper.py:68](https://github.com/flashinfer-ai/flashinfer/blob/cbb79124f304d0fa82c8ad47e33c215f229e0f62/flashinfer/mla/_batch_mla/_wrapper.py#L68-L73)，构造器及合法 backend 参数见 [_wrapper.py:253](https://github.com/flashinfer-ai/flashinfer/blob/cbb79124f304d0fa82c8ad47e33c215f229e0f62/flashinfer/mla/_batch_mla/_wrapper.py#L253-L280)，另一 API 的完整签名与架构说明见 [_core.py:3406](https://github.com/flashinfer-ai/flashinfer/blob/cbb79124f304d0fa82c8ad47e33c215f229e0f62/flashinfer/mla/_core.py#L3406-L3444)。向 wrapper 传 `backend="trtllm-gen"` 或 `backend="cute-dsl"` 会触发 `ValueError`。这些 backend 名称不能在两个 API 间直接互换。

**B200/B300 上，wrapper 的 `backend="auto"` 会选 FA2，不会自动切到 TRTLLM-gen。** 实际选择只有下面两个函数：

```python
def determine_mla_backend(device: torch.device) -> str:
    return "fa3" if is_sm90a_supported(device) else "fa2"

def is_sm90a_supported(device: torch.device) -> bool:
    major, _ = get_compute_capability(device)
    return major == 9 and version_at_least(torch.version.cuda, "12.3")
```

出处：[utils.py:713](https://github.com/flashinfer-ai/flashinfer/blob/cbb79124f304d0fa82c8ad47e33c215f229e0f62/flashinfer/utils.py#L713-L714)、[utils.py:642](https://github.com/flashinfer-ai/flashinfer/blob/cbb79124f304d0fa82c8ad47e33c215f229e0f62/flashinfer/utils.py#L642-L644)。SM100/103 的 major 为 10，不满足这里的 SM90 判断；显式 `cutlass` / `cutile` 需要调用方自己选择。源码还专门对 Blackwell 上的 `auto` 回退发出 decode 性能警告，并推荐外部 TRTLLM MLA API，或硬件支持时 wrapper 内的 cuTile，见 [_wrapper.py:132](https://github.com/flashinfer-ai/flashinfer/blob/cbb79124f304d0fa82c8ad47e33c215f229e0f62/flashinfer/mla/_batch_mla/_wrapper.py#L132-L157)。这是源码警告，不是本文实测的性能排序。

因此，框架声称使用“FlashInfer MLA”时，还要继续看它调用哪个入口。这里的 `fa3` 是 FlashInfer wrapper 的后端名，不能等同于 DeepSeek FlashMLA；`cutlass`、`cutile` 与另一个 API 的 `cute-dsl` 也分别指不同实现。前面针对 TRTLLM-gen cubin 和 CuTe DSL token-sparse 的结论，不能直接套到这个 wrapper 上。

**2. B200/B300 上实际如何选择。**

| 框架 / 场景 | 核查版本的行为 | 证据 |
| --- | --- | --- |
| SGLang，V3/R1/V3.1 dense MLA | 未指定 attention/prefill/decode 后端时，SM100 家族模型分支选择 `trtllm_mla` | [deepseek_v2.py:150](https://github.com/sgl-project/sglang/blob/acfde25d345d45960003b4428a0b6f9bb412d14e/python/sglang/srt/arg_groups/model_overrides/deepseek_v2.py#L150-L161) |
| SGLang，V3.2/GLM 类 DSA，普通 FP8 KV | 常规分支的 prefill/decode 均默认 `trtllm` | [overrides.py:768](https://github.com/sgl-project/sglang/blob/acfde25d345d45960003b4428a0b6f9bb412d14e/python/sglang/srt/arg_groups/overrides.py#L768-L780) |
| SGLang，DSA，BF16 KV | 常规分支 prefill=`flashmla_sparse`，Blackwell decode=`trtllm`；learnable sinks、HiSparse 等另有优先分支 | [overrides.py:768](https://github.com/sgl-project/sglang/blob/acfde25d345d45960003b4428a0b6f9bb412d14e/python/sglang/srt/arg_groups/overrides.py#L768-L780) |
| vLLM，dense MLA decode | Blackwell 候选首位 `FLASHINFER_MLA`，其次 `TOKENSPEED_MLA`，再到 CUTLASS 等；这是优先表，不是逐请求跨库测速 | [cuda.py:95](https://github.com/vllm-project/vllm/blob/de24e5190820cc6640f2d55295405f228fffbc41/vllm/platforms/cuda.py#L95-L131) |
| vLLM，sparse MLA | 量化 KV 优先 FlashInfer；BF16 且本地 heads≤16 优先 FlashInfer，否则优先 FlashMLA；仍要通过格式/功能检查 | [cuda.py:95](https://github.com/vllm-project/vllm/blob/de24e5190820cc6640f2d55295405f228fffbc41/vllm/platforms/cuda.py#L95-L131) |
| vLLM，`FLASHINFER_MLA` 内部 | DCP>1 或 TRTLLM-gen 无法处理的部分 head 数显式走 CuTe DSL，其他配置可走 FlashInfer auto | [flashinfer_mla.py:327](https://github.com/vllm-project/vllm/blob/de24e5190820cc6640f2d55295405f228fffbc41/vllm/v1/attention/backends/mla/flashinfer_mla.py#L327-L355) |
| TokenSpeed，普通 MLA 自动入口 | `AttentionArch.MLA → "mla" → MLAAttnBackend → tokenspeed_kernel`；按 signature、traits、solution/override 选实现 | [registry.py:404](https://github.com/lightseekorg/tokenspeed/blob/a98c3fc06ed1f760b44cc5818519932c89393417/python/tokenspeed/runtime/layers/attention/registry.py#L404-L413)、[mla.py:101](https://github.com/lightseekorg/tokenspeed/blob/a98c3fc06ed1f760b44cc5818519932c89393417/python/tokenspeed/runtime/layers/attention/backends/paged/mla.py#L101-L137) |
| TokenSpeed，显式 `tokenspeed_mla` | `CuteDSLMLABackend` 调用自家库的 decode/verify 和 prefill；不是 TRTLLM-gen 主 kernel 的别名 | [tokenspeed_mla.py:99](https://github.com/lightseekorg/tokenspeed/blob/a98c3fc06ed1f760b44cc5818519932c89393417/python/tokenspeed/runtime/layers/attention/backends/paged/tokenspeed_mla.py#L99-L175)、[tokenspeed_mla.py:662](https://github.com/lightseekorg/tokenspeed/blob/a98c3fc06ed1f760b44cc5818519932c89393417/python/tokenspeed/runtime/layers/attention/backends/paged/tokenspeed_mla.py#L662) |
| TokenSpeed，NVIDIA Kimi K3 的相关 hybrid/cache-plan 分支 | 未显式指定时选择 `tokenspeed_mla`；这是有条件的模型分支，不能概括全部 MLA 模型 | [registry.py:641](https://github.com/lightseekorg/tokenspeed/blob/a98c3fc06ed1f760b44cc5818519932c89393417/python/tokenspeed/runtime/layers/attention/registry.py#L641-L645) |
| TokenSpeed，DSA | 独立 `dsa` 入口；NVIDIA dense delegate 为 TRTLLM MLA，sparse registry 同时注册 FlashInfer TRTLLM 和 FlashMLA | [dsa.py:59](https://github.com/lightseekorg/tokenspeed/blob/a98c3fc06ed1f760b44cc5818519932c89393417/python/tokenspeed/runtime/layers/attention/backends/paged/dsa.py#L59-L71)、[flashinfer.py:103](https://github.com/lightseekorg/tokenspeed/blob/a98c3fc06ed1f760b44cc5818519932c89393417/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsa/flashinfer.py#L103-L138)、[cuda.py:247](https://github.com/lightseekorg/tokenspeed/blob/a98c3fc06ed1f760b44cc5818519932c89393417/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsa/cuda.py#L247-L271) |

本表集中于 V3/R1 dense 和 V3.2/GLM 类 DSA。DSv4/V4.1 有独立模型路径、SWA/压缩 cache 和格式契约，不套用这张默认表。

**3. TRTLLM-gen 到底开放了什么。**

FlashInfer 公开 Python API、参数验证、kernel 选型、metadata/launcher 和部分辅助计算。JIT 模块会编译调用层；主 attention kernel 由名字定位 `.cubin`，经 `getCubin()` 取得二进制，再交给 `cuModuleLoadData()` / `cuModuleGetFunction()`。因此仓库存在 `.cu` launcher 或能 JIT，并不意味着可以从公开源码重建同一主 kernel。

入口：[modules.py:1946](https://github.com/flashinfer-ai/flashinfer/blob/cbb79124f304d0fa82c8ad47e33c215f229e0f62/flashinfer/jit/attention/modules.py#L1946-L1987)；cubin 加载位置：[fmhaKernels.cuh:1316](https://github.com/flashinfer-ai/flashinfer/blob/cbb79124f304d0fa82c8ad47e33c215f229e0f62/include/flashinfer/trtllm/fmha/fmhaKernels.cuh#L1316-L1332)。

同名 `trtllm_batch_decode_with_kv_cache_mla` 已接入多个实现。指定 `backend="cute-dsl"` 可以进入开源 CuTe DSL dense MLA，但它是另一个实现；不能据其源代码断言 TRTLLM-gen cubin 使用相同 mainloop 或调度。

**4. 为什么 FlashInfer 这条 CuTe DSL 路径不接 token-sparse MLA。**

限定结论：`trtllm_batch_decode_with_kv_cache_mla` 所接入的 CuTe DSL dense decode 拒绝 `sparse_mla_top_k>0`。实际 guard 在 [_core.py:2789](https://github.com/flashinfer-ai/flashinfer/blob/cbb79124f304d0fa82c8ad47e33c215f229e0f62/flashinfer/mla/_core.py#L2789)。

| 项目 | Dense paged MLA | Token-sparse MLA |
| --- | --- | --- |
| 被访问的逻辑 token | 请求的连续历史范围；物理页可不连续 | 每个 query 自己的 top-k token 集合 |
| 索引 | 请求级 page table | query 级 token indices，以及有效长度/无效索引 |
| KV 加载 | 页内连续，可围绕规则 tile 搬运 | token gather，可能跨大量页 |
| 调度依据 | KV 长度、连续 KV tiles、split-KV | top-k 长度、query 之间不同的索引、gather 与计算重叠 |

当前 dense kernel 的 `load_page_table()` 使用 `k_index * page_per_tile + idx` 索引页表，见 [mla_decode_fp16.py:1964](https://github.com/flashinfer-ai/flashinfer/blob/cbb79124f304d0fa82c8ad47e33c215f229e0f62/flashinfer/cute_dsl/attention/monolithic/mla_decode_fp16.py#L1964-L2032)。要高效支持任意 token top-k，需要改加载、mask、调度和 pipeline，不能仅删除 guard。以上是源码结构推断；没有查到维护者对接入排期的明确说明。

CuTe DSL 语言本身可以实现 sparse MLA：FA4 的 `FlashAttentionMLAForwardSm100` 已有 `is_topk_gather`、`CpasyncGatherKVManager` 和 `mIndexTopk`；该固定版本的 DSA 分支还要求 128 query heads，见 [flash_fwd_mla_sm100.py:48](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/flash_fwd_mla_sm100.py#L48-L80)。

还需避免扩大结论：FlashInfer 另有 `trtllm_batch_decode_sparse_mla_dsv4` 的 CuTe DSL HCA 路径，属于不同 API/契约，见 [_core.py:1841](https://github.com/flashinfer-ai/flashinfer/blob/cbb79124f304d0fa82c8ad47e33c215f229e0f62/flashinfer/mla/_core.py#L1841-L1902)。不能写成“FlashInfer 所有 CuTe DSL attention 都不支持稀疏”。

**5. FlashMLA 并非没有被采用。**

| 原因 / 事实 | 准确范围 | 固定代码位置 |
| --- | --- | --- |
| Dense decode 的硬件限制 | vLLM 的 `is_flashmla_dense_supported()` 限定 capability family 90；DeepSeek upstream 支持表也将标准 dense decode 列为 SM90 | [flashmla.py:51](https://github.com/vllm-project/vllm/blob/de24e5190820cc6640f2d55295405f228fffbc41/vllm/v1/attention/ops/flashmla.py#L51-L58)、[README.md:75](https://github.com/deepseek-ai/FlashMLA/blob/ba89a3466e9470ad08ab39738d4e7bb66989e1e7/README.md#L75-L79) |
| Sparse 支持 Blackwell | vLLM 的 sparse 检查接受 capability family 90/100，不能将 dense 限制套给 sparse | [flashmla.py:61](https://github.com/vllm-project/vllm/blob/de24e5190820cc6640f2d55295405f228fffbc41/vllm/v1/attention/ops/flashmla.py#L61-L74) |
| 小 head 数有 padding 代价 | vLLM 代码注释说明低 head 数优先 FlashInfer 是因为 FlashMLA padding；SGLang `_forward_flashmla_sparse()` 的 Blackwell 分支将不足 128 的合法 head 数补到 128 | [cuda.py:95](https://github.com/vllm-project/vllm/blob/de24e5190820cc6640f2d55295405f228fffbc41/vllm/platforms/cuda.py#L95-L131)、[dsa_backend.py:2431](https://github.com/sgl-project/sglang/blob/acfde25d345d45960003b4428a0b6f9bb412d14e/python/sglang/srt/layers/attention/dsa_backend.py#L2431-L2451) |
| 普通 FP8 与 `fp8_ds_mla` 不同 | DeepSeek V3.2 每 token 为 512 B FP8 latent + 16 B 分组 FP32 scales + 128 B BF16 RoPE，总计 656 B | [flash_mla_interface.py:92](https://github.com/deepseek-ai/FlashMLA/blob/ba89a3466e9470ad08ab39738d4e7bb66989e1e7/flash_mla/flash_mla_interface.py#L92-L98) |
| 格式会排除后端 | vLLM 的 FlashInfer sparse SM10 后端明确拒绝 `fp8_ds_mla`；不应将“量化 KV 优先 FlashInfer”读成最终一定使用它 | [flashinfer_mla_sparse.py:126](https://github.com/vllm-project/vllm/blob/de24e5190820cc6640f2d55295405f228fffbc41/vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py#L126-L129) |

例如 TP 后每卡只有 16/32 heads，补到 128 会增加无效工作，但不能据 padding 比例直接算出 8×/4×延迟。该 padding 证据属于特定 SGLang 路径，不代表 FlashMLA 所有 kernel 都有相同约束。

**6. TokenSpeed MLA：内核库、运行时、外部集成分别看。**

**6.1 独立 `tokenspeed-mla` 库。**

它提供 dense paged decode/verify 与 ragged MLA prefill。decode 的输入是 query `[B,Q,H,latent+rope]`、分页 KV 和 page table；prefill 是展开后的 Q/K/V。本文核查的这两个公开入口不接收 DSA 的 top-k token indices，因此不能把它作为 `sparse_mla_top_k` 的直接替代。入口分别在 [mla_decode.py:396](https://github.com/lightseekorg/tokenspeed/blob/a98c3fc06ed1f760b44cc5818519932c89393417/tokenspeed-mla/python/tokenspeed_mla/mla_decode.py#L396-L424)、[mla_prefill.py:262](https://github.com/lightseekorg/tokenspeed/blob/a98c3fc06ed1f760b44cc5818519932c89393417/tokenspeed-mla/python/tokenspeed_mla/mla_prefill.py#L262-L287)。

decode 通过 split-KV 增加并行度，并在需要时合并部分结果；小 head 数、多 query 场景可把 query 与 head 折叠到 MMA 的 M 维：

```text
F = 满足 Q % F == 0 且 H * F <= M_tile 的最大可用折叠因子
H_eff = H * F
Q_eff = Q / F
```

对应 [mla_helpers.py:69](https://github.com/lightseekorg/tokenspeed/blob/a98c3fc06ed1f760b44cc5818519932c89393417/tokenspeed-mla/python/tokenspeed_mla/mla_helpers.py#L69-L81)。例如 M128、H64、Q4，可取 F2，得到 H_eff=128、Q_eff=2；普通 Q1 无法靠折叠 query 获得同样的利用率改善。该版本还有 FP8 SM100 小形状的 M64 tile 选择，见 [mla_helpers.py:26](https://github.com/lightseekorg/tokenspeed/blob/a98c3fc06ed1f760b44cc5818519932c89393417/tokenspeed-mla/python/tokenspeed_mla/mla_helpers.py#L26-L49)，不应把 M128 当作所有配置的唯一 tile。

另外，`enable_packed_q` 路径按连续 query/head 行打包，代码以 layout/view 方式映射，并受 M128 和 stride 条件约束；它与整除式 fold_sq 是两种不同路线。见 [mla_decode.py:578](https://github.com/lightseekorg/tokenspeed/blob/a98c3fc06ed1f760b44cc5818519932c89393417/tokenspeed-mla/python/tokenspeed_mla/mla_decode.py#L578-L595)。不要把某个 Q4、长 KV 场景的优势推广到 Q1 或所有 batch。

源码与文档存在需标明的差异：README 仍介绍可选 binary prefill 版本，见 [README.md:21](https://github.com/lightseekorg/tokenspeed/blob/a98c3fc06ed1f760b44cc5818519932c89393417/tokenspeed-mla/README.md#L21-L23)；但本次固定提交的 `tokenspeed_mla_prefill()` / `_compile_prefill_kernel()` 实际入口走 CuTe DSL JIT，见 [mla_prefill.py:81](https://github.com/lightseekorg/tokenspeed/blob/a98c3fc06ed1f760b44cc5818519932c89393417/tokenspeed-mla/python/tokenspeed_mla/mla_prefill.py#L81-L116)。本文没有在该入口确认到 README 所述 binary selector，因而不把“当前集成会调用可选 binary”记为已验证事实。

**6.2 TokenSpeed 自身运行时。**

```text
普通 MLA 自动选择
  AttentionArch.MLA → "mla" → MLAAttnBackend
  → tokenspeed_kernel.mla_decode_with_kvcache → 按参数和 traits 选择 kernel

显式 --attention-backend tokenspeed_mla
  → CuteDSLMLABackend
  → tokenspeed_mla_decode / tokenspeed_mla_prefill

DSA
  AttentionArch.DSA → "dsa" → dsa_decode / dsa_prefill registry
  → 符合各自契约的 FlashInfer TRTLLM、FlashMLA 等实现
```

通用 MLA dispatcher 对 `tokenspeed_mla` 的一条注册仅服务非因果 proposal block：Q=2…8、H=1…128、latent/RoPE=512/64、page=32/64，且该注册不返回 LSE；不能把这一注册当作所有 Q1 decode 的默认。见 [tokenspeed_mla.py:108](https://github.com/lightseekorg/tokenspeed/blob/a98c3fc06ed1f760b44cc5818519932c89393417/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/mla/tokenspeed_mla.py#L108-L167)。通用入口也注册了 Q1 Triton 路径，见 [triton.py:91](https://github.com/lightseekorg/tokenspeed/blob/a98c3fc06ed1f760b44cc5818519932c89393417/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/mla/triton.py#L91-L105)。这些限制属于 registry 包装，不等于独立库 API 的全部能力。

显式运行时 `tokenspeed_mla` 后端要求 FP8 E4M3 KV、page=32/64，decode 和 prefill 直接调用自家库：[tokenspeed_mla.py:99](https://github.com/lightseekorg/tokenspeed/blob/a98c3fc06ed1f760b44cc5818519932c89393417/python/tokenspeed/runtime/layers/attention/backends/paged/tokenspeed_mla.py#L99-L175)、[tokenspeed_mla.py:553](https://github.com/lightseekorg/tokenspeed/blob/a98c3fc06ed1f760b44cc5818519932c89393417/python/tokenspeed/runtime/layers/attention/backends/paged/tokenspeed_mla.py#L553-L569)、[tokenspeed_mla.py:635](https://github.com/lightseekorg/tokenspeed/blob/a98c3fc06ed1f760b44cc5818519932c89393417/python/tokenspeed/runtime/layers/attention/backends/paged/tokenspeed_mla.py#L635-L655)。NVIDIA K3 的特定 hybrid/cache-plan 默认也会选它，但普通 MLA 默认名仍是 `mla`。

TokenSpeed 的 sparse DSA 路径中，FlashInfer TRTLLM 注册为 `SPECIALIZED`，FlashMLA 为 `PERFORMANT`，两者的 dtype/cache/shape traits 不同；FlashMLA decode 注册要求可用的 packed sparse cache。仅满足硬件条件不足以推断最后命中哪个 kernel。FlashInfer 这一分支确实调用 `trtllm_batch_decode_with_kv_cache_mla(..., backend="trtllm-gen")`，见 [flashinfer.py:290](https://github.com/lightseekorg/tokenspeed/blob/a98c3fc06ed1f760b44cc5818519932c89393417/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsa/flashinfer.py#L290-L315)。

TokenSpeed `dsa/cute_dsl.py` 中的 `cute_dsl_decode_topk` 是 indexer 的 top-k 选择，并不是“自家 CuTe DSL sparse MLA attention 已替代 TRTLLM”的证据，见 [cute_dsl.py:87](https://github.com/lightseekorg/tokenspeed/blob/a98c3fc06ed1f760b44cc5818519932c89393417/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsa/cute_dsl.py#L87-L100)。

**6.3 SGLang / vLLM 如何接入 TokenSpeed。**

| 项目 | 调用范围 | 核查到的包装层条件 | 入口 |
| --- | --- | --- | --- |
| SGLang `tokenspeed_mla` | 自家库 prefill + decode，复用 TRTLLMMLABackend 的准备/metadata 逻辑 | FP8 E4M3 KV、page 32/64；继承类名不意味着主 kernel 仍是 cubin | [tokenspeed_mla_backend.py:110](https://github.com/sgl-project/sglang/blob/acfde25d345d45960003b4428a0b6f9bb412d14e/python/sglang/srt/layers/attention/tokenspeed_mla_backend.py#L110-L136)、[tokenspeed_mla_backend.py:333](https://github.com/sgl-project/sglang/blob/acfde25d345d45960003b4428a0b6f9bb412d14e/python/sglang/srt/layers/attention/tokenspeed_mla_backend.py#L333-L350)、[tokenspeed_mla_backend.py:479](https://github.com/sgl-project/sglang/blob/acfde25d345d45960003b4428a0b6f9bb412d14e/python/sglang/srt/layers/attention/tokenspeed_mla_backend.py#L479-L495) |
| vLLM `TOKENSPEED_MLA` decode | `tokenspeed_mla_decode()`，支持包装层 DCP/LSE | SM10，FP8 KV，page 32/64；模型维度检查 qk_nope=128、qk_rope=64、v=128 | [tokenspeed_mla.py:82](https://github.com/vllm-project/vllm/blob/de24e5190820cc6640f2d55295405f228fffbc41/vllm/v1/attention/backends/mla/tokenspeed_mla.py#L82-L155)、[tokenspeed_mla.py:155](https://github.com/vllm-project/vllm/blob/de24e5190820cc6640f2d55295405f228fffbc41/vllm/v1/attention/backends/mla/tokenspeed_mla.py#L155-L161)、[tokenspeed_mla.py:297](https://github.com/vllm-project/vllm/blob/de24e5190820cc6640f2d55295405f228fffbc41/vllm/v1/attention/backends/mla/tokenspeed_mla.py#L297-L319) |
| vLLM `TOKENSPEED_MLA` prefill | 单独的 MLA prefill backend | SM10，R1 类 prefill 维度；独立选择，不因 decode 选择 TokenSpeed 就必然跟随 | [tokenspeed_mla.py:22](https://github.com/vllm-project/vllm/blob/de24e5190820cc6640f2d55295405f228fffbc41/vllm/v1/attention/backends/mla/prefill/tokenspeed_mla.py#L22-L39)、[selector.py:70](https://github.com/vllm-project/vllm/blob/de24e5190820cc6640f2d55295405f228fffbc41/vllm/v1/attention/backends/mla/prefill/selector.py#L70-L91) |

上述 vLLM 的模型维度 128/64/128 描述的是原始 MLA 投影维度，不是说 absorbed decode 的 query 宽度只有 192。实际 decode 仍按 latent rank + RoPE 构造 query。

vLLM Blackwell dense 候选表把 TokenSpeed 放在 FlashInfer 后面，注释记录它在 batch 较大时有优势、极小 batch 有回退；这是该选择策略的上游依据，不是跨版本通用性能定理。当前 prefill 的常规优先表为 FlashAttention → TRTLLM Ragged → FlashInfer → TokenSpeed；特定 192/64/256 形状把 TRTLLM Ragged 提前。见 [cuda.py:95](https://github.com/vllm-project/vllm/blob/de24e5190820cc6640f2d55295405f228fffbc41/vllm/platforms/cuda.py#L95-L131)、[selector.py:70](https://github.com/vllm-project/vllm/blob/de24e5190820cc6640f2d55295405f228fffbc41/vllm/v1/attention/backends/mla/prefill/selector.py#L70-L91)。

**7. 可读源码与性能对照选择。**

| 实现 | 对 B200/B300 研究最直接的用途 | 使用边界 / 入口 |
| --- | --- | --- |
| TRTLLM-gen | dense/sparse 生产性能对照、研究 launcher 与参数 | 主 attention kernel 是 cubin；[fmhaKernels.cuh:1316](https://github.com/flashinfer-ai/flashinfer/blob/cbb79124f304d0fa82c8ad47e33c215f229e0f62/include/flashinfer/trtllm/fmha/fmhaKernels.cuh#L1316-L1332) |
| FlashInfer CuTe DSL MLA | dense paged decode、QK/PV pipeline、split-KV | 当前同名 API 的 generic token-sparse 模式不支持；[mla_decode_fp16.py:137](https://github.com/flashinfer-ai/flashinfer/blob/cbb79124f304d0fa82c8ad47e33c215f229e0f62/flashinfer/cute_dsl/attention/monolithic/mla_decode_fp16.py#L137-L140)、[mla_decode_fp8.py:135](https://github.com/flashinfer-ai/flashinfer/blob/cbb79124f304d0fa82c8ad47e33c215f229e0f62/flashinfer/cute_dsl/attention/monolithic/mla_decode_fp8.py#L135-L138) |
| TokenSpeed-MLA | dense decode/verify 小 heads、多 query 利用率与 prefill | 分清独立库、框架包装与 dispatcher；[mla_helpers.py:69](https://github.com/lightseekorg/tokenspeed/blob/a98c3fc06ed1f760b44cc5818519932c89393417/tokenspeed-mla/python/tokenspeed_mla/mla_helpers.py#L69-L81)、[mla_prefill.py:81](https://github.com/lightseekorg/tokenspeed/blob/a98c3fc06ed1f760b44cc5818519932c89393417/tokenspeed-mla/python/tokenspeed_mla/mla_prefill.py#L81-L116) |
| DeepSeek FlashMLA | Blackwell sparse MLA；Hopper dense 对照 | packed FP8 格式、head padding、硬件范围分别检查；[README.md:75](https://github.com/deepseek-ai/FlashMLA/blob/ba89a3466e9470ad08ab39738d4e7bb66989e1e7/README.md#L75-L79)、[flash_mla_interface.py:92](https://github.com/deepseek-ai/FlashMLA/blob/ba89a3466e9470ad08ab39738d4e7bb66989e1e7/flash_mla/flash_mla_interface.py#L92-L98) |
| FA4 CuTe DSL MLA | 开源 SM100 token gather、softmax/MMA pipeline | 固定版本 top-k 路径仍有 128-head 等条件；[flash_fwd_mla_sm100.py:48](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/flash_fwd_mla_sm100.py#L48-L80) |
| FlashInfer / CUTLASS C++ MLA | Blackwell TMA、TMEM、warp specialization | dense MLA 模板不是 token-sparse 的即插即用实现；[sm100_fmha_mla_tma_warpspecialized.hpp:56](https://github.com/flashinfer-ai/flashinfer/blob/cbb79124f304d0fa82c8ad47e33c215f229e0f62/include/flashinfer/attention/blackwell/kernel/sm100_fmha_mla_tma_warpspecialized.hpp#L56-L61) |
| TileLang、cuTile / TileGym | 算法原型、可修改 dense/sparse 示例 | PR、集成范围及作者性能数据见 [2026-09-14 PR 地图](mla-implementation-pr-map-20260914.md)；本次未重新核查它们的主干 |

如果重点是 sparse attention mainloop，先看 FlashMLA 与 FA4；如果重点是 dense decode/verify，先对照 FlashInfer CuTe DSL、TokenSpeed-MLA 和 CUTLASS。选择时固定 GPU、每卡 H、Q、batch、KV 长度/top-k、KV 格式、page size、并行模式和是否含预处理/reduction 的计时范围。本文不提供未经同条件复测的统一排行榜。

**8. 固定版本与证据。**

| 项目 | 固定 commit |
| --- | --- |
| vllm-project/vllm | [`de24e5190820`](https://github.com/vllm-project/vllm/commit/de24e5190820cc6640f2d55295405f228fffbc41) |
| sgl-project/sglang | [`acfde25d345d`](https://github.com/sgl-project/sglang/commit/acfde25d345d45960003b4428a0b6f9bb412d14e) |
| deepseek-ai/FlashMLA | [`ba89a3466e94`](https://github.com/deepseek-ai/FlashMLA/commit/ba89a3466e9470ad08ab39738d4e7bb66989e1e7) |
| lightseekorg/tokenspeed | [`a98c3fc06ed1`](https://github.com/lightseekorg/tokenspeed/commit/a98c3fc06ed1f760b44cc5818519932c89393417) |
| Dao-AILab/flash-attention | [`1bda8f9290cd`](https://github.com/Dao-AILab/flash-attention/commit/1bda8f9290cd48d030f1516f0e680cd464ef3554) |
| flashinfer-ai/flashinfer | [`cbb79124f304`](https://github.com/flashinfer-ai/flashinfer/commit/cbb79124f304d0fa82c8ad47e33c215f229e0f62) |

所有源码链接都固定到上述 commit，并指向已读取的行。FlashInfer 使用本地 checkout 的已提交版本；相关文件无本地改动。其余源自本次读取的远端固定提交。

机器可读索引：[backend-source-index.json](evidence/20260917/backend-source-index.json)，记录引用 ID、仓库、commit、路径、行号、匹配文本及源文件 SHA-256。它用于复查引用，不代表已执行对应 CUDA kernel。
