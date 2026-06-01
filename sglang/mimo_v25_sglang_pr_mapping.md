# MiMo-V2.5 Inference Optimizations and SGLang PR Mapping

Source article: [MiMo-V2.5 系列推理全链路优化](https://mimo.xiaomi.com/zh/blog/mimo-v2-5-inference)

GitHub status checked on 2026-06-01. The mapping below is based on PR titles, PR bodies, and the article's optimization categories. It is not an official one-to-one mapping from Xiaomi.

## Summary

| Article area | SGLang coverage | Notes |
| --- | --- | --- |
| MiMo-V2.5 model support and MTP | Mostly merged, with some HiCache/Mooncake MTP work still open | Day-0 support, MiMoV2MTP draft model, cookbook path, MTP weight-loading fixes, and PD MTP KV transfer fixes are already merged. |
| Hybrid SWA KVCache and SWA-aware prefix cache | Core pieces merged; several HiCache consistency/event PRs open | Dual-pool/SWA pool, SWA prefix cache, SWA eviction, mapping fixes, and UnifiedTree write-through fixes exist upstream. |
| Layerwise KV loadback / PD transfer overlap | Mostly open/WIP | The closest upstream PR is layer-pipelined KV transfer. |
| Cache-aware routing and scheduling | Mixed | Prefix/cache-aware routing exists; more DP/PD cache-aware and balance-weight work is open. |
| Decode memory optimization | Several merged | SWA decode preallocation sizing and optimization are merged. |
| Multimodal EPD encoder optimization | Tracked by RFC #24945; several PRs merged, several open | MiMo EPD support, GPU image preprocess, parallel video decode, and cross-request batching are merged. |
| Internal Xiaomi infra/deployment practices | No clear single upstream PR found | GCache L3 service, Redis-backed LLM-Router, length buckets, EP-size deployment choices, MoE-load monitoring, and NUMA policy appear to be internal/deployment-level items. |

## PR Mapping

| Article optimization point | Upstream item | Status | Correspondence |
| --- | --- | --- | --- |
| MiMo-V2.5 base model, multimodal support, MiMoV2MTP | [#23811: Xiaomi MiMo-V2.5 day0 support](https://github.com/sgl-project/sglang/pull/23811) | Merged 2026-04-30 | Registers MiMoV2, multimodal processor/model pieces, MiMoV2MTP draft model, and multi-layer EAGLE plumbing. |
| MiMo-V2.5 MTP deployment path | [#23945: enable MiMo V2.5 MTP cookbook path](https://github.com/sgl-project/sglang/pull/23945) | Merged 2026-04-28 | Documents the MiMo-V2.5 EAGLE/MTP serving path and benchmark configuration. |
| Multi-layer EAGLE / MTP weight loading | [#25748: yield filtered MTP weights lazily](https://github.com/sgl-project/sglang/pull/25748) | Merged 2026-05-20 | Fixes MiMo-V2.5-Pro style multi-layer EAGLE checkpoint loading OOM/hang. |
| MTP KV transfer in disaggregated serving | [#23539: missing index/KV transfer for MTP layer in NSA disaggregation](https://github.com/sgl-project/sglang/pull/23539) | Merged 2026-04-30 | Adds missing draft/MTP layer index/KV transfer in PD disaggregation. |
| MTP + HiCache / Mooncake adaptation | [#24984: support draft offload for Mooncake](https://github.com/sgl-project/sglang/pull/24984) | Open | Related to offloading draft/MTP state through HiCache/Mooncake. |
| Hybrid SWA dual KV pool | [#6563: Hybrid kv cache for LLaMA4](https://github.com/sgl-project/sglang/pull/6563) | Merged 2025-06-28 | Introduces global/full and local/SWA-style hybrid KV cache split. |
| SWA-aware prefix cache | [#7367: SWA Prefix Cache](https://github.com/sgl-project/sglang/pull/7367) | Merged 2025-07-13 | Directly matches the article's SWA prefix-cache tree direction. |
| SWA eviction during decode | [#17220: Evict swa kv cache during decoding](https://github.com/sgl-project/sglang/pull/17220) | Merged 2026-01-19 | Frees out-of-window SWA KV during long generation. |
| SWA eviction with CUDA graph | [#21754: Enable evict swa with piecewise cuda graph](https://github.com/sgl-project/sglang/pull/21754) | Merged 2026-03-31 | Allows SWA eviction with piecewise CUDA graph enabled. |
| Full-to-SWA mapping inside SWAKVPool | [#25824: Encapsulate SWA loc translation inside SWAKVPool](https://github.com/sgl-project/sglang/pull/25824) | Merged 2026-05-21 | Moves full-to-SWA loc translation into SWAKVPool and reduces caller-side side channels. |
| SWA mapping rebuild correctness | [#25889: DSV4 cached_loc invalidated when SWA mapping is rebuilt](https://github.com/sgl-project/sglang/pull/25889) | Merged 2026-05-21 | Fixes stale cached full-to-SWA translation after HiCache commit/loadback remaps. |
| Device/host consistency for SWA split nodes | [#25065: backup SWA-split parent before child under write-through](https://github.com/sgl-project/sglang/pull/25065) | Merged 2026-05-24 | Fixes UnifiedRadixCache write-through host backup for SWA-split nodes. |
| SWA prefix-cache LRU policy | [#26615: Window-aware LRU refresh for SWA prefix cache](https://github.com/sgl-project/sglang/pull/26615) | Open | Matches article's point that SWA hit validity depends on the active window, not just token equality. |
| SWA-aware cache event metadata | [#26579: Add SWA-aware KV event metadata](https://github.com/sgl-project/sglang/pull/26579) | Open | Adds SWA metadata to KV cache events for consumers to distinguish full and SWA validity. |
| SWA HiCache loadback host locking | [#26728: Fix SWA HiCache host locks during load back](https://github.com/sgl-project/sglang/pull/26728) | Open | Protects SWA host-side chunks from eviction while H2D loadback is in flight. |
| Layerwise KV loadback overlap with compute | [#23515: Layer-pipelined KV transfer](https://github.com/sgl-project/sglang/pull/23515) | Open | Closest upstream equivalent to the article's loadback-stream / compute-stream overlap diagram. |
| Decode-side prefix reuse in PD disaggregation | [#19746: support decode side radix cache](https://github.com/sgl-project/sglang/pull/19746) | Merged 2026-05-01 | Lets decode workers reuse shared prefixes and request only delta KV from prefill. |
| Decode-side radix cache for SWA hybrid models | [#26218: Support decode-side radix cache for SWA hybrid models](https://github.com/sgl-project/sglang/pull/26218) | Open | Extends decode-side radix cache to SWA hybrid models. |
| HiCache prefetch and incremental transfer on decode | [#26227: HiCache prefetching and PD incremental transfer on decode side](https://github.com/sgl-project/sglang/pull/26227) | Open | WIP for decode-side HiCache prefetch and incremental transfer. |
| PD decode SWA prealloc sizing | [#24036: Fix disagg decode SWA prealloc sizing](https://github.com/sgl-project/sglang/pull/24036) | Merged 2026-05-02 | Allocates full KV for the full prompt while sizing SWA KV only for the sliding-window tail. |
| PD decode SWA memory preallocation | [#24857: Optimize SWA memory preallocation for disaggregated decode](https://github.com/sgl-project/sglang/pull/24857) | Merged 2026-05-13 | Ports and refines SWA decode preallocation optimization. |
| Cache-aware routing by prefix | [#15935: PrefixHash load balancing policy](https://github.com/sgl-project/sglang/pull/15935) | Merged 2025-12-27 | Provides prefix-hash cache locality routing with load fallback. |
| PD cache-aware routing isolation | [#25184: Fix cache-aware policy pool isolation in PD mode](https://github.com/sgl-project/sglang/pull/25184) | Merged 2026-05-14 | Prevents prefill and decode pools from sharing/fighting over the same cache-aware trie. |
| Cache-aware routing load balance term | [#26293: Add cache balance weight](https://github.com/sgl-project/sglang/pull/26293) | Open | Adds a balance-weight knob so routing can trade off cache locality against load/cache-footprint skew. |
| Full chat history for cache-aware routing | [#26285: Use full chat history for PD cache-aware routing text](https://github.com/sgl-project/sglang/pull/26285) | Open | Improves multi-turn agent routing signal beyond the first chat message. |
| In-instance DP prefix-match routing | [#26612: prefix_match load balance for DP attention](https://github.com/sgl-project/sglang/pull/26612) | Open | Adds cache-aware routing inside plain DP-attention serving without an external gateway. |
| Decode disaggregation cache-aware DP routing | [#26561: cache-aware DP routing for decode disaggregation](https://github.com/sgl-project/sglang/pull/26561) | Open | Routes decode requests to the DP rank with best radix-cache prefix match. |
| HiRadixCache and cache-aware DP routing | [#26046: Add HiRadixCache and cache-aware DP routing](https://github.com/sgl-project/sglang/pull/26046) | Open | Combines hierarchical radix cache with decode-disagg DP cache-aware routing. |
| MiMo-V2 EPD disaggregation | [#24931: add EPD disaggregation support](https://github.com/sgl-project/sglang/pull/24931) | Merged 2026-05-18 | Splits MiMo encoder-only and language-only roles and adds encoder-server preprocessing hooks. |
| GPU image preprocessing and parallel video decode | [#25588: enable GPU image preprocess and parallel video decode](https://github.com/sgl-project/sglang/pull/25588) | Merged 2026-05-19 | Matches article's image GPU preprocessing and parallel video decoding bullets. |
| Encoder cross-request batching | [#25964: Cross-request batching for image/audio encoder](https://github.com/sgl-project/sglang/pull/25964) | Merged 2026-05-26 | Fuses concurrent image/audio encoder requests into one batch, improving encoder throughput. |
| Encoder data parallel mode | [#26576: encoder DP mode with per-rank subprocess workers](https://github.com/sgl-project/sglang/pull/26576) | Open | Matches article's Encoder DP deployment direction. |
| Prefill-side multimodal processing overlap | [#26708: pipeline mm_inputs processing with GPU forward in prefill](https://github.com/sgl-project/sglang/pull/26708) | Open | Overlaps multimodal input processing with GPU forward in disaggregated prefill. |
| VLM embed-path CUDA sync elimination | [#26082: eliminate CUDA syncs in VLM embed path](https://github.com/sgl-project/sglang/pull/26082) | Open | Removes syncs in VLM embedding path; listed in the EPD RFC as a performance item. |
| EPD encoder benchmark | [#24700: QPS-based encoder benchmark](https://github.com/sgl-project/sglang/pull/24700) | Open | Benchmark support for encoder-only `/encode` endpoint throughput and latency. |

## Related Issues and RFCs

| Topic | Upstream item | Status | Why it matters |
| --- | --- | --- | --- |
| Multimodal EPD optimization rollout | [#24945: SGLang EPD Performance / Architecture / Observability Enhancements](https://github.com/sgl-project/sglang/issues/24945) | Open | Xiaomi's article explicitly references this RFC and its PR list for EPD work. |
| PD disaggregation roadmap | [#21703: Prefill-Decode Disaggregation Roadmap](https://github.com/sgl-project/sglang/issues/21703) | Open | Tracks broader PD routing, EPD, and serving architecture work. |
| HiCache for hybrid/sparse LLMs | [#12826: HiCache for Hybrid and Sparse LLMs](https://github.com/sgl-project/sglang/issues/12826) | Closed 2026-02-07 | Historical umbrella issue for HiCache with hybrid/sparse/SWA-style models. |

## Article Points Without a Clear Single Upstream PR

| Article point | Likely status |
| --- | --- |
| GCache L3 storage service | Xiaomi internal infrastructure, not a single SGLang upstream PR. |
| Redis-backed stateless LLM-Router | Xiaomi internal/router-level system; upstream has related cache-aware routing PRs but not this exact design. |
| Prefill length buckets: 0-64K, 64K-256K, 256K-1M | Deployment/scheduling policy; no clear single upstream PR found. |
| Reducing EP size after SWA KV memory savings | Deployment topology choice; no clear single upstream PR found. |
| MoE load-balance monitoring without adding EPLB | Operational observation; no clear single upstream PR found. |
| Disabling Linux `numa_balancing` to remove kernel gaps | Deployment/system tuning; no clear SGLang PR found. |
