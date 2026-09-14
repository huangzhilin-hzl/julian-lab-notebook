# MLA 实现相关 PR 地图：FlashInfer、TensorRT-LLM、CUTLASS / CuTe DSL、TileLang、cuTile

核查日期：2026-09-14。将提问中的 `tiielang` 按 **TileLang** 理解；MLA 指 Multi-head Latent Attention。

这是实现主线与关键优化的筛选汇总，不是所有命中关键词的 PR 清单。以 GitHub PR REST API 的 `merged_at`、`state`、`base.ref` 核对状态，并结合 PR 正文、关键 diff、主干源码和文件历史判断功能范围。日期均为 UTC 合并日期；**合并到主干不等于已进入用户安装的发行版**。性能数据均为上游作者报告，本次没有运行 GPU benchmark。

**先分清三种计算形态。** DeepSeek-V2/V3 的 absorbed MLA decode 常见 `d_qk=512+64=576`、`d_vo=512`，共享 latent KV；unabsorbed MLA prefill 常见 `d_qk=128+64=192`、`d_vo=128`。因此标题写着“MLA”的 prefill shape PR，不能直接当作 compressed-KV decode 的实现。Sparse MLA 则额外处理被选中的 KV 索引；DSv4 HCA/CSA 又增加 SWA、压缩 cache、attention sinks 等不同的契约。

上述形态可对照 [FlashInfer #804](https://github.com/flashinfer-ai/flashinfer/pull/804)、[CUTLASS #2472](https://github.com/NVIDIA/cutlass/pull/2472)、[FlashInfer #3269](https://github.com/flashinfer-ai/flashinfer/pull/3269)。

| 项目 / 实现线 | 在生态中的位置 | 最值得读的内容 |
| --- | --- | --- |
| FlashInfer FA2/FA3 MLA | 自有 kernel + serving 算子接口 | latent KV 复用、warpgroup 分工、动态 split-K、paged/varlen、Graph |
| TensorRT-LLM / TRTLLM-gen / XQA | 推理运行时、生成 kernel 集成、架构专用路径 | FP8、multi-CTA、独立 reduction、cache reuse、chunked prefill、Helix |
| CUTLASS C++ / CuTe DSL | kernel 模板与 Python DSL 参考实现 | Blackwell TMA、Tensor Core/TMEM、persistent scheduler、pipeline 与 split-KV |
| TileLang | DSL 与可执行示例 | 从简洁 decode 到 paged、warp specialization、稀疏流水线和 Seesaw |
| cuTile / TileGym / FlashInfer cuTile | 编程语言、示例库、已接入的可选 serving 后端 | split-KV、编译调优、QK operand swap、运行时 wrapper 约束 |

这些项目不是五个互斥后端：FlashInfer 同时集成 CUTLASS、TRTLLM-gen、CuTe DSL 和 cuTile；TRT-LLM 也有独立 CuTe DSL FMHA library。下面的“阅读价值”是基于实现内容的归纳，不代表统一硬件和工作负载下的性能排名。

**FlashInfer：从自有 MLA kernel 到多后端运行时。**


| PR | 合并日期 / 状态 | 实现内容与阅读价值 |
| --- | --- | --- |
| [#551](https://github.com/flashinfer-ai/flashinfer/pull/551) | 2024-11-02 | 初始 MLA decode；讨论 matrix absorption 与 prefill/decode 的计算图差异。适合理解算法入口。 |
| [#804](https://github.com/flashinfer-ai/flashinfer/pull/804) | 2025-02-12 | 重做 memory-efficient fused MLA PageAttention；覆盖 decode、prefill、chunked prefill。通过 head-group fusion、QK/PV 不同维度分工，减少旧版重复 QK 与 KV 读取。 |
| [#813](https://github.com/flashinfer-ai/flashinfer/pull/813) | 2025-02-12 | 加入 CUDA Graph 兼容的 MLA API。与 kernel 首次跑通是不同的工程里程碑。 |
| [#863](https://github.com/flashinfer-ai/flashinfer/pull/863) | 2025-02-17 | 动态 split-K 与负载均衡 scheduler；改善 batch/head 较小时并行度不足。 |
| [#887](https://github.com/flashinfer-ai/flashinfer/pull/887) | 2025-02-23 | Hopper FA3 风格：异步 WGMMA、producer/consumer warp specialization，调整共享 latent KV 的多级流水。 |
| [#952](https://github.com/flashinfer-ai/flashinfer/pull/952) | 2025-03-29 | Hopper 从 3WG 调整为 FlashMLA 风格 2WG pipeline。适合比较寄存器、共享内存与流水重叠。 |
| [#1031](https://github.com/flashinfer-ai/flashinfer/pull/1031) | 2025-04-23 | 在 BatchMLAPagedAttentionWrapper 接入 Blackwell CUTLASS MLA backend；是 kernel 到 serving wrapper 的关键接点。 |
| [#1222](https://github.com/flashinfer-ai/flashinfer/pull/1222) | 2025-07-14 | 接入 TRTLLM-gen MLA cubin，最初范围明确为 decode。阅读重点是 launcher、参数与生成内核集成。 |
| [#1258](https://github.com/flashinfer-ai/flashinfer/pull/1258) | 2025-07-16 | TRTLLM-gen MLA 支持 q>1 / DeepSeek MTP；不要把单 token decode 支持等同于 speculative verify 支持。 |
| [#1339](https://github.com/flashinfer-ai/flashinfer/pull/1339) | 2025-08-09 | 融合 RoPE 与 FP8 quantize，覆盖 attention 主 kernel 之前的数据准备开销。 |
| [#2053](https://github.com/flashinfer-ai/flashinfer/pull/2053) | 2025-11-10 | 接入 XQA MLA backend 与测试；对应另一条架构专用 MLA 路径。 |
| [#2743](https://github.com/flashinfer-ai/flashinfer/pull/2743) | 2026-03-26 | 正式接入 Blackwell CuTe DSL MLA decode，BF16/FP16 与 FP8；通过布局重解释和编译缓存减少 Python 热路径开销。 |
| [#2805](https://github.com/flashinfer-ai/flashinfer/pull/2805) | 2026-04-14 | CuTe DSL 模块化：Loader、MMA、Softmax、Correction、Epilogue 角色与声明式 pipeline；同时包含 FMHA prefill 与 MLA decode。 |
| [#3235](https://github.com/flashinfer-ai/flashinfer/pull/3235) | 2026-05-11 | 支持 Kimi K2.5 的 H=64；以物理 128-head MMA 宽度填充 split-KV workspace，未采用实验性的 64-wide MMA-M 方案。 |
| [#3296](https://github.com/flashinfer-ai/flashinfer/pull/3296) | 2026-05-21 | 恢复 monolithic CuTe DSL MLA，与 modular 并存，用 cute_dsl_impl 选择；增加 modular sinks 路径。不能把 #2805 当作旧实现永久退出。 |
| [#3355](https://github.com/flashinfer-ai/flashinfer/pull/3355) | 2026-05-26 | 在 trtllm_batch_decode_with_kv_cache_mla 的 auto + autotune 模式比较 TRTLLM-gen 与 CuTe DSL；根据工作负载选择实现。 |
| [#3269](https://github.com/flashinfer-ai/flashinfer/pull/3269) | 2026-05-21 | DSv4 sparse MLA TRTLLM-gen：BF16 / per-tensor FP8、SWA 与 compressed KV pool、每 query 的 top-k 长度。 |
| [#3395](https://github.com/flashinfer-ai/flashinfer/pull/3395) | 2026-06-15 | SM120/121 sparse MLA：DSv3.2/GLM 与 DSv4 的 decode/prefill kernel，架构路线不同于 SM100/103 TRTLLM-gen。 |
| [#3694](https://github.com/flashinfer-ai/flashinfer/pull/3694) | 2026-06-25 | SM90 FA3 原生读取 FP8 KV：KV 在 HBM/L2/shared memory 保持 FP8，再在共享内存 staging 中反量化，WGMMA 仍为 BF16×BF16。不是全 FP8 算术。 |
| [#3943](https://github.com/flashinfer-ai/flashinfer/pull/3943) | 2026-08-05 | CuTe DSL DSv4 FP8 HCA：SM100/103、2-CTA、split-K、persistent scheduling；SWA 使用绝对行索引，compressed cache 使用物理页表。 |
| [#4178](https://github.com/flashinfer-ai/flashinfer/pull/4178) | 2026-07-28 | packed low-head 与 variable-Q decode；扩展不同 head 数和 query 打包方式的适配。 |
| [#4697](https://github.com/flashinfer-ai/flashinfer/pull/4697) | 2026-08-28 | 隔离 planned FA2、FA3、CUTLASS backend，重构 plan/run 所有权。 |
| [#4719](https://github.com/flashinfer-ai/flashinfer/pull/4719) | 2026-08-31 | CuTe DSL variable-Q decode 与 DCP 组合支持；属于分布式和变长查询的工程扩展。 |
| [#4018](https://github.com/flashinfer-ai/flashinfer/pull/4018) | 2026-09-04 | cuTile 接入现有 attention wrapper，包含 compressed latent KV 的 paged MLA decode；显式 backend="cutile"。详见下文。 |


FlashInfer 近期仍在推进的 PR：


| PR | 合并日期 / 状态 | 实现内容与阅读价值 |
| --- | --- | --- |
| [#4906](https://github.com/flashinfer-ai/flashinfer/pull/4906) | Open，未合并 | SM90 小 M decode 的 swap-AB Q tile 与并行 split-KV merge；适合关注 TP 后少 head、低延迟场景，但尚未合并。 |
| [#5041](https://github.com/flashinfer-ai/flashinfer/pull/5041) | Open，未合并 | MLA CUDA Graph plan update API；截至核查时仍未合并。 |
| [#5070](https://github.com/flashinfer-ai/flashinfer/pull/5070) | Open，未合并 | 把 TRTLLM-gen、XQA、CuTe DSL 纳入统一 planned wrapper；已有直接 API 不等于这一层统一 plan/run 已完成。 |


**TensorRT-LLM：量化 kernel 与 serving 功能共同演进。**


| PR | 合并日期 / 状态 | 实现内容与阅读价值 |
| --- | --- | --- |
| [#3190](https://github.com/NVIDIA/TensorRT-LLM/pull/3190) | 2025-04-07 | Hopper / Blackwell FP8 MLA；Q 与 latent KV per-tensor E4M3，BF16 输出。早期 recipe 的 scale 固定为 1，不能泛化为完整校准方案。 |
| [#3752](https://github.com/NVIDIA/TensorRT-LLM/pull/3752) | 2025-04-23 | 加入 QMMA-based MLA kernels；属于 TensorRT-LLM 生成内核路径的实现更新。 |
| [#4535](https://github.com/NVIDIA/TensorRT-LLM/pull/4535) | 2025-05-23 | MLA FP8 KV cache reuse，与运行时复用流程优化。 |
| [#4858](https://github.com/NVIDIA/TensorRT-LLM/pull/4858) | 2025-06-06 | SM120 XQA-based MLA 正式合并项；同名 #4856 已关闭未合并。 |
| [#4467](https://github.com/NVIDIA/TensorRT-LLM/pull/4467) | 2025-06-17 | MLA piecewise CUDA Graph 支持。 |
| [#5426](https://github.com/NVIDIA/TensorRT-LLM/pull/5426) | 2025-06-25 | 高吞吐 MLA 的 multiCtasKvMode 与 heuristic 更新；让多 CTA 分担 KV 方向工作。 |
| [#4651](https://github.com/NVIDIA/TensorRT-LLM/pull/4651) | 2025-06-26 | Blackwell MLA chunked prefill；分块处理 KV，并通过 LSE 合并 attention 输出。 |
| [#6655](https://github.com/NVIDIA/TensorRT-LLM/pull/6655) | 2025-08-14 | Hopper MLA chunked prefill。 |
| [#7597](https://github.com/NVIDIA/TensorRT-LLM/pull/7597) | 2025-09-09 | 大 reduction tile 改用独立 reduction kernel；同名 #7536 未合并，最终应引用本项。 |
| [#8692](https://github.com/NVIDIA/TensorRT-LLM/pull/8692) | 2025-10-31 | DSv3.2 TRTLLM-gen sparse MLA；覆盖 FP8 / NVFP4 flow 与 BF16 / per-tensor FP8 KV cache 组合。 |
| [#8104](https://github.com/NVIDIA/TensorRT-LLM/pull/8104) | 2025-11-04 | MLA 完整 Helix 支持，包含 position ID 与 post-process kernel。 |
| [#10813](https://github.com/NVIDIA/TensorRT-LLM/pull/10813) | 2026-01-26 | Blackwell Skip Softmax MLA，以及 NVFP4 KV scale 的精度修复。 |
| [#15409](https://github.com/NVIDIA/TensorRT-LLM/pull/15409) | 2026-06-27 | DSv4 sparse MLA attention backend；代码落在 sparse/deepseek_v4，含 cache manager、索引变换及测试。 |
| [#15138](https://github.com/NVIDIA/TensorRT-LLM/pull/15138) | 2026-08-05 | CuTe DSL decode 作为 TRTLLM attention backend 内部 FMHA library 接入，SM100/103，FP16/BF16/FP8；带形状与性能准入条件。 |
| [#18131](https://github.com/NVIDIA/TensorRT-LLM/pull/18131) | 2026-08-28 | 为独立 CuTe DSL MLA FMHA 路线补齐 Helix 支持。 |
| [#18653](https://github.com/NVIDIA/TensorRT-LLM/pull/18653) | 2026-09-08 | 撤回 #17800 的 FlashInfer→CuTeDSL dispatch，保留独立 CuTeDSL FMHA 路径及后续 Helix/DSA 改动。不是删除全部 CuTe DSL MLA 支持。 |


`TRTLLM-gen` 相关 PR 有些主要更新 cubin、kernel metadata 和 launcher。它们对集成、选型与调度很有价值，但若目标是逐行学习可修改的 kernel mainloop，CUTLASS / CuTe DSL、FlashInfer FA3、TileLang 和 FlashMLA 的源码更直接。可对照 [FlashInfer #1222](https://github.com/flashinfer-ai/flashinfer/pull/1222) 与 [TRT-LLM #7597](https://github.com/NVIDIA/TensorRT-LLM/pull/7597) 的改动文件。

**CUTLASS / CuTe DSL：需要沿文件历史查版本 PR。**


| PR | 合并日期 / 状态 | 实现内容与阅读价值 |
| --- | --- | --- |
| [#2130](https://github.com/NVIDIA/cutlass/pull/2130) | 2025-02-24；目标 `Deepseek` | 早期把 DeepSeek FlashMLA 移植为 CUTLASS C++ example；合入 Deepseek 分支，正文还列有 reference check 等 TODO。不能当作 main 的 Blackwell MLA 上线点。 |
| [#2134](https://github.com/NVIDIA/cutlass/pull/2134) | 2025-02-26；目标 `Deepseek` | 上述分支移植的第二步，补输入输出处理等。 |
| [#2213](https://github.com/NVIDIA/cutlass/pull/2213) | 2025-04-03 | v3.9 整包更新。77_blackwell_fmha/77_blackwell_mla.cu 的主干文件历史从此 PR 开始；是 Blackwell C++ MLA 的重要入口。 |
| [#2472](https://github.com/NVIDIA/cutlass/pull/2472) | 2025-07-18 | Blackwell MLA prefill：d_qk=192、d_vo=128，FP16/FP8；修改 TMA pipeline、共享内存/寄存器和 causal tile scheduler。不是 576/512 absorbed decode。 |
| [#2466](https://github.com/NVIDIA/cutlass/pull/2466) | 2025-07-24 | 相同 192/128 形态的 backward，FP16/FP8；推理 kernel 调研可作为布局补充阅读。 |
| [#2709](https://github.com/NVIDIA/cutlass/pull/2709) | 2025-10-21 | v4.3 整包更新；旧路径 examples/python/CuTeDSL/blackwell/mla.py 在该 PR 进入公开主干。标题不含 MLA。 |
| [#3032](https://github.com/NVIDIA/cutlass/pull/3032) | 2026-02-14 | v4.4 tag 整包更新；公开的 mla/mla_decode_fp16.py 分文件路线可追到此处，并包含相邻 FP8 实现。 |
| [#3202](https://github.com/NVIDIA/cutlass/pull/3202) | 2026-05-06 | v4.5 tag 更新；当前 CuTeDSL/cute/blackwell/kernel/attention/mla 路径的文件历史由此衔接。不是 MLA 功能首次出现。 |
| [#3224](https://github.com/NVIDIA/cutlass/pull/3224) | 2026-05-13 | 修复 CuTe DSL MLA decode 的 Thor 支持；反映同为 Blackwell 也需要正确的架构 dispatch。 |
| [#3357](https://github.com/NVIDIA/cutlass/pull/3357) | Open，未合并 | 尚未合并的公开示例更新：FP8 16-warp、QK/PV 分离 issue warp、双 softmax warpgroup，FP16 fold_sq 与多 query 调度。 |


还需注意 [#2366](https://github.com/NVIDIA/cutlass/pull/2366) 虽已合并，但次日被 [#2370](https://github.com/NVIDIA/cutlass/pull/2370) revert；不能单独作为当前已生效能力的依据。

CuTe DSL 当前 FP16 MLA 示例明确展示 TMA、Blackwell Tensor Core、warp specialization、persistent scheduler、paged/variable KV 与 split-KV + reduction。它是研究 Blackwell 实现的好入口，但示例约束不等于下游集成的能力矩阵。

**TileLang：从 dense decode 示例到 sparse MLA 流水线。**


| PR | 合并日期 / 状态 | 实现内容与阅读价值 |
| --- | --- | --- |
| [#109](https://github.com/tile-ai/tilelang/pull/109) | 2025-02-23 | 加入 MLA / GQA decode 示例，作为起点。 |
| [#120](https://github.com/tile-ai/tilelang/pull/120) | 2025-02-25 | 整理 Q/Q_pe、KV/K_pe 分离及共享 buffer、split combine 逻辑。 |
| [#134](https://github.com/tile-ai/tilelang/pull/134) | 2025-03-03 | DeepSeek MLA 示例、教程与 benchmark；讲解 swizzling、layout inference、pipeline 等优化。 |
| [#158](https://github.com/tile-ai/tilelang/pull/158) | 2025-03-06 | 加入 paged MLA decode 与 split-KV、combine、benchmark，是最贴近 serving KV layout 的早期示例。 |
| [#165](https://github.com/tile-ai/tilelang/pull/165) | 2025-03-07 | PV 改用 SS-GEMM，减少 T.copy 后再走 RS-GEMM 的路径。 |
| [#363](https://github.com/tile-ai/tilelang/pull/363) | 2025-04-09 | AMD MLA 实现；跨平台部分的起点。 |
| [#366](https://github.com/tile-ai/tilelang/pull/366) | 2025-04-10 | AMD FlashMLA 的 num-split template 扩展。 |
| [#896](https://github.com/tile-ai/tilelang/pull/896) | 2025-09-29 | DeepSeek-V3.2 sparse MLA forward / pipelined 示例及 indexer logits 相关文件。 |
| [#901](https://github.com/tile-ai/tilelang/pull/901) | 2025-09-29 | 增加 Top-K 选择与文档，把稀疏索引生成与 attention 示例连接起来。 |
| [#928](https://github.com/tile-ai/tilelang/pull/928) | 2025-10-01 | 单独的 MLA decode warp-specialized 示例。 |
| [#1636](https://github.com/tile-ai/tilelang/pull/1636) | 2026-01-08 | Seesaw sparse MLA：两个 consumer 分别处理奇偶 KV block，交换 softmax statistics 和概率块，各自计算输出维度的一半，重用 shared memory。 |
| [#3224](https://github.com/tile-ai/tilelang/pull/3224) | Open，未合并 | 新开的 Hopper FP8 sparse MLA forward：关注 FP8 WGMMA 的 K-major 约束与 PV 的 shared-memory transpose；尚未合并。 |


TileLang 的代表性贡献是可以完整阅读、修改和测量的 kernel 示例。`examples/deepseek_mla` 和 `examples/deepseek_v32` 中的结果，不自动证明任意推理框架已接入同样的页表、Graph、batch scheduler 或量化格式。尤其不要把 [#1636](https://github.com/tile-ai/tilelang/pull/1636) 的稀疏 Seesaw 与通用 dense decode 混为同一种 workload。

**cuTile：实现主要在 TileGym 与 FlashInfer，不能只搜 cutile-python。**

对 `NVIDIA/cutile-python` 的 MLA PR 搜索本次没有结果；这仅说明没有检索到匹配 PR，不代表 cuTile 不能实现 MLA。TileGym 主干已有 `mla.py`、`mla_decoding.py`、`mla_decoding_split_kv.py`。split-KV 文件历史可追溯到仓库初始提交，因此不存在必须引用的“首次 MLA PR”。


| PR | 合并日期 / 状态 | 实现内容与阅读价值 |
| --- | --- | --- |
| [#45](https://github.com/NVIDIA/TileGym/pull/45) | 2026-01-31 | 修复 MLA num_kv_split 可能计算为 0 的问题，确保至少 1 个 split。适合看 split-KV heuristic 的边界。 |
| [#91](https://github.com/NVIDIA/TileGym/pull/91) | 2026-04-02 | experimental sparse MLA forward，按索引只读取 top-k KV；有 correctness 与 benchmark。依然标为 experimental。 |
| [#151](https://github.com/NVIDIA/TileGym/pull/151) | 2026-06-10 | 包含 dense MLA autotune config 更新；是调优更新，不是首次加入 MLA。 |
| [#166](https://github.com/NVIDIA/TileGym/pull/166) | 2026-07-13 | 包含 split-KV 测试参数与其他维护更新；不应包装成新 kernel 实现。 |


[FlashInfer #4018](https://github.com/flashinfer-ai/flashinfer/pull/4018) 于 **2026-09-04 合并**，是 cuTile 从示例走向现有推理算子 API 的关键 PR。它提供 `BatchMLAPagedAttentionWrapper(backend="cutile")`，实现 paged compressed-KV decode、split-KV 和可捕获的 `run()`。

按本次核查的主干 `cutile_backend.py`，wrapper 的具体边界为：SM100/103/120/121；Q/KV/output 同为 FP16 或 BF16；latent/RoPE 维度固定为 512/64；每请求一个 query；不接受 `causal=True` 和量化 output scale。这里是 wrapper 的具体参数约束，不能扩写成通用 FP8 MLA 或多 token verify 支持。

该 PR 报告的 B200、H=32 MLA decode 测试中，page=64 且 batch≥128 的子集相对 TRTLLM-gen 为 1.17–1.28×；但完整 page{32,64} × batch{1…256} × KV{1k…8k} 集合的几何平均为 **0.85×**（大于 1 表示 cuTile 更快）。因此只能说某些吞吐场景已具竞争力，不能说全面优于 TRTLLM-gen。以上均为作者报告，非本次复测。

**补充 FlashMLA：是 Hopper 对照和多条实现线的重要来源。**


| PR | 合并日期 / 状态 | 实现内容与阅读价值 |
| --- | --- | --- |
| [#71](https://github.com/deepseek-ai/FlashMLA/pull/71) | 2025-04-22 | 2025-04 performance update，针对 compute-bound MLA workload 的优化，附实现深入说明。 |
| [#98](https://github.com/deepseek-ai/FlashMLA/pull/98) | 2025-09-29 | 真正合并的 Hopper sparse attention：DSv3.2 sparse prefill + FP8 KV paged decode。不要误引已关闭未合并的 #101。 |


**如果目的是读实现，建议按问题选择 PR。**

| 想解决的问题 | 阅读顺序 | 应盯住的代码设计 |
| --- | --- | --- |
| MLA 与普通 FlashAttention 有什么不同 | FlashInfer #551 → #804 | latent KV 同时作为 K/V，QK 的 RoPE 分量，512 维输出带来的寄存器压力 |
| Hopper 如何提高利用率 | FlashInfer #863 → #887 → #952 | split-K、WGMMA、共享内存复用、2WG/3WG 分工 |
| Blackwell 低层 kernel 怎么写 | CUTLASS #2213 → #3032 → FlashInfer #2743 | TMA、Tensor Core/TMEM、2-CTA、persistent scheduling、split-KV reduction |
| 如何组织可维护的 CuTe DSL attention | FlashInfer #2805 → #3296 | role/pipeline 抽象，monolithic/modular 的功能与性能边界 |
| TP 后少 head / 多 token decode | FlashInfer #3235、#4178、#4719；跟踪 #4906 | 物理 tile 宽度与逻辑 head 数、packed Q、DCP、operand swap |
| 稀疏 gather 与流水线 | FlashMLA #98 → TileLang #896 → #1636 | index prefetch、KV gather、双 consumer、softmax statistics 交换 |
| DSv4 HCA | FlashInfer #3269 → #3943 | SWA 与压缩 KV 的不同索引契约、sinks、split-K 与 metadata |
| cuTile 能否用于现有服务 | TileGym split-KV 源码 → FlashInfer #4018 | compiler hints、QK operand swap、page/head/batch 范围、wrapper 校验 |

共同的性能主线可以归纳为：**避免重复读取 latent KV；把宽输出与少 query/head 映射到合适的 tensor-core tile；用 split-KV 补并行度；重叠 gather/TMA、MMA 与 softmax；最后处理量化、Graph、varlen、MTP 和分布式等组合。** 这是对上述实现的归纳，不是某个 PR 独有的算法。

**已固定到核查 commit 的源码入口。**

| 实现 | 源码 |
| --- | --- |
| CUTLASS Blackwell C++ MLA | [77_blackwell_mla.cu](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/77_blackwell_fmha/77_blackwell_mla.cu) |
| CUTLASS CuTe DSL FP16 MLA | [mla_decode_fp16.py](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/python/CuTeDSL/cute/blackwell/kernel/attention/mla/mla_decode_fp16.py) |
| CUTLASS CuTe DSL FP8 MLA | [mla_decode_fp8.py](https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/examples/python/CuTeDSL/cute/blackwell/kernel/attention/mla/mla_decode_fp8.py) |
| TileLang paged MLA | [example_mla_decode_paged.py](https://github.com/tile-ai/tilelang/blob/030556de976356b21061d413843b4a29f75b20a8/examples/deepseek_mla/example_mla_decode_paged.py) |
| TileLang Seesaw | [sparse_mla_fwd_seesaw.py](https://github.com/tile-ai/tilelang/blob/030556de976356b21061d413843b4a29f75b20a8/examples/deepseek_v32/sparse_mla_fwd_seesaw.py) |
| TileGym cuTile split-KV | [mla_decoding_split_kv.py](https://github.com/NVIDIA/TileGym/blob/ff2a52ec4895764cb48473312164471fd031021d/src/tilegym/ops/cutile/mla_decoding_split_kv.py) |
| FlashInfer cuTile wrapper | [cutile_backend.py](https://github.com/flashinfer-ai/flashinfer/blob/5d0c89eacae6ca08f2a1ce92eba557bbad7a1bfc/flashinfer/mla/_batch_mla/_backends/cutile_backend.py) |

**容易误读的状态，集中留档。**

| 项目 / PR | 本次核查结论 |
| --- | --- |
| FlashInfer #766 | 关闭未合并，不能作为 SM80 实现已合并的证据 |
| FlashInfer #1248、#1273 | 均关闭未合并；one-copy/dedup KV 的最终落点需要另查，不能按标题认定落地 |
| FlashInfer #5070、#5041、#4906 | Open，不能计入现有主干能力 |
| TRT-LLM #3004 | 关闭未合并；FP8 MLA 应优先读 #3190 |
| TRT-LLM #4856、#7536 | 同名未合并项；分别使用 #4858、#7597 |
| TRT-LLM #15333、#17266 | 关闭未合并；CuTe DSL FMHA 正式入口是 #15138 |
| TRT-LLM #18653 | 只撤回一条 dispatch，保留独立 CuTe DSL MLA FMHA |
| CUTLASS #2130、#2134 | 合并目标为 Deepseek 分支，不是 main |
| CUTLASS #2366 | 已合并后被 #2370 revert |
| CUTLASS #3357、TileLang #3224 | Open，作为后续跟踪项 |
| FlashMLA #101、#186 | 关闭未合并，不能据此认定 sparse MLA 或 BF16 DSv4 新路径已落地主干 |

本次保存了 PR 元数据索引及相关文件历史，用于复查同名 PR、合并目标、revert 和批量版本更新。索引涵盖调研候选，不代表全部候选均推荐阅读。

证据文件：[PR 元数据索引](evidence/20260914/pr-index.json)、[源码文件历史](evidence/20260914/file-histories.json)。
