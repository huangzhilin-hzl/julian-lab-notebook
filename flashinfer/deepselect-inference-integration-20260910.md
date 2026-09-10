# DeepSelect 与推理框架 TopK 集成进展（2026-09-10）

> 核查快照：2026-09-10 15:10，Asia/Shanghai。本文日期均为北京时间；“已合并”指进入主干，不等同于已进入框架稳定发行版。性能数字来自上游 README、PR 正文或讨论，本次未进行本地 GPU 复测。

## 结论

截至核查时间，**尚未检索到 SGLang、vLLM、TensorRT-LLM、FlashInfer、LMDeploy 明确接入 DeepSelect 的公开 PR 或主干代码引用**。DeepSelect 官方于当天宣布发布 v1.0.0，自身仓库的全部 PR 列表为空。

已经落地的相关工作包括 SGLang 的 FlashInfer TopK 后端、vLLM 原生 cooperative TopK 调优，以及 TensorRT-LLM GVR V2 向 FlashInfer 的移植。这些工作都早于 DeepSelect 发布，不能计为 DeepSelect 的框架集成。

来源：[DeepSelect 官方仓库](https://github.com/deepseek-ai/DeepSelect)、[官方 PR API](https://api.github.com/repos/deepseek-ai/DeepSelect/pulls?state=all&per_page=100)。

## DeepSelect 提供什么

DeepSelect 是用于 DeepSeek Sparse Attention（DSA）和采样场景的 TopK 内核库。官方 README 列出的目标模型包括 DeepSeek V3.2、V4、V4.1，主要覆盖：

| 场景 | 输入类型与范围 | 接口特征 |
| --- | --- | --- |
| Lightning Indexer | BF16，优化不同 batch size 和行宽，K ≤ 4096 | 可只返回索引；可选择索引排序或值排序。 |
| Sampling | FP32，重点覆盖约 128K 词表，K ≤ 4096 | 可返回 TopK 值与索引。 |

`deep_select.topk` 支持逐行有效长度上界 `end` 和调用方提供的 `output_idx`。输入行 stride 与输出缓冲区存在对齐要求，因此框架接入仍需处理张量布局、输出语义及缓冲区生命周期。

官方给出的 **2–20×** 是相对于相同输入下的 `torch.topk` 的算子基准。它不代表相对 FlashInfer、SGLang JIT TopK 或 vLLM cooperative TopK 的收益，也不代表端到端推理提速。

来源：[README 支持范围与性能说明](https://github.com/deepseek-ai/DeepSelect#supported-cases)、[Python 接口](https://github.com/deepseek-ai/DeepSelect/blob/main/deep_select/interface.py)。

## 相关 PR 状态

以下均为现有 DSA／TopK 实现的优化或其他后端接入，未发现与 DeepSelect 的直接集成关系。

| 框架／组件 | PR | 状态 | 进展与边界 |
| --- | --- | --- | --- |
| SGLang | [#33237：DeepSeek V4 FlashInfer TopK 后端](https://github.com/sgl-project/sglang/pull/33237) | 已合并，9/1 | 支持 `--dsa-topk-backend flashinfer`，接入融合 TopK 与 compact-page 转换；要求 FlashInfer ≥ 0.6.18。 |
| SGLang | [#37970：SM100 cooperative indexer TopK](https://github.com/sgl-project/sglang/pull/37970) | Open，未合并 | 添加 grid-wide cooperative TopK，默认关闭；针对小 batch、长上下文。作者报告主要收益在 GLM，DeepSeek V4 的端到端收益较小。 |
| vLLM | [#55872：可选的确定性 FlashInfer TopK 后端](https://github.com/vllm-project/vllm/pull/55872) | Open，未合并 | 针对 TopK 边界值相同时的确定性选择，保留原生默认后端；当前有合并冲突，讨论中还有 GB10 兼容性反馈。 |
| vLLM | [#53382：中等 batch 的 cooperative TopK 调优](https://github.com/vllm-project/vllm/pull/53382) | 已合并，9/1 | 调优原生内核。PR 报告在 BS=33/40/48/64、K=512/1024/2048、64K–250K 行宽下，相对 FilteredTopK 的算子级提速约 1.37–3.28×。 |
| TensorRT-LLM | [#17821：GVR V2 self-sampling TopK](https://github.com/NVIDIA/TensorRT-LLM/pull/17821) | 已合并，8/26 | 接入 DSA decode 路径；通过 `TRTLLM_GVR_SELF_SAMPLING=1` 配合 heuristic TopK 配置及硬件、布局条件启用。 |
| FlashInfer | [#4811：移植 TRT-LLM GVR V2](https://github.com/flashinfer-ai/flashinfer/pull/4811) | 已合并，9/5 | 为 `top_k_varlen` 引入 `gvr_2` 后端，同时更新 `auto` 选择策略并修复原有 GVR V1 的阈值搜索问题。 |
| FlashInfer | [#4986：GVR V2 调度与无 hint 支持](https://github.com/flashinfer-ai/flashinfer/pull/4986) | Open，未合并 | 延续 #4811，优化 4K–8K 行宽、按 GPU SM 数调整调度，并支持无 `pre_idx` hint 的调用及自动选择。 |

LMDeploy 的 `DeepSelect`／`deep_select` 定向检索没有命中；本文未进一步盘点其所有通用 TopK 优化。

## FlashInfer 相关主线

### TensorRT-LLM → FlashInfer 的来源关系已经明确

[TensorRT-LLM #17821](https://github.com/NVIDIA/TensorRT-LLM/pull/17821) 提供 GVR V2 self-sampling TopK。它从当前 logits 行的稀疏采样中估计阈值，再验证和细化；上一轮 TopK 索引作为 hint，不直接决定结果正确性。该 PR 同时提供按设备上逐行长度处理的 `run_varlen`，并接入 DSA decode。

[FlashInfer #4811](https://github.com/flashinfer-ai/flashinfer/pull/4811) 明确以该 TRT-LLM 实现为来源，移植设备内核与 host dispatch，加入 FlashInfer 的编译缓存和后端管理。在该 PR 描述的接口范围内，`gvr_2` 面向 FP32、K ∈ {512, 1024, 2048}，并要求 `pre_idx`；无 hint 支持属于后续 #4986。

因此，这条可以确认的来源链是 **TensorRT-LLM GVR V2 → FlashInfer `top_k_varlen/gvr_2`**。没有证据把 DeepSelect 放入这条移植链。

### SGLang 消费 FlashInfer TopK 的接口已经落地

[SGLang #33237](https://github.com/sgl-project/sglang/pull/33237) 使用 FlashInfer 的 `top_k_page_table_transform`，支持 DeepSeek V4 中 `page_size=64` 的 compact page table、带 padding 的 score stride、调用方提供的输出缓冲区，以及可选 raw-index 输出。

开启 `SGLANG_DSA_FUSE_TOPK=true` 时使用融合转换；关闭时保留已有的非融合 FlashInfer 路径。PR 记录了 B300 上的单元测试和 CUDA Graph 验证，但明确没有完成 DeepSeek V4 模型 serving 精度测试或合并后的性能重测。

这里消费的是 FlashInfer TopK／页表转换接口，不能仅因为同属 FlashInfer，就推断该路径已经调用 #4811 的 `top_k_varlen/gvr_2`，更不能推断已经调用 DeepSelect。

### vLLM 的确定性后端仍有待处理问题

[vLLM #55872](https://github.com/vllm-project/vllm/pull/55872) 当前仍开放，机器人提示需要解决合并冲突。讨论中，测试者报告 FlashInfer 0.6.18 的相关确定性路径在 GB10 上无法启动：其所需每 SM 共享内存为 131072 字节，而测试设备提供 102400 字节。

这些是该 PR 及特定 FlashInfer 路径的反馈，不应外推为 DeepSelect 的硬件限制。另有讨论指出，TopK 对相同输入的确定性选择，也不能消除上游 score 计算差异导致的端到端输出变化。

来源：[GB10 测试反馈](https://github.com/vllm-project/vllm/pull/55872#issuecomment-5603071616)、[进一步的确定性实验](https://github.com/vllm-project/vllm/pull/55872#issuecomment-5604003805)、[合并冲突提示](https://github.com/vllm-project/vllm/pull/55872#issuecomment-5604013184)。

## 如何判断后续是否真正完成 DeepSelect 集成

后续更新应分别记录以下证据，而不能只凭 PR 标题出现 “TopK” 或 “DeepSeek” 判断：

1. **接入证据**：依赖、vendored kernel 或实际调用链明确指向 DeepSelect；确认覆盖 indexer、sampling 中的哪些路径。
2. **框架适配**：确认变长行、stride、索引填充值、页表转换、CUDA Graph 和输出 buffer 生命周期符合框架要求。
3. **合入与发行**：分别记录 PR 是否合并、是否默认启用，以及首个包含它的框架版本。
4. **有效收益**：与框架当前优化后端在相同 GPU、dtype、K、batch 和上下文长度下比较，并补充端到端吞吐、TTFT／TPOT 及模型正确性结果。

## 核查方法与范围

- 通过 GitHub 搜索检索 `DeepSelect`、`deep_select`，覆盖 PR 标题、正文、评论和代码；定向仓库为 SGLang、vLLM、TensorRT-LLM、FlashInfer、LMDeploy。
- 直接读取 SGLang、vLLM、FlashInfer、TensorRT-LLM 最新各 100 个 PR 的标题与正文，补查 `deepselect`、`deep_select`、`deep-select`，均无命中。
- 直接读取 DeepSelect 全部 PR 列表，结果为空；逐一读取表中相关 PR 元数据和正文，并读取 vLLM #55872 的讨论。
- “未检索到”是上述公开范围与核查时间下的结论，不能排除未公开分支、尚未建立 PR 的工作或搜索索引延迟。
