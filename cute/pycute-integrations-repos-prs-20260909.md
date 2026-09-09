# PyCuTe 与 cuTile：集成仓库、PR 和进展

核查日期：2026-09-09。文中 PR 合并日期按 GitHub 返回的 UTC 时间记录。

PyCuTe 部分来源是 Cris Cecka 在 GPU MODE #114（2026-09-04）的演讲材料。需要区分 **CuTe IR / cuda.lang 本身的实现**与 **PyCuTe 对这些项目的专门适配**；下面列出的相关 PR 不能直接作为 PyCuTe 集成已经完成的证据。

本文另记录 **cuTile Python（`cuda.tile`）接入 FlashInfer** 的进展。cuTile Python 与实验性的 `cuda.lang` 同在 `NVIDIA/cutile-python` 仓库，但 FlashInfer 的 cuTile 后端使用的是 `cuda.tile`，不属于 PyCuTe 专门适配或 cuda.lang 的生产集成。

## 仓库与 PR 对应关系

| PPT 条目 | 对应仓库及代码目录 | 已查到的相关 PR | 核查时的状态与范围 |
|---|---|---|---|
| **CuTeIR** | [NVIDIA/cutlass — cutlass_compiler/cute_ir](https://github.com/NVIDIA/cutlass/tree/main/cutlass_compiler/cute_ir) | [#3426：Cutlass compiler: Introducing CuTe IR MLIR dialect](https://github.com/NVIDIA/cutlass/pull/3426) | **已合并，2026-08-03**。引入 CuTe IR MLIR dialect、Cutegen 类型推导与 lowering 后端，以及 Base lowering 基础设施。这是 CuTe IR 的引入 PR，并非专门的 PyCuTe 接入 PR。 |
| **cuda.lang（enable meta-programming）** | [NVIDIA/cutile-python — experimental/cuda-lang](https://github.com/NVIDIA/cutile-python/tree/main/experimental/cuda-lang) | [#100：[lang][nfc] Add execution and metaprogramming documentation](https://github.com/NVIDIA/cutile-python/pull/100) | **Open / Draft**，最后更新于 2026-08-26。补充执行模型和元编程文档，介绍 `static_eval`、`static_iter`、`static_assert`、`ensure_constant`、`static_def` 等能力；属于文档改动，不实现 PyCuTe 集成。 |

## cuda.lang 中相关的功能提交

以下提交与 PPT 第 12 页展示的适配机制相关，但它们本身不是 PyCuTe 的适配补丁。

| 功能 | 提交 | 与 PPT 的对应关系 |
|---|---|---|
| `@static_def` 与编译期求值语义 | [`a8e3c0b`：[lang] Rename @metafunction to @static_def, use static_eval semantics](https://github.com/NVIDIA/cutile-python/commit/a8e3c0b1f3619ec7b4637fc4e98c0d15a811746c) | 将 `@metafunction` 改为 `@static_def`，采用 `static_eval` 语义。对应显式声明 trace-time 元编程函数的机制。 |
| frozen dataclass 作为 kernel 参数 | [`cc98c05`：Allow frozen dataclasses as kernel arguments](https://github.com/NVIDIA/cutile-python/commit/cc98c055622fbd98c4471f23c05cc4118873fa40) | 对应以 frozen dataclass 表示值类型并传入 kernel 的支持，与 PPT 中将布局对象改成 frozen dataclass 的方向相关。 |

核查时，这两个 commit 的 GitHub 关联 PR 查询均返回空列表，因此这里记录功能提交链接，不推定它们对应某个公开 PR。

## PyCuTe 专门适配的公开状态

PyCuTe 的参考实现仓库是 [NVlabs/CuTe](https://github.com/NVlabs/CuTe)。本次检查的主分支快照为 [`5c39482690ff9e08219cfae7a86390caa3d84b62`](https://github.com/NVlabs/CuTe/tree/5c39482690ff9e08219cfae7a86390caa3d84b62)。

PPT 第 12 页展示的 cuda.lang 适配包括：

- 给 `pycute/algebra.py` 的 `zipped_divide` 等函数添加 `@_cuda_lang_static_def`。
- 给 `pycute/layout.py` 的 `Layout.__call__` 添加相同的装饰器。
- 将 `Layout` 改为 `@dataclass(frozen=True, init=False)`。

上述改动尚未出现在本次检查的公开快照对应代码中：

- [algebra.py:355](https://github.com/NVlabs/CuTe/blob/5c39482690ff9e08219cfae7a86390caa3d84b62/pycute/algebra.py#L355)
- [layout.py:45](https://github.com/NVlabs/CuTe/blob/5c39482690ff9e08219cfae7a86390caa3d84b62/pycute/layout.py#L45)

本次未找到能够明确对应 PPT 中 **PyCuTe → CuTeIR** 或 **PyCuTe → cuda.lang** 专门集成工作的独立公开 PR。这是本次公开检索的结果，不代表不存在未公开的开发分支或后续工作。

## cuTile 生产库集成进展：FlashInfer

以下三项均已合并到 [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)。信息来自对应 PR 的实时状态、说明和测试记录。

| 集成方向 | PR | 状态与合并日期（UTC） | 已合并的功能 | 使用范围与后续关注点 |
|---|---|---|---|---|
| **Unified fused MoE** | [#4646：feat(moe): add cuTile fused MoE backend for BF16 and NVFP4 Unified MoE](https://github.com/flashinfer-ai/flashinfer/pull/4646) | **Merged，2026-09-01** | 在 Unified MoE API 中增加 cuTile 后端；支持 BF16、NVFP4 W4A4，包含权重准备、自动调优、正确性测试和 CUTLASS 性能对比。 | 该 PR 的 BF16 支持 SM89、SM90、SM120、SM121；NVFP4 支持 SM120、SM121。支持 SwiGLU 和 ReLU2。此处记录的是该 PR 的合并范围，不能推定所有架构和量化格式都已覆盖。 |
| **Attention：prefill 与 decode** | [#4018：feat(attention): cuTile paged/ragged prefill + paged/MLA decode](https://github.com/flashinfer-ai/flashinfer/pull/4018) | **Merged，2026-09-04** | 在现有 FlashInfer wrapper 中接入 paged/ragged prefill、paged GQA decode 和 DeepSeek MLA decode；包含正确性验证、性能比较及 decode CUDA Graph 测试。 | 通过 **`backend="cutile"` 显式启用**，默认后端保持原样。性能依赖形状、page size、GQA 分组与架构；PR 中部分 MLA 大 batch 场景有优势，部分小 page 和 ragged prefill 场景仍落后。 |
| **CUDA 13 CI 编译环境** | [#4939：ci: provision cuTile compiler in CUDA 13 images](https://github.com/flashinfer-ai/flashinfer/pull/4939) | **Merged，2026-09-04** | 为 CUDA 13 CI 镜像安装 cuTile 编译工具链，增加 API、包元数据、编译器发现与可执行性 smoke 检查，以及 CUDA runtime 主版本一致性检查。 | 这是分阶段部署的 **Stage 1**，该 PR 本身不修改 GPU 测试矩阵。CUDA 12 镜像仍按可用性跳过 cuTile；不能将 CUDA 12 runtime 与 CUDA 13 编译器混用解读为官方 CUDA 12 支持。 |

这些 PR 表明 cuTile 已进入实际推理库的 API、测试和构建流程。**后端已合并、发行版已包含、线上默认启用和大规模生产部署是不同状态**；本表核实的是 PR 合并状态和功能范围，没有据此推定下游发行版覆盖或线上部署规模。

从已合并的工作看，当前评估重点包括：

- **算子覆盖**：核对具体后端的 GPU、dtype、shape 和 activation 支持，不能将 cuTile 语言的硬件支持范围直接等同于每个 FlashInfer 算子的支持范围。
- **性能**：按目标工作负载比较，Attention PR 已公开有优势和仍落后的场景，不能假设 cuTile 对所有形状都更快。
- **部署环境**：区分 `cuda-tile` Python 包与必需的编译工具链。CI PR 特意分开管理编译器依赖，避免影响 PyTorch 所选的 CUDA runtime。

## 演讲材料中的状态说明

本地材料：[GPUMODE_114_PyCuTe (2).pdf](</Users/huangzhilin/Downloads/GPUMODE_114_PyCuTe (2).pdf>)。

- **第 3 页**：CuTeDSL、CuTeIR、cuda.lang 集成列在 **Ongoing work** 下。
- **第 12 页**：说明 cuda.lang 是 AST 编译器，PyCuTe 互操作需要适配，并需要声明 trace-time 代码与被编译代码的边界。
- **第 98 页**：`Cuda.lang metaprogramming` 仍列在 **Coming Soon / IOUs** 下。

这里的状态均以 2026-09-09 的核查结果为准；后续阅读时应重新确认 PR 状态与主分支代码。
