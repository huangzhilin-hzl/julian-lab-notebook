# PyCuTe 集成：仓库、PR 与功能提交

核查日期：2026-09-09。

来源是 Cris Cecka 在 GPU MODE #114（2026-09-04）的演讲材料。需要区分 **CuTe IR / cuda.lang 本身的实现**与 **PyCuTe 对这些项目的专门适配**；下面列出的相关 PR 不能直接作为 PyCuTe 集成已经完成的证据。

## 仓库与 PR 对应关系

| PPT 条目 | 对应仓库及代码目录 | 已查到的相关 PR | 核查时的状态与范围 |
|---|---|---|---|
| **CuTeIR** | [NVIDIA/cutlass — cutlass_compiler/cute_ir](https://github.com/NVIDIA/cutlass/tree/main/cutlass_compiler/cute_ir) | [#3426：Cutlass compiler: Introducing CuTe IR MLIR dialect](https://github.com/NVIDIA/cutlass/pull/3426) | **已合并，2026-08-03**。引入 CuTe IR MLIR dialect、Cutegen 类型推导与 lowering 后端，以及 Base lowering 基础设施。这是 CuTe IR 的引入 PR，并非专门的 PyCuTe 接入 PR。 |
| **cuda.lang（enable meta-programming）** | [NVIDIA/cutile-python — experimental/cuda-lang](https://github.com/NVIDIA/cutile-python/tree/main/experimental/cuda-lang) | [#100：[lang][nfc] Add execution and metaprogramming documentation](https://github.com/NVIDIA/cutile-python/pull/100) | **Open**。补充执行模型和元编程文档，介绍 `static_eval`、`static_iter`、`static_assert`、`ensure_constant`、`static_def` 等能力；属于文档改动，不实现 PyCuTe 集成。 |

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

## 演讲材料中的状态说明

本地材料：[GPUMODE_114_PyCuTe (2).pdf](</Users/huangzhilin/Downloads/GPUMODE_114_PyCuTe (2).pdf>)。

- **第 3 页**：CuTeDSL、CuTeIR、cuda.lang 集成列在 **Ongoing work** 下。
- **第 12 页**：说明 cuda.lang 是 AST 编译器，PyCuTe 互操作需要适配，并需要声明 trace-time 代码与被编译代码的边界。
- **第 98 页**：`Cuda.lang metaprogramming` 仍列在 **Coming Soon / IOUs** 下。

这里的状态均以 2026-09-09 的核查结果为准；后续阅读时应重新确认 PR 状态与主分支代码。
