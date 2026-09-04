# Looped Transformer 论文阅读清单

> 更新日期：2026-09-04
>
> 主题：Looped Transformer、Recursive/ recurrent-depth Transformer、动态计算深度、latent reasoning 与推理系统实现。

## 结论先行

Looped Transformer 的核心不是“把输出反复喂给模型”，而是沿网络深度重复使用同一组 Transformer 参数：

\[
h^{(t+1)} = F_{\theta}\left(h^{(t)}; e_t\right), \qquad t=0,1,\ldots,T-1.
\]

普通 Transformer 通常在每一层使用不同参数 \(F_{\theta_l}\)；looped model 则让一个层或一个 layer stack 多次处理 hidden states。这样可以把三个原本绑定的量拆开：

- 参数量：由物理存储的唯一权重决定。
- 有效深度：由循环次数决定。
- 推理计算量：可以在 inference time 通过增加循环次数扩展。

但它首先是参数共享和计算深度设计，不天然等于 latent reasoning，也不天然节省 FLOPs。若一个 block 循环 \(T\) 次，通常仍需大约 \(T\) 倍 block compute，并引入串行延迟和额外 KV-cache 问题。

## Astra 相关事实边界

截至 2026-09-04，OpenAI 的 [GPT-6 Astra 发布页](https://openai.com/index/gpt-6-astra/)和 [Astra System Card](https://deploymentsafety.openai.com/gpt-6-astra)均未披露模型架构，也没有确认 recurrent depth 或 looped Transformer。

因此应区分：

- 已确认：Astra 是 reasoning model，官方公开了能力、训练与安全评测信息。
- 尚未确认：Astra 是否使用 looped Transformer、循环哪部分层、循环次数和停止策略。
- 可以成立：Astra 的发布让 recurrent-depth、latent reasoning 和内部 test-time compute 再次成为热门讨论方向。

Sebastian Raschka 的 [OpenAI Astra and Looped Transformers](https://www.sebastianraschka.com/blog/2026/openai-astra-looped-transformers.html)适合作为五分钟热身，但它是评论文章，不是 Astra 架构的一手材料。

## 最短阅读路线

如果只读三篇：

1. [Universal Transformers](https://arxiv.org/abs/1807.03819)
2. [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171)
3. [Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741)

如果希望建立完整判断框架，按下面六篇顺序阅读：

| 顺序 | 论文 | 核心问题 | 阅读重点 |
| --- | --- | --- | --- |
| 1 | [Universal Transformers](https://arxiv.org/abs/1807.03819)（2018） | Transformer 能否沿深度递归，并为不同位置分配不同计算量？ | Depth recurrence、timestep encoding、Adaptive Computation Time（ACT）。 |
| 2 | [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171)（2025） | 不生成额外 CoT token，能否靠增加 recurrent depth 扩展 test-time compute？ | 随机训练循环深度、测试时增加循环、per-token adaptive compute、KV sharing 和 self-speculative decoding。 |
| 3 | [Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation](https://arxiv.org/abs/2507.10524)（2025） | 如何让困难 token 多算、简单 token 提前退出？ | Token-level router、active-token attention、选择性 KV cache 和 KV sharing。 |
| 4 | [Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741)（2025） | LoopLM 能否扩展到多万亿 token 预训练？ | Ouro 模型、7.7T-token pretraining、entropy-regularized depth allocation、learned exit，以及知识存储与知识操作能力的区分。 |
| 5 | [Parallel Loop Transformer for Efficient Test-Time Computation Scaling](https://arxiv.org/abs/2510.24824)（2025） | Loops 的串行延迟和 KV-cache 膨胀能否被系统化解决？ | Cross-Loop Parallelism、跨 token 流水、首轮 KV sharing、gated sliding-window attention。 |
| 6 | [SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers](https://arxiv.org/abs/2609.01343)（2026） | 固定参数、FLOPs 和 KV cache 后，looping 是否仍有优势？ | Middle layers loop twice、MoE、compute-matched scaling law；重点审查其实验预算匹配方法。 |

其中第 2、4 篇最接近当前“内部 latent computation 作为第三条 scaling axis”的讨论；第 5、6 篇分别补齐系统代价与公平比较问题。

## 基础与思想来源

- [ ] [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983)（2016）
  - 提出可微的动态计算步数机制。
  - 重点理解 halting probability、ponder cost，以及为什么不同输入应获得不同计算预算。

- [ ] [Universal Transformers](https://arxiv.org/abs/1807.03819)（2018）
  - 把自注意力、深度递归和 per-position ACT 组合起来。
  - 它是现代 looped Transformer 最直接的概念祖先。

- [ ] [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377)（2019）
  - 将无限深、权重共享网络写成固定点求解问题，并用隐式微分训练。
  - 适合用来理解“有限次数展开”与“直接求平衡点”两条路线的差异。

- [ ] [PonderNet: Learning to Ponder](https://arxiv.org/abs/2107.05407)（2021）
  - 用概率化停止过程重新表述 adaptive computation。
  - 对阅读 MoR、Ouro 和后续 learned halting 很有帮助。

## 算法能力与理论分支

- [ ] [Looped Transformers as Programmable Computers](https://arxiv.org/abs/2301.13196)（2023）
  - 构造性展示固定规模、循环执行的 Transformer 如何模拟指令、内存、条件跳转和迭代算法。
  - 回答“为什么循环结构原则上能够承载算法”这一表达能力问题，但不代表梯度训练一定能学到这些构造。

- [ ] [Looped Transformers are Better at Learning Learning Algorithms](https://arxiv.org/abs/2311.12424)（2023）
  - 研究 in-context learning 中的迭代优化归纳偏置。
  - 关注 looped model 如何以更少参数逼近多步优化算法，以及比较是否匹配了计算量。

- [ ] [Looped Transformers for Length Generalization](https://openreview.net/pdf?id=PEdOdntGJG)（NeurIPS 2024）
  - 将循环次数随问题长度调整，研究算术和算法任务的长度外推。
  - 值得重点检查训练长度、测试长度与 inference loop count 之间是否存在人为耦合。

- [ ] [Can Looped Transformers Learn to Implement Multi-step Gradient Descent for In-context Learning?](https://arxiv.org/abs/2410.08292)（2024）
  - 理论分析线性 looped Transformer 是否真的能通过训练学到多步预条件梯度下降。
  - 用来区分 expressivity 结果和 learnability 结果。

## 现代 LLM 与参数共享分支

- [ ] [Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA](https://arxiv.org/abs/2410.20672)（2024）
  - 把普通预训练 Transformer 转成重复共享 layer stack 的 recursive model。
  - 通过 depth-wise LoRA 允许不同递归深度做少量差异化计算，在严格共享和完全独立参数之间折中。
  - 还提出 Continuous Depth-wise Batching 与 early exit 的吞吐思路。

- [ ] [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171)（NeurIPS 2025）
  - 在 3.5B 参数、800B training tokens 的尺度验证 recurrent-depth 预训练。
  - 核心问题是：训练时见过一组循环深度后，推理时继续增加深度能否稳定带来能力增益。
  - 阅读图表时应同时看任务收益、饱和点、额外 FLOPs 和 wall-clock latency，而不只看“materialized parameter equivalent”。

- [ ] [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524)（NeurIPS 2025）
  - 将参数共享和 token-level adaptive computation 合并。
  - 它比固定循环次数更接近理想目标：按 token 难度分配深度，而不是整个序列一起多跑一遍。

- [ ] [Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741)（2025）
  - 发布 Ouro 1.4B/2.6B 等开放模型，把循环结构直接放入大规模预训练。
  - 重点关注多步 loss、退出分布正则、不同循环数的 scaling curve，以及作者如何证明收益来自 knowledge manipulation 而非知识容量。

- [ ] [Nanbeige4.2-3B: Unlocking Agentic Capabilities in a Compact Model](https://arxiv.org/abs/2607.22083)（2026）
  - 一个可实际下载研究的开放案例：同一个 layer stack 额外执行一轮，并从头在 28T tokens 上预训练。
  - 该模型同时使用大规模数据、SFT 和多阶段 RL，不能把最终 agent benchmark 增益单独归因于 looping。

## Latent reasoning 分支

- [ ] [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769)（Coconut，2024）
  - 将上一时刻的 hidden state 直接作为后续输入 embedding，避免每个中间思维都投影成离散文字。
  - 它属于 continuous-thought recurrence，不等同于沿网络深度重复 layer stack，适合作为对照组。

- [ ] [Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers](https://arxiv.org/abs/2606.31779)（LOTUS，2026）
  - 用 padded latent slots、looped backbone 和并行 step-aligned supervision 逼近显式 CoT。
  - 作者报告在 3B 规模接近显式 CoT，并将 thought-phase latency 降低 2.5–6.9 倍；需要关注任务范围和 OOD 结果。

- [ ] [Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers](https://arxiv.org/abs/2606.18206)（2026）
  - 研究深度循环的信号传播稳定性，并用 fixed-point convergence 作为停止条件。
  - 连接了 Universal Transformer、DEQ 与现代 reasoning model 三条线。

- [ ] [Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping](https://arxiv.org/abs/2606.29983)（2026）
  - 指出 loop-count extrapolation 可能非常脆弱，训练时随机化循环数可降低 OOD 方差。
  - 重要结论是“何时停止”不仅是推理策略，也是训练分布设计问题。

## 推理系统与 KV-cache 分支

- [ ] [Parallel Loop Transformer for Efficient Test-Time Computation Scaling](https://arxiv.org/abs/2510.24824)（2025）
  - 将当前 token 的第 1 轮、前一个 token 的第 2 轮等组成 displaced micro-batch，以跨 token 流水隐藏 loop 串行延迟。
  - 同时研究全局 KV sharing 与局部 sliding-window KV 的组合。

- [ ] [Looped Latent Attention: Cross-Loop KV Compression for Looped Transformers](https://arxiv.org/abs/2607.15456)（2026）
  - 将 loop-indexed K/V 视为沿 recurrence axis 的低秩轨迹，再通过 post-training codec 压缩。
  - 适合重点复核压缩率、重建开销、实际 batch capacity 和端到端吞吐，而不只看 cache bytes。

- [ ] [SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers](https://arxiv.org/abs/2609.01343)（2026）
  - 重复中间一半层两次，并同时匹配每 token FLOPs、非 embedding 参数量和 KV cache。
  - 作者报告 compute-optimal frontier 上可节省 6.8%–18.0% training FLOPs，但该工作刚发布，应等待独立复现。

## 统一阅读框架

阅读每篇论文时，建议记录下面七项，避免把不同意义的“效率”混在一起：

| 维度 | 要记录的问题 |
| --- | --- |
| 共享粒度 | 重复单层、若干层、整个 stack，还是只共享部分模块？ |
| 深度信号 | 每轮是否有 timestep/depth embedding、独立 norm、LoRA 或 gate？ |
| 训练循环分布 | 固定循环、随机循环，还是学习退出分布？训练最多见过多少轮？ |
| 推理停止策略 | 固定 \(T\)、per-sequence exit、per-token routing，还是固定点收敛？ |
| 预算口径 | 固定参数、训练 FLOPs、推理 FLOPs、延迟、显存或 KV cache 中的哪一个？ |
| 跨深度泛化 | 测试循环次数超过训练范围后，性能单调提升、饱和还是退化？ |
| 推理证据 | 论文是否证明内部状态执行了可解释的迭代算法，还是只观察到准确率提升？ |

## 当前值得追踪的问题

1. **Compute-matched scaling**：固定总 FLOPs 后，looping 是否仍优于增加普通层、增加宽度或 MoE experts？
2. **Loop extrapolation**：训练 2–4 轮、推理 16–64 轮的增益何时可靠，何时只是分布外碰运气？
3. **Adaptive depth 的硬件利用率**：per-token routing 能节省理论计算，但是否造成 token compaction、负载不均和 kernel launch 开销？
4. **KV cache 语义**：不同 loop 的 K/V 是否可以共享或压缩，而不破坏每轮逐步修正表示的能力？
5. **Latent reasoning 的可验证性**：性能提升是否真来自多步推理，以及如何监控、解释或监督隐藏状态中的计算？
6. **预训练还是 upcycling**：从头训练 looped model，与把现有 dense model 转成 recursive model，哪种方案的成本收益更合理？

## 建议阅读顺序

### 偏模型架构

Universal Transformer → Relaxed Recursive Transformer → Recurrent Depth → Mixture-of-Recursions → Ouro → SMELT。

### 偏推理与 latent CoT

ACT/PonderNet → Recurrent Depth → Coconut → Ouro → LOTUS → Fixed-Point Reasoners。

### 偏推理系统

Recurrent Depth 的 KV-cache 部分 → Mixture-of-Recursions → Parallel Loop Transformer → Looped Latent Attention → SMELT。

### 偏理论与算法泛化

Universal Transformer → Programmable Computers → Learning Learning Algorithms → Length Generalization → Multi-step Gradient Descent → Stochastic Stopping。
