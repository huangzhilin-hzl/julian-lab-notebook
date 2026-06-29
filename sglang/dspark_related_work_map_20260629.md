# DSpark Related Work Reference Map

记录时间：2026-06-29

来源：本地 `DSpark_paper.pdf`，远程链接：[DSpark_paper.pdf](https://github.com/deepseek-ai/DeepSpec/blob/main/DSpark_paper.pdf)。范围为 Section 6 Related Work 和 References。

## 一句话结论

DSpark 的 Related Work 可以分成三条线：

| 主线 | DSpark 借鉴 / 对比点 |
| --- | --- |
| Speculative decoding algorithms | 从小模型 drafter、多头/MTP、self-speculation，发展到 DFlash/DART/PARD 这类 block-parallel drafter。DSpark 属于 parallel drafter 的增强版。 |
| System-aware scheduling | 研究重点从“能 draft 几个 token”转向“当前负载下应该生成/验证几个 token”。DSpark 的 hardware-aware prefix scheduler 属于这一线。 |
| Parallel generation / NAT | 并行生成容易出现 multi-modal mixing / cross-mode collision。DSpark 的 sequential head 是用轻量局部修正解决这个问题，同时保留 per-token softmax 概率。 |

## Speculative Decoding Algorithms

| 论文 / 项目 | 作者团队 | 方向 | 链接 |
| --- | --- | --- | --- |
| Lossless Acceleration for Seq2Seq Generation with Aggressive Decoding | Ge, Xia, Sun, Chen, Wei | 早期 blockwise / aggressive decoding | [arXiv](https://arxiv.org/abs/2205.10350) |
| Blockwise Parallel Decoding for Deep Autoregressive Models | Stern, Shazeer, Uszkoreit | blockwise parallel decoding | [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2018/file/c4127b9194fe8562c64dc0f5bf2c93bc-Paper.pdf) |
| Instantaneous Grammatical Error Correction with Shallow Aggressive Decoding | Sun, Ge, Wei, Wang | aggressive decoding 用于 GEC | [ACL](https://aclanthology.org/2021.acl-long.462/) |
| Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2Seq Generation | Xia, Ge, Wang, Chen, Wei, Sui | seq2seq speculative decoding | [ACL](https://aclanthology.org/2023.findings-emnlp.257/) |
| Accelerating Large Language Model Decoding with Speculative Sampling | Chen, Borgeaud, Irving, Lespiau, Sifre, Jumper | rejection-sampling speculative decoding | [arXiv](https://arxiv.org/abs/2302.01318) |
| Fast Inference from Transformers via Speculative Decoding | Leviathan, Kalman, Matias | 标准 draft-then-verify 框架 | [PMLR](https://proceedings.mlr.press/v202/leviathan23a.html) |
| Hydra: Sequentially-dependent Draft Heads for Medusa Decoding | Ankner, Parthasarathy, Nrusimha, Rinard, Ragan-Kelley, Brandon | 多 draft heads 加 head 间依赖 | [OpenReview](https://openreview.net/forum?id=FbhjirzvJG) |
| Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads | Cai, Li, Geng, Peng, Lee, Chen, Dao | target 上接多个 decoding heads | [PMLR](https://proceedings.mlr.press/v235/cai24b.html) |
| FastMTP: Accelerating LLM Inference with Enhanced Multi-token Prediction | Cai et al. | 增强 multi-token prediction | [arXiv](https://arxiv.org/abs/2509.18362) |
| DeepSeek-V3 Technical Report | DeepSeek-AI | MTP / 大模型推理技术报告 | [arXiv](https://arxiv.org/abs/2412.19437) |
| Better & Faster Large Language Models via Multi-token Prediction | Gloeckle, Idrissi, Roziere, Lopez-Paz, Synnaeve | 训练期 MTP 提升推理 | [PMLR](https://proceedings.mlr.press/v235/gloeckle24a.html) |
| EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty | Li, Wei, Zhang, Zhang | feature-level autoregressive drafter | [PMLR](https://proceedings.mlr.press/v235/li24bt.html) |
| EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees | Li, Wei, Zhang, Zhang | 动态 draft tree | [ACL](https://aclanthology.org/2024.emnlp-main.422/) |
| EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test | Li, Wei, Zhang, Zhang | EAGLE 系列扩展 | [OpenReview](https://openreview.net/forum?id=4exx1hUffq) |
| Learning Harmonized Representations for Speculative Sampling | Zhang, Wang, Huang, Xu | draft-target 表示对齐 | [arXiv](https://arxiv.org/abs/2408.15766) |
| LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding | Elhoushi et al. | early-exit self-speculation | [ACL](https://aclanthology.org/2024.acl-long.681/) |
| Kangaroo: Lossless Self-Speculative Decoding for Accelerating LLMs via Double Early Exiting | Liu, Tang, Liu, Ni, Tang, Han, Wang | 双 early-exit self-speculation | [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/16336d94a5ffca8de019087ab7fe403f-Paper-Conference.pdf) |
| SWIFT: On-the-fly Self-Speculative Decoding for LLM Inference Acceleration | Xia, Yang, Dong, Wang, Li, Ge, Liu, Li, Sui | 在线 self-speculative decoding | [OpenReview](https://openreview.net/forum?id=EKJhH5D5wA) |
| Draft & Verify: Lossless LLM Acceleration via Self-Speculative Decoding | Zhang et al. | self-speculative decoding | [ACL](https://aclanthology.org/2024.acl-long.607/) |
| Speculative Decoding with a Speculative Vocabulary | Williams, Kwon, Li, Kouris, Venieris | speculative vocabulary / 词表压缩 | [arXiv](https://arxiv.org/abs/2602.13836) |
| FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling | Zhao et al. | 频率排序降低大词表 speculative 开销 | [ACL](https://aclanthology.org/2025.acl-long.198/) |
| Prompt Lookup Decoding | Saxena | prompt n-gram lookup drafter | [GitHub](https://github.com/apoorvumang/prompt-lookup-decoding/) |
| PLD+: Accelerating LLM Inference by Leveraging Language Model Artifacts | Somasundaram, Phukan, Saxena | prompt lookup 扩展 | [ACL](https://aclanthology.org/2025.findings-naacl.338/) |
| SAM Decoding: Speculative Decoding via Suffix Automaton | Hu et al. | suffix automaton drafter | [ACL](https://aclanthology.org/2025.acl-long.595/) |
| REST: Retrieval-Based Speculative Decoding | He, Zhong, Cai, Lee, He | 检索式 speculative drafter | [arXiv](https://arxiv.org/abs/2311.08252) |
| Draft Less, Retrieve More: Hybrid Tree Construction for Speculative Decoding | Shen et al. | retrieval + hybrid draft tree | [arXiv](https://arxiv.org/abs/2605.20104) |
| P-EAGLE: Parallel-Drafting EAGLE with Scalable Training | Hui et al. | 并行化 EAGLE | [arXiv](https://arxiv.org/abs/2602.01469) |
| PARD: Accelerating LLM Inference with Low-cost PARallel Draft Model Adaptation | An, Bai, Liu, Li, Barsoum | low-cost parallel drafter adaptation | [OpenReview](https://openreview.net/forum?id=XbOyv7iVGL) |
| DFlash: Block Diffusion for Flash Speculative Decoding | Chen, Liang, Liu | block diffusion parallel drafter | [arXiv](https://arxiv.org/abs/2602.06036) |
| DART: Diffusion-Inspired Speculative Decoding for Fast LLM Inference | Liu et al. | diffusion-inspired block drafter | [arXiv](https://arxiv.org/abs/2601.19278) |
| Accelerating Speculative Decoding with Block Diffusion Draft Trees | Ringel, Romano | diffusion draft tree | [arXiv](https://arxiv.org/abs/2604.12989) |
| Domino: Decoupling Causal Modeling from Autoregressive Drafting in Speculative Decoding | Huang, Zhang, Zhang, Lin, Xu, Zhang | DFlash 上加 CausalEncoder | [arXiv](https://arxiv.org/abs/2605.29707) |
| DFlare: Scaling up Draft Capacity for Block Diffusion Speculative Decoding | Zhang et al. | layer-wise fusion 改善 DFlash conditioning | [arXiv](https://arxiv.org/abs/2606.02091) |

## System-Aware Scheduling For Speculative Decoding

| 论文 / 项目 | 作者团队 | 方向 | 链接 |
| --- | --- | --- | --- |
| GliDe with a CaPE: A Low-Hassle Method to Accelerate Speculative Decoding | Du et al. | adaptive length / low-hassle speculative decoding | [PMLR](https://proceedings.mlr.press/v235/du24c.html) |
| TALON: Confidence-Aware Speculative Decoding with Adaptive Token Trees | Liu, Lv, Shen, Sun, Sun | confidence-aware adaptive token tree | [arXiv](https://arxiv.org/abs/2601.07353) |
| Dynamic Speculation Lookahead Accelerates Speculative Decoding of Large Language Models | Mamou et al. | dynamic speculation lookahead | [PMLR](https://proceedings.mlr.press/v262/mamou24a.html) |
| SpecBound: Adaptive Bounded Self-Speculation with Layer-Wise Confidence Calibration | Wen, Feng | layer-wise confidence calibration | [arXiv](https://arxiv.org/abs/2604.12247) |
| SpecDec++: Boosting Speculative Decoding via Adaptive Candidate Lengths | Huang, Guo, Wang | adaptive candidate length | [arXiv](https://arxiv.org/abs/2405.19715) |
| AutoMTP_vLLM | Zacks917 | vLLM MTP early-stop 工程实现 | [GitHub](https://github.com/Zacks917/AutoMTP_vLLM) |
| Not-a-Bandit: Provably No-Regret Drafter Selection in Speculative Decoding for LLMs | Liu, Huang, Jia, Park, Wang | 多 drafter 在线选择 | [arXiv](https://arxiv.org/abs/2510.20064) |
| D-Cut: Adaptive Verification Depth Pruning for Speculative Decoding | AngelSlim Team | adaptive verification depth pruning | [Docs](https://angelslim.readthedocs.io/zh-cn/latest/dcut.html) |
| ECHO: Elastic Speculative Decoding with Sparse Gating for High-Concurrency Scenarios | Hu et al. | 高并发下 sparse gating / budget scheduling | [arXiv](https://arxiv.org/abs/2604.09603) |
| AdaSpec: Adaptive Speculative Decoding for Fast, SLO-Aware LLM Serving | Huang, Wu, Shi, Zou, Yu, Shi | SLO-aware serving 调度 | [ACM DOI](https://doi.org/10.1145/3772052.3772239) |
| Nightjar: Dynamic Adaptive Speculative Decoding for Large Language Models Serving | Li et al. | 按负载动态调整 speculation | [arXiv](https://arxiv.org/abs/2512.22420) |
| Optimizing Speculative Decoding for Serving LLMs Using Goodput | Liu, Daniel, Hu, Kwon, Li, Mo, Cheung, Deng, Stoica, Zhang | goodput-oriented serving 分析；后续 arXiv 版本演进为 TurboSpec 口径 | [arXiv](https://arxiv.org/abs/2406.14066) |
| TurboSpec: Closed-loop Speculation Control System for Optimizing LLM Serving Goodput | Liu et al. | closed-loop speculation control；与上一项属于同一 arXiv 演进线 | [arXiv](https://arxiv.org/abs/2406.14066) |
| SpecInfer: Accelerating LLM Serving with Tree-Based Speculative Inference and Verification | Miao et al. | tree-based speculative serving system | [ACM](https://dl.acm.org/doi/10.1145/3620666.3651335) |
| MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding | Sadhukhan et al. | long-context speculative serving | [OpenReview](https://openreview.net/forum?id=CS2JWaziYr) |
| TETRIS: Optimal Draft Token Selection for Batch Speculative Decoding | Wu, Zhou, Verma, Prakash, Rus, Low | batch 内 token selection / scheduling | [ACL](https://aclanthology.org/2025.acl-long.1598/) |

## Parallel Generation / NAT Background

| 论文 | 作者团队 | 方向 | 链接 |
| --- | --- | --- | --- |
| Non-Autoregressive Neural Machine Translation | Gu, Bradbury, Xiong, Li, Socher | NAT 基础工作 | [OpenReview](https://openreview.net/forum?id=B1l8BtlCb) |
| Fast Decoding in Sequence Models Using Discrete Latent Variables | Kaiser et al. | latent variable parallel decoding | [PMLR](https://proceedings.mlr.press/v80/kaiser18a.html) |
| FlowSeq: Non-Autoregressive Conditional Sequence Generation with Generative Flow | Ma, Zhou, Li, Neubig, Hovy | flow-based NAT | [ACL](https://aclanthology.org/D19-1437/) |
| Order-Agnostic Cross Entropy for Non-Autoregressive Machine Translation | Du, Tu, Jiang | 放松训练目标，缓解多模态平均 | [PMLR](https://proceedings.mlr.press/v139/du21c.html) |
| Glancing Transformer for Non-Autoregressive Neural Machine Translation | Qian et al. | glancing training | [ACL](https://aclanthology.org/2021.acl-long.155/) |
| Sequence-Level Training for Non-Autoregressive Neural Machine Translation | Shao, Feng, Zhang, Meng, Zhou | sequence-level NAT training | [ACL](https://aclanthology.org/2021.cl-4.29/) |
| Beyond MLE: Convex Learning for Text Generation | Shao, Ma, Zhang, Feng | 放松 MLE 目标，提升生成一致性 | [OpenReview](https://openreview.net/forum?id=sla7V80uWA) |
| Structured Denoising Diffusion Models in Discrete State-Spaces | Austin, Johnson, Ho, Tarlow, van den Berg | discrete diffusion / iterative re-prediction | [OpenReview](https://openreview.net/forum?id=h7-XixPCAL) |
| Mask-Predict: Parallel Decoding of Conditional Masked Language Models | Ghazvininejad, Levy, Liu, Zettlemoyer | iterative mask-predict | [ACL](https://aclanthology.org/D19-1633/) |
| Diffusion-LM Improves Controllable Text Generation | Li, Thickstun, Gulrajani, Liang, Hashimoto | diffusion language model | [OpenReview](https://openreview.net/forum?id=3s9IrEsjLyk) |
| Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models | Arriola et al. | block-level autoregression / diffusion | [OpenReview](https://openreview.net/forum?id=tyEyYT267x) |
| Semi-Autoregressive Neural Machine Translation | Wang, Zhang, Chen | block-level semi-autoregressive NMT | [ACL](https://aclanthology.org/D18-1044/) |
| Fast Structured Decoding for Sequence Models | Sun et al. | CRF-NAT / structured output layer | [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2019/file/74563ba21a90da13dacf2a73e3ddefa7-Paper.pdf) |
| End-to-End Non-Autoregressive Neural Machine Translation with Connectionist Temporal Classification | Libovicky, Helcl | CTC output layer | [ACL](https://aclanthology.org/D18-1336/) |
| Non-Autoregressive Machine Translation with Latent Alignments | Saharia, Chan, Saxena, Norouzi | latent alignment NAT | [ACL](https://aclanthology.org/2020.emnlp-main.83/) |
| Directed Acyclic Transformer for Non-Autoregressive Machine Translation | Huang, Zhou, Liu, Li, Huang | DAT / HMM-like path modeling | [PMLR](https://proceedings.mlr.press/v162/huang22m.html) |
| Non-Autoregressive Machine Translation with Probabilistic Context-Free Grammar | Gui, Shao, Ma, Zhang, Chen, Feng | PCFG structured output | [OpenReview](https://openreview.net/forum?id=LloZFVwWvj) |
| Speculative Decoding with CTC-Based Draft Model for LLM Inference Acceleration | Wen, Gui, Feng | CTC drafter for speculative decoding | [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/a79054a9da91d73ed3cb1a9e87d7cd2d-Paper-Conference.pdf) |

## DSpark 在这张图谱里的位置

| 对比维度 | DSpark 的位置 |
| --- | --- |
| Drafter architecture | 继承 DFlash 这类 block-parallel drafter，但用 Markov / RNN sequential head 给 block 内 token 注入 causal dependency。 |
| Verification policy | 不固定验证整段 draft block，而是用 calibrated confidence 和 SPS(B) 曲线选择 per-request prefix length。 |
| 与 Domino 的关系 | Domino 主要改 DFlash 的 causal modeling；DSpark 同时做 causal modeling 和 production-oriented verification scheduling。 |
| 与 NAT 的关系 | DSpark 借鉴 NAT 中“并行生成会混合多种 plausible modes”的问题意识，但不用全局归一化结构，保留逐 token softmax，便于标准 speculative rejection sampling。 |
| 与 serving 调度的关系 | DSpark 把 draft token selection 写成期望吞吐 `Theta = tau * SPS(B)` 最大化问题，和 TETRIS / AdaSpec / TurboSpec / MagicDec 等系统线相近。 |

## 推荐阅读顺序

| 顺序 | 资料 | 目标 |
| ---: | --- | --- |
| 1 | Leviathan et al.; Chen et al. | 先理解 lossless speculative decoding 为什么保持 target distribution。 |
| 2 | Medusa / MTP / EAGLE 系列 | 看 drafter 如何从小模型发展到多头和 feature extrapolator。 |
| 3 | DFlash / DART / PARD / Domino / DFlare | 看 block-parallel drafter 如何降低 draft-side sequential cost。 |
| 4 | SpecDec++ / TALON / TETRIS / AdaSpec / TurboSpec | 看 verification length / token selection 如何变成调度问题。 |
| 5 | NAT / semi-autoregressive / CRF / CTC / DAT / PCFG | 回看 DSpark sequential head 为什么选择局部 logit bias，而不是全局 structured output。 |
