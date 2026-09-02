'use client';

import { useMemo, useState } from 'react';

const stages = [
  { name: 'Schedule', base: 2.3, type: 'cpu', note: 'metadata + KV alloc' },
  { name: 'QKV GEMM', base: 9.2, type: 'compute', note: 'projection' },
  { name: 'Attention', base: 14.8, type: 'memory', note: 'read KV cache' },
  { name: 'O Proj', base: 7.4, type: 'compute', note: 'projection' },
  { name: 'MLP / MoE', base: 20.5, type: 'compute', note: 'expert GEMM' },
  { name: 'Collective', base: 4.7, type: 'comm', note: 'all-reduce' },
  { name: 'Sampling', base: 2.2, type: 'cpu', note: 'top-k + stream' },
];

const metrics = [
  ['TTFT', 'Time To First Token', '请求进入后，到首个 token 返回的时间', 'Prefill + 排队'],
  ['TPOT', 'Time Per Output Token', '稳定生成阶段，每个输出 token 的平均间隔', 'Decode 关键路径'],
  ['TPS', 'Tokens Per Second', '系统或单用户单位时间生成的 token 数', '吞吐 / 交互性'],
  ['ITL', 'Inter-Token Latency', '相邻 token 的实际到达间隔分布', '抖动 / 尾延迟'],
];

const conceptSteps = [
  { title: 'Plan', lane: 'CPU', color: 'blue', detail: '选择本轮请求，分配 KV block，并准备 device metadata。', code: 'batch = scheduler.select()\nkv = block_table.allocate(batch)' },
  { title: 'Launch', lane: 'CPU → GPU', color: 'green', detail: '提交 CUDA Graph 或 kernel 序列；固定开销在小 batch 下尤其显眼。', code: 'graph.replay(input_ids, positions)\n# avoid host-device sync here' },
  { title: 'Forward', lane: 'GPU', color: 'purple', detail: 'Attention 读取历史 KV，MLP / MoE 完成主要计算。', code: 'x = attention(q, kv_cache)\nx = moe(x, router_logits)' },
  { title: 'Collect', lane: 'GPU ↔ GPU', color: 'orange', detail: 'TP / EP 引入 collective；通信能否与计算重叠决定扩展效率。', code: 'x = all_reduce(x, group=tp)\n# overlap on comm stream' },
  { title: 'Emit', lane: 'GPU → CPU', color: 'green', detail: '采样得到 token，更新状态并流式返回；随后立即进入下一轮。', code: 'token = sample(logits)\nstream.write(token)' },
];

export default function DecodePathArticle() {
  const [batch, setBatch] = useState(4);
  const [optimized, setOptimized] = useState(false);
  const [focusStep, setFocusStep] = useState(0);

  const profile = useMemo(() => {
    const workloadScale = 0.88 + Math.sqrt(batch) * 0.2;
    const items = stages.map((stage) => {
      const optimization = optimized
        ? stage.type === 'cpu' ? 0.45 : stage.type === 'comm' ? 0.82 : 0.9
        : 1;
      return { ...stage, time: stage.base * workloadScale * optimization };
    });
    const total = items.reduce((sum, stage) => sum + stage.time, 0);
    return { items, total, throughput: Math.round((batch * 1000) / total) };
  }, [batch, optimized]);

  return (
    <main id="top">
      <header className="site-header article-site-header">
        <a className="wordmark" href="../../">
          <span className="wordmark-glyph">I<span>/</span>P</span>
          <span>Inference Playbook</span>
        </a>
        <a className="repo-link" href="../../">返回总入口 <span>↖</span></a>
      </header>

      <div className="reading-layout">
        <aside className="toc" aria-label="本页目录">
          <p>ARTICLE CONTENTS</p>
          <ol>
            <li className="active"><a href="#top"><span>00</span>结论先行</a></li>
            <li><a href="#explorer"><span>01</span>交互概念拆解</a></li>
            <li><a href="#metrics"><span>02</span>四个核心指标</a></li>
            <li><a href="#path"><span>03</span>Decode 关键路径</a></li>
            <li><a href="#lab"><span>04</span>交互时间线</a></li>
            <li><a href="#method"><span>05</span>优化与证伪</a></li>
          </ol>
          <div className="toc-note"><span className="live-dot" /> 可交互文章<small>FOUNDATIONS / 01</small></div>
        </aside>

        <article className="article decode-article">
          <div className="breadcrumbs"><a href="../../">PLAYBOOK</a><b>/</b><span>SYSTEMS</span><b>/</b><em>DECODE PATH</em></div>
          <section className="article-hero">
            <div className="chapter-badge">FOUNDATIONS 01 · SYSTEMS</div>
            <h1>一次 Decode，<br />时间花在了<span>哪里？</span></h1>
            <p className="abstract">先学会画出一枚 token 的关键路径，再谈 kernel fusion、CUDA Graph 或 speculative decoding。端到端优化的第一步，是知道每一微秒属于谁。</p>
            <div className="article-meta">
              <div className="author-mark">JL</div>
              <p><strong>Julian Lab</strong><span>阅读约 12 分钟 · 基础</span></p>
              <div className="topic-tags"><span>Decode</span><span>Latency</span><span>Profiling</span></div>
            </div>
          </section>

          <section className="tldr-card">
            <div className="tldr-title"><span>TL;DR</span><small>先记住三句话</small></div>
            <ol>
              <li><b>TPOT 由整条循环决定。</b> kernel 很快，不代表 kernel 之间没有 gap。</li>
              <li><b>batch 改变瓶颈形态。</b> 小 batch 更怕 launch overhead，大 batch 更接近计算或带宽上限。</li>
              <li><b>优化必须能被 profile 证伪。</b> 用 trace 解释变化，再用端到端指标确认收益。</li>
            </ol>
          </section>

          <section className="concept-explorer" id="explorer">
            <header>
              <div><span className="live-dot" /> LIVE CONCEPT</div>
              <strong>Decode Loop Explorer</strong>
              <small>STEP {focusStep + 1} / {conceptSteps.length}</small>
            </header>
            <div className="explorer-body">
              <aside aria-label="概念步骤">
                {conceptSteps.map((step, index) => (
                  <button className={focusStep === index ? 'active' : ''} key={step.title} onClick={() => setFocusStep(index)}>
                    <span>{String(index + 1).padStart(2, '0')}</span><b>{step.title}</b><small>{step.lane}</small>
                  </button>
                ))}
              </aside>
              <div className="explorer-stage">
                <div className="explorer-canvas">
                  <div className="canvas-label">one decode iteration</div>
                  <div className="concept-flow">
                    {conceptSteps.map((step, index) => (
                      <button className={`${step.color} ${focusStep === index ? 'active' : ''}`} key={step.title} onClick={() => setFocusStep(index)}>
                        <i>{index + 1}</i><strong>{step.title}</strong><small>{step.lane}</small>
                      </button>
                    ))}
                    <div className="loop-back">next token ↩</div>
                  </div>
                </div>
                <div className="explorer-explanation">
                  <div>
                    <span>当前步骤</span>
                    <h3>{conceptSteps[focusStep].title}</h3>
                    <p>{conceptSteps[focusStep].detail}</p>
                  </div>
                  <pre><code>{conceptSteps[focusStep].code}</code></pre>
                </div>
                <footer>
                  <button disabled={focusStep === 0} onClick={() => setFocusStep((step) => Math.max(0, step - 1))}>← 上一步</button>
                  <span>点击流程块，逐步理解一轮生成</span>
                  <button disabled={focusStep === conceptSteps.length - 1} onClick={() => setFocusStep((step) => Math.min(conceptSteps.length - 1, step + 1))}>下一步 →</button>
                </footer>
              </div>
            </div>
          </section>

          <section className="chapter-section article-block" id="metrics">
            <div className="section-label">02 / MEASURE THE RIGHT THING</div>
            <h2>四个指标，不要混着说</h2>
            <p className="section-lede">吞吐变高时，单请求体验可能正在变差。先固定指标、统计口径与 workload，后面的优化讨论才有意义。</p>
            <div className="metric-table" role="table" aria-label="推理性能指标">
              {metrics.map(([short, name, meaning, dominatedBy]) => (
                <div className="metric-row" role="row" key={short}>
                  <strong>{short}</strong><span><b>{name}</b>{meaning}</span><em>{dominatedBy}</em>
                </div>
              ))}
            </div>
            <aside className="article-callout"><span>测量纪律</span>所有结果至少同时记录：模型、精度、GPU、并行策略、输入/输出长度、并发、缓存命中率与分位数。</aside>
          </section>

          <section className="chapter-section article-block" id="path">
            <div className="section-label">03 / CRITICAL PATH</div>
            <h2>把一轮 Decode 画成两条 lane</h2>
            <p className="section-lede">CPU 为下一步准备元数据并发起 kernel；GPU 执行模型、通信和采样相关算子。优化的核心，是缩短最长路径并让两条 lane 尽量重叠。</p>
            <figure className="sketch-card decode-sketch">
              <div className="sketch-title">One token, two asynchronous lanes</div>
              <div className="lane-diagram">
                <div className="lane-label blue">CPU / HOST</div>
                <div className="lane-track host-track"><i>schedule</i><i>metadata</i><b>sync?</b><i>launch</i></div>
                <div className="lane-label purple">GPU / DEVICE</div>
                <div className="lane-track gpu-track"><i>QKV</i><i>Attn</i><i>MLP</i><i>AR</i><i>Sample</i></div>
                <div className="lane-arrow">next iteration ↩</div>
              </div>
              <figcaption><span>Figure 1.</span> 简化的 Decode 关键路径。红色同步点会同时阻断 CPU 提前准备与 GPU 连续执行。</figcaption>
            </figure>
          </section>

          <section className="chapter-section article-block" id="lab">
            <div className="section-label">04 / INTERACTIVE PROFILE</div>
            <h2>拖动 batch，观察瓶颈移动</h2>
            <p className="section-lede">下面是用于建立直觉的简化模型，不代表某一真实 GPU。每个块的宽度表示相对耗时；切换重叠优化，观察 CPU 与通信占比变化。</p>
            <div className="profile-lab">
              <div className="profile-controls">
                <label><span>BATCH SIZE</span><input type="range" min="1" max="32" value={batch} onChange={(event) => setBatch(Number(event.target.value))} /><output>{batch}</output></label>
                <button className={optimized ? 'active' : ''} onClick={() => setOptimized((value) => !value)}><i />Overlap + Fusion {optimized ? 'ON' : 'OFF'}</button>
              </div>
              <div className="profile-timeline">
                {profile.items.map((stage) => (
                  <div className={`profile-stage ${stage.type}`} style={{ flexGrow: stage.time }} key={stage.name}>
                    <strong>{stage.name}</strong><small>{stage.time.toFixed(1)} ms</small><em>{stage.note}</em>
                  </div>
                ))}
              </div>
              <div className="profile-results">
                <div><span>STEP LATENCY</span><strong>{profile.total.toFixed(1)}<small> ms</small></strong></div>
                <div><span>THROUGHPUT</span><strong>{profile.throughput}<small> tok/s</small></strong></div>
                <div><span>MODE</span><strong>{optimized ? 'OVERLAPPED' : 'BASELINE'}</strong></div>
                <p><b>读图：</b>{batch <= 4 ? '小 batch 下，固定 launch 与同步开销更显眼。' : batch <= 16 ? '中等 batch 开始摊薄固定开销，GEMM 占比上升。' : '大 batch 吞吐更高，但单轮延迟和排队成本也更高。'}</p>
              </div>
            </div>
          </section>

          <section className="chapter-section article-block" id="method">
            <div className="section-label">05 / OPTIMIZE & FALSIFY</div>
            <h2>一个可复用的优化闭环</h2>
            <div className="method-steps">
              <div><span>1</span><strong>Observe</strong><p>用端到端曲线找到异常区间：哪种并发、哪类长度、哪个分位数。</p></div>
              <div><span>2</span><strong>Localize</strong><p>用 CPU/GPU trace 定位 gap、同步、热点 kernel 或通信等待。</p></div>
              <div><span>3</span><strong>Change</strong><p>一次只改变一个假设相关因素，保留 baseline 与回滚路径。</p></div>
              <div><span>4</span><strong>Verify</strong><p>先验证局部机制，再确认端到端收益、精度和其他 workload 无回退。</p></div>
            </div>
            <a className="next-blog" href="../../"><span>回到 Playbook 总入口</span><strong>选择下一篇 Blog →</strong></a>
          </section>
        </article>
      </div>
    </main>
  );
}
