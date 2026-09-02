'use client';

import { useMemo, useState } from 'react';
import { categories, topics } from '../content/topics';

const mapItems = [
  ['01', '系统与指标', '调度 · KV Cache · Profiling'],
  ['02', 'Kernel 机制', 'GEMM · Attention · MoE'],
  ['03', '并行与通信', 'TP · EP · AllReduce'],
  ['04', '量化与部署', 'FP8 · FP4 · Serving'],
];

const statusText = { published: '已发布', draft: '编写中', planned: '待规划' };

export default function Home() {
  const [category, setCategory] = useState<(typeof categories)[number]>('全部');
  const visibleTopics = useMemo(
    () => category === '全部' ? topics : topics.filter((topic) => topic.category === category),
    [category],
  );

  return (
    <main id="top">
      <header className="site-header">
        <a className="wordmark" href="#top">
          <span className="wordmark-glyph">I<span>/</span>P</span>
          <span>Inference Playbook<small>推理优化实战手册</small></span>
        </a>
        <nav aria-label="主导航">
          <a href="#map">知识地图</a>
          <a href="#topics">全部主题</a>
          <a href="#visuals">可视化</a>
          <a href="#contribute">添加 Blog</a>
        </nav>
        <a className="repo-link" href="https://github.com/huangzhilin-hzl/julian-lab-notebook" target="_blank" rel="noreferrer">
          GitHub <span>↗</span>
        </a>
      </header>

      <div className="hub-layout">
        <aside className="hub-sidebar" aria-label="Playbook 导航">
          <p>PLAYBOOK INDEX</p>
          <a className="sidebar-home" href="#top"><span>⌂</span>总览</a>
          <div className="sidebar-group">
            <small>学习路径</small>
            {mapItems.map(([number, title]) => (
              <a href="#map" key={number}><span>{number}</span>{title}</a>
            ))}
          </div>
          <div className="sidebar-group">
            <small>浏览方式</small>
            <a href="#topics"><span>↳</span>按主题</a>
            <a href="#visuals"><span>↳</span>按可视化</a>
            <a href="#recent"><span>↳</span>最近更新</a>
          </div>
          <div className="toc-note"><span className="live-dot" /> 持续更新<small>CURATED BY JULIAN LAB</small></div>
        </aside>

        <div className="hub-content">
          <div className="breadcrumbs"><span>JULIAN LAB</span><b>/</b><em>INFERENCE PLAYBOOK</em></div>

          <section className="hub-hero">
            <div className="hub-kicker">SYSTEMS · KERNELS · VISUALS</div>
            <h1>LLM 推理优化<br /><span>Playbook</span></h1>
            <p>
              一张通往推理系统各层的地图。每个子主题都是一篇独立 blog：从性能现象出发，用手绘机制图、交互实验和真实代码建立工程直觉。
            </p>
            <div className="hub-actions">
              <a className="start-button" href="blog/decode-path">从第一篇开始 <span>→</span></a>
              <a className="quiet-button" href="#topics">浏览全部主题 ↓</a>
            </div>
            <dl className="hub-stats">
              <div><dt>{topics.length}</dt><dd>主题</dd></div>
              <div><dt>{topics.filter((topic) => topic.status === 'published').length}</dt><dd>已发布</dd></div>
              <div><dt>4</dt><dd>知识层级</dd></div>
            </dl>
          </section>

          <section className="map-section" id="map">
            <div className="hub-section-head">
              <div><span>01</span><small>KNOWLEDGE MAP</small></div>
              <h2>先找到你所在的性能层</h2>
              <p>同一个“慢”可能来自四个不同层级。沿着请求路径向下定位，再进入对应 blog。</p>
            </div>

            <figure className="system-map-card">
              <div className="map-title">Inference optimization stack</div>
              <div className="map-flow">
                {mapItems.map(([number, title, detail], index) => (
                  <div className={`map-layer map-layer-${index + 1}`} key={number}>
                    <span>{number}</span>
                    <strong>{title}</strong>
                    <small>{detail}</small>
                    <em>{index === 0 ? '端到端现象' : index === 1 ? 'GPU 热点' : index === 2 ? '扩展瓶颈' : '精度 / 成本'}</em>
                  </div>
                ))}
              </div>
              <div className="map-arrows"><span>请求路径 →</span><i /><b>定位瓶颈后再下钻，不从“热门优化”开始</b></div>
              <figcaption><span>Figure 1.</span> Playbook 的四层知识地图。每篇 blog 都会标注它影响的层级与最终指标。</figcaption>
            </figure>
          </section>

          <section className="topic-section" id="topics">
            <div className="hub-section-head compact">
              <div><span>02</span><small>TOPIC REGISTRY</small></div>
              <h2>子主题 Blog</h2>
              <p>独立阅读，互相链接；实际文件位置不限制。</p>
            </div>

            <div className="topic-filters" aria-label="按分类筛选">
              {categories.map((item) => (
                <button className={category === item ? 'active' : ''} key={item} onClick={() => setCategory(item)}>{item}</button>
              ))}
              <span>{visibleTopics.length} ARTICLES</span>
            </div>

            <div className="topic-grid">
              {visibleTopics.map((topic, index) => {
                const content = (
                  <>
                    <div className="topic-card-meta"><span>{String(index + 1).padStart(2, '0')} / {topic.category}</span><i className={`status-${topic.status}`}>{statusText[topic.status]}</i></div>
                    <div className={`topic-visual ${topic.accent}`} aria-hidden="true">
                      <div className="visual-chip">{topic.subtitle}</div>
                      <div className="visual-bars"><i /><i /><i /><i /><i /></div>
                    </div>
                    <h3>{topic.title}</h3>
                    <p>{topic.summary}</p>
                    <div className="topic-concepts">{topic.concepts.map((concept) => <span key={concept}>{concept}</span>)}</div>
                    <footer><span>{topic.level}</span><time>{topic.updated}</time><b>{topic.href ? '阅读 →' : '即将上线'}</b></footer>
                  </>
                );
                return topic.href ? <a className="topic-card" href={topic.href} key={topic.slug}>{content}</a> : <article className="topic-card is-planned" key={topic.slug}>{content}</article>;
              })}
            </div>
          </section>

          <section className="visuals-section" id="visuals">
            <div className="hub-section-head compact">
              <div><span>03</span><small>VISUAL LAB</small></div>
              <h2>图要解释机制，也要回答性能问题</h2>
              <p>统一使用蓝、紫、橙、绿区分数据路径；交互图保留条件、单位和可复现结论。</p>
            </div>
            <div className="visual-principles">
              <div><span className="scribble blue">01</span><strong>Mechanism</strong><p>手绘流程图解释数据如何移动、线程如何协作。</p></div>
              <div><span className="scribble purple">02</span><strong>Timeline</strong><p>时间线暴露 kernel gap、同步点和重叠机会。</p></div>
              <div><span className="scribble orange">03</span><strong>Benchmark</strong><p>曲线同时给出 workload、硬件、精度与测量口径。</p></div>
              <div><span className="scribble green">04</span><strong>Source</strong><p>结论落回真实代码、PR、profile 或复现脚本。</p></div>
            </div>
          </section>

          <section className="contribute-section" id="contribute">
            <div><span>ADD A NEW BLOG</span><h2>目录不固定，入口保持统一。</h2></div>
            <ol>
              <li><span>1</span><p><strong>写在哪里都可以</strong>内部路由、仓库静态 HTML、外部文章都能接入。</p></li>
              <li><span>2</span><p><strong>登记 metadata</strong>在 topic registry 中补标题、分类、状态与 href。</p></li>
              <li><span>3</span><p><strong>自动出现在首页</strong>筛选、状态和更新信息由首页统一呈现。</p></li>
            </ol>
          </section>
        </div>
      </div>
    </main>
  );
}
