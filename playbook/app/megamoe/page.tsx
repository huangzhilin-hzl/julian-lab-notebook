import type { Metadata } from 'next';
import { megaMoeReportHtml } from '../../megamoe/report-content';

export const metadata: Metadata = {
  title: 'SM90 Humming MegaMoE：从路由到 WGMMA | Inference Playbook',
  description: 'DeepGEMM SM90 FP8 × MXFP4 MegaMoE 的 as-built 设计与实现：配置、张量、persistent kernel、symmetric memory、dispatch、WGMMA、combine 与调优收益。',
  openGraph: { images: [] },
  twitter: { images: [] },
};

export default function MegaMoeArticle() {
  return (
    <main id="top">
      <header className="site-header article-site-header">
        <a className="wordmark" href="../">
          <span className="wordmark-glyph">I<span>/</span>P</span>
          <span>Inference Playbook</span>
        </a>
        <a className="repo-link" href="../">返回总入口 <span>↖</span></a>
      </header>

      <div className="reading-layout mega-reading-layout">
        <aside className="toc" aria-label="本页目录">
          <p>ARTICLE CONTENTS</p>
          <ol>
            <li className="active"><a href="#top"><span>00</span>结论先行</a></li>
            <li><a href="#section-1"><span>01</span>Flash / Pro 配置</a></li>
            <li><a href="#section-2"><span>02</span>Tensor 与前处理</a></li>
            <li><a href="#section-3"><span>03</span>Kernel 启动与选型</a></li>
            <li><a href="#section-4"><span>04</span>Symmetric Memory</a></li>
            <li><a href="#section-5"><span>05</span>主执行流程</a></li>
            <li><a href="#section-6"><span>06</span>Combine</a></li>
            <li><a href="#section-7"><span>07</span>优化与增益</a></li>
            <li><a href="#section-8"><span>08</span>NSYS / Roofline</a></li>
          </ol>
          <div className="toc-note"><span className="live-dot" /> AS-BUILT<small>KERNELS / MEGAMOE</small></div>
        </aside>

        <article className="article mega-article">
          <div className="breadcrumbs"><a href="../">PLAYBOOK</a><b>/</b><span>KERNELS</span><b>/</b><em>SM90 MEGAMOE</em></div>
          <section className="article-hero mega-hero">
            <div className="chapter-badge">SM90 / H20 · KERNEL DEEP DIVE</div>
            <h1>SM90 Humming <span>MegaMoE</span></h1>
            <p className="abstract">FP8 activation × MXFP4 weight MegaMoE 的实现拆解：配置、数据布局、persistent kernel、dispatch、WGMMA、combine 与调优收益。</p>
            <div className="article-meta">
              <div className="author-mark">JL</div>
              <p><strong>Julian Lab</strong><span>实现拆解 · 深入</span></p>
              <div className="topic-tags"><span>SM90</span><span>MXFP4</span><span>WGMMA</span></div>
            </div>
          </section>

          <section className="tldr-card mega-tldr">
            <div className="tldr-title"><span>TL;DR</span><small>三个结论</small></div>
            <ol>
              <li><b>单 kernel：</b>dispatch、两层 GEMM、scatter 与 combine 全部融合。</li>
              <li><b>256 threads/CTA：</b>WG0 搬运和调度，WG1 解码、WGMMA 与 epilogue。</li>
              <li><b>稳定点 R203：</b>22 点 aggregate gap 从 +20.36% 收敛到 -1.86%。</li>
            </ol>
          </section>

          <div className="mega-report-content" dangerouslySetInnerHTML={{ __html: megaMoeReportHtml }} />

          <a className="next-blog" href="../"><span>回到 Playbook 总入口</span><strong>选择下一篇 Blog →</strong></a>
        </article>
      </div>
    </main>
  );
}
