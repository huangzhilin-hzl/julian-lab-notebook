# LLM Inference Playbook

推理优化知识总入口。首页提供知识地图、主题筛选与更新状态；每个子主题是一篇独立 blog，实际目录不受首页结构限制。

## 当前结构

```text
playbook/
├── app/
│   ├── page.tsx                    # Playbook 总入口
│   ├── blog/decode-path/page.tsx  # 第一篇示例 blog
│   ├── globals.css                 # 首页与文章共用视觉系统
│   └── layout.tsx                  # 全站 metadata
├── content/topics.ts              # 主题 registry（唯一入口）
└── public/                         # 分享预览等静态资源
```

## 添加一篇 Blog

1. Blog 可以放在 `playbook/app/` 的任意内部路由，也可以是仓库中的静态页面或外部 URL。
2. 在 `content/topics.ts` 增加一条 topic，填写分类、状态、摘要和 `href`。
3. 首页会自动生成卡片并参与分类筛选；没有 `href` 的条目显示为规划中。

内部文章示例：

```ts
{
  slug: 'decode-path',
  title: '一次 Decode，时间花在哪？',
  category: '系统',
  status: 'published',
  href: 'blog/decode-path',
}
```

如果文章不在 `playbook/` 目录，只需把 `href` 改成它最终可访问的相对地址或完整 URL；首页无需知道文章的物理位置。

## 本地运行

```bash
npm install
npm run dev
```

## 构建

```bash
# Cloudflare / OpenAI Sites 兼容构建
npm run build

# GitHub Pages 静态导出，产物位于 out/
npm run build:pages
```

部署到 GitHub Project Pages 时，通过 `NEXT_PUBLIC_BASE_PATH` 指定仓库路径。例如站点位于 `https://user.github.io/repo/playbook/`：

```bash
NEXT_PUBLIC_BASE_PATH=/repo/playbook npm run build:pages
```

`public/.nojekyll` 会随静态导出复制，避免 GitHub Pages 忽略 `_next` 资源目录。
