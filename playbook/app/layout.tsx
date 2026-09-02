import type { Metadata } from 'next';
import './globals.css';

const title = 'LLM Inference Playbook — 推理优化实战手册';
const description = '用独立 blog、手绘机制图、交互实验与真实代码，理解 LLM 推理系统、性能优化和核心算子。';

export const metadata: Metadata = {
  metadataBase: new URL('https://huangzhilin-hzl.github.io/julian-lab-notebook/playbook/'),
  title,
  description,
  openGraph: {
    title,
    description,
    type: 'website',
    images: [{ url: 'og.png', width: 1731, height: 909, alt: 'LLM Inference Playbook — Systems, Kernels, Visuals' }],
  },
  twitter: {
    card: 'summary_large_image',
    title,
    description,
    images: ['og.png'],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
