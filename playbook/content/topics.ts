export type TopicStatus = 'published' | 'draft' | 'planned';

export type Topic = {
  slug: string;
  title: string;
  subtitle: string;
  summary: string;
  category: '系统' | '算子' | '并行' | '工程';
  level: '基础' | '进阶' | '深入';
  status: TopicStatus;
  updated: string;
  href?: string;
  accent: 'blue' | 'purple' | 'orange' | 'green';
  concepts: string[];
};

/**
 * 首页只依赖这份 registry，不依赖 blog 的物理目录。
 * href 可以指向 playbook 内部路由、仓库中的静态 HTML，或任意外部 URL。
 */
export const topics: Topic[] = [
  {
    slug: 'decode-path',
    title: '一次 Decode，时间花在哪？',
    subtitle: '端到端关键路径',
    summary: '从 scheduler 到 sampling，建立 TTFT、TPOT、吞吐与 GPU 利用率之间的第一层直觉。',
    category: '系统',
    level: '基础',
    status: 'published',
    updated: '2026-09-02',
    href: 'blog/decode-path',
    accent: 'blue',
    concepts: ['关键路径', '性能指标', 'Profiler'],
  },
  {
    slug: 'continuous-batching',
    title: 'Continuous Batching',
    subtitle: '调度器如何喂饱 GPU',
    summary: '用时间线理解请求插入、chunked prefill、抢占与 batch size 对延迟吞吐曲线的影响。',
    category: '系统',
    level: '进阶',
    status: 'draft',
    updated: 'COMING NEXT',
    accent: 'green',
    concepts: ['Scheduling', 'Chunked Prefill', 'Queueing'],
  },
  {
    slug: 'kv-cache',
    title: 'KV Cache 与 Paged Attention',
    subtitle: '显存里的虚拟内存系统',
    summary: '从连续缓存到分页管理，观察 block table、碎片率与 prefix cache 如何改变容量和访存。',
    category: '系统',
    level: '进阶',
    status: 'planned',
    updated: 'PLANNED',
    accent: 'purple',
    concepts: ['Block Table', 'Prefix Cache', 'Memory'],
  },
  {
    slug: 'flash-attention',
    title: 'FlashAttention',
    subtitle: '把 IO 复杂度画出来',
    summary: '沿着 tile 的搬运路径理解 online softmax、shared memory 复用与不同 attention backend。',
    category: '算子',
    level: '深入',
    status: 'planned',
    updated: 'PLANNED',
    accent: 'orange',
    concepts: ['Tiling', 'Online Softmax', 'IO-aware'],
  },
  {
    slug: 'gemm-pipeline',
    title: 'GEMM：从 Tile 到 Pipeline',
    subtitle: 'Tensor Core 的喂数艺术',
    summary: '拆解 CTA / warp 分工、TMA、WGMMA、多 stage pipeline，以及小 M decode GEMM 的特殊困难。',
    category: '算子',
    level: '深入',
    status: 'draft',
    updated: 'IN PROGRESS',
    accent: 'blue',
    concepts: ['Tensor Core', 'TMA', 'Warp Specialization'],
  },
  {
    slug: 'moe',
    title: 'MoE：路由、重排与 Expert GEMM',
    subtitle: '稀疏模型的真实代价',
    summary: '把 TopK、token permutation、grouped GEMM、all-to-all 与负载均衡放进同一张执行图。',
    category: '算子',
    level: '深入',
    status: 'planned',
    updated: 'PLANNED',
    accent: 'purple',
    concepts: ['TopK', 'Grouped GEMM', 'All-to-All'],
  },
  {
    slug: 'sm90-megamoe',
    title: 'SM90 Humming MegaMoE',
    subtitle: 'FP8 × MXFP4 Persistent Kernel',
    summary: '沿着真实代码拆解 route dispatch、MXFP4 在线解码、WGMMA、symmetric ring、remote scatter 与 combine，并复盘 203 轮调优。',
    category: '算子',
    level: '深入',
    status: 'published',
    updated: '2026-09-02',
    href: 'megamoe',
    accent: 'purple',
    concepts: ['SM90', 'MXFP4', 'Warp Specialization'],
  },
  {
    slug: 'quantization',
    title: '量化推理：精度不是唯一变量',
    subtitle: 'FP8 / FP4 / INT4',
    summary: '比较权重、激活与 KV cache 量化路径，理解 scale 粒度、反量化开销和硬件支持边界。',
    category: '工程',
    level: '进阶',
    status: 'planned',
    updated: 'PLANNED',
    accent: 'green',
    concepts: ['FP8', 'NVFP4', 'Calibration'],
  },
  {
    slug: 'parallelism',
    title: '并行策略：TP、EP、PP 与 PD',
    subtitle: '扩展到多卡、多机',
    summary: '用通信量和关键路径选择并行策略，而不是只记缩写；同时观察不同 workload 的扩展效率。',
    category: '并行',
    level: '进阶',
    status: 'planned',
    updated: 'PLANNED',
    accent: 'orange',
    concepts: ['Tensor Parallel', 'Expert Parallel', 'Disaggregation'],
  },
];

export const categories = ['全部', '系统', '算子', '并行', '工程'] as const;
