# CUDA Kernel Pipeline 调优与可视化指南

> 目标读者：自己。
> 适用场景：判断一个手写 CUDA GEMM / attention / 量化 kernel 是否"流水跑满了"，
> 特别针对 Hopper warp-specialized kernel（producer/consumer 分工 + TMA + WGMMA）。

---

## 0. 什么叫"pipeline 充分使用"

严格定义：

> kernel 运行的每一个时钟周期，SM 上的 **Tensor Core / 内存子系统 / 数学 ALU** 至少有一个在干活，没有发生大段的"全员等待"。

对 warp specialization GEMM，分解成三件事：

1. **算（Tensor Core）**：WGMMA 是否接近峰值吞吐？
   - H100 fp16 ≈ 989 TFLOPS
   - H100 fp8 / int8 ≈ 1979 TFLOPS
2. **搬（TMA + L2 + HBM）**：内存带宽是否接近峰值？
   - H100 HBM3 ≈ 3 TB/s
3. **重叠**：搬和算是不是并行？consumer 是否经常在等 producer？

**判定原则**：算 / 搬 两项中至少一项达到 80%+（compute-bound 或 memory-bound 之一）。
两项都 < 50% → pipeline 有空泡，**这才是真正的优化机会**。

---

## 1. 工具栈（按重要性）

| 工具 | 干什么 | 学习曲线 |
| --- | --- | --- |
| **Nsight Compute (`ncu`)** | kernel 级深度分析：SM 利用率、pipeline utilization、stall reasons、roofline | ⭐⭐⭐ 必学 |
| **Nsight Systems (`nsys`)** | 系统级时间线：kernel 间 gap、CPU-GPU 同步、cudaMalloc 等 | ⭐⭐ 必学 |
| **PyTorch Profiler** | 跟 PyTorch 一起用，输出 Chrome trace | ⭐ |
| **`cuobjdump` / `nvdisasm`** | 看编译出来的 **SASS**（GPU 真实指令） | ⭐⭐⭐⭐ 深水区 |
| **CUDA Events / `clock64()`** | 自己埋点测时间 | ⭐ |
| **CUTLASS Profiler** | 作为对比 baseline | ⭐⭐ |

主力是 `ncu`。

---

## 2. ncu 上手最小命令

```bash
# 收集完整指标集
ncu --set full -o profile_output your_program

# 只 profile 特定 kernel（按名字正则过滤）
ncu --set full -k regex:humming --launch-skip 0 --launch-count 1 \
    -o profile_output your_program

# 命令行直接看几个关键指标（不开 GUI）
ncu --metrics \
  sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active \
  your_program
```

跑完拿到 `.ncu-rep` 文件，用 **Nsight Compute GUI** 打开（macOS / Windows / Linux 都有）。

源码级关联需要编译时加 `-lineinfo`（NVCC）/ `--generate-line-info`。

---

## 3. 关键指标分类

打开 ncu GUI 在 **Details** 页能看到全部。按"能回答什么问题"分组：

### A. 总体瓶颈定位 —— 算 vs 搬

| 指标 | 含义 | 健康值 |
| --- | --- | --- |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM 总体利用率 | > 80% 优秀 |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | HBM 带宽利用率 | > 80% 是 memory-bound |
| Compute (SM) Pipeline Utilization | 不同 pipeline 各自利用率 | 看哪个最高 |

判断方法：

| SM 高 | Memory 高 | 结论 |
| --- | --- | --- |
| ✓ | ✗ | **compute-bound**（好事，TC 在跑）|
| ✗ | ✓ | **memory-bound**（搬不上来）|
| ✗ | ✗ | **pipeline 有泡泡**（最该优化）|

### B. 算的部分 —— Tensor Core 跑满了吗

最重要的一组：

| Pipeline metric | 对应 | 对应 humming 路径 |
| --- | --- | --- |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed` | HMMA（fp16/bf16）| WGMMA fp16/bf16 |
| `sm__inst_executed_pipe_tensor_op_imma_*` | IMMA（int8）| WGMMA int8 |
| `sm__inst_executed_pipe_tensor_op_qmma_*` | QMMA（fp8/fp4，Hopper/Blackwell）| WGMMA fp8/fp4 |

健康值 **> 80% 接近峰值**。若 fp16 GEMM 只有 30%，说明 Tensor Core 大量时间在等数据。

### C. 搬的部分 —— 内存子系统跑满了吗

| 指标 | 含义 |
| --- | --- |
| `dram__bytes_read.sum`, `dram__bytes_write.sum` | DRAM 实际读写字节数 |
| `l2_tex__throughput.avg.pct_of_peak_sustained_elapsed` | L2 带宽利用率 |
| `lts__t_sector_hit_rate.pct` | L2 命中率（高 = 数据复用好） |
| `smsp__inst_executed_op_ldgsts.sum` | cp.async 指令数（pre-Hopper） |
| `sm__bytes_loaded_from_global_via_tma.*` | TMA 吞吐（Hopper） |

TMA 上不去往往是 **prefetch 不够提前** 或 **stage 数太少**。

### D. ⭐ Warp Stall Reasons —— pipeline 泡泡的真相

最关键一组指标。ncu 把 warp 不执行指令的时间分类到不同原因：

| Stall Reason | 含义 | humming 对应 |
| --- | --- | --- |
| `stall_long_scoreboard` | 等长延迟内存（global load）回来 | producer 等 TMA |
| `stall_mio_throttle` | 等 memory I/O 单元（smem / constant） | smem load 太密 |
| `stall_barrier` | 在 `__syncthreads()` / mbarrier 等同步 | **producer/consumer 互等** |
| `stall_short_scoreboard` | 等短延迟内存（shared memory） | s2r 没及时回来 |
| `stall_no_instruction` | 指令缓存 miss | 一般问题不大 |
| `stall_wait` | 等 warp 屏障 | mbarrier wait |
| `stall_drain` | 退出阶段排空 pipeline | 收尾 |
| `stall_not_selected` | 有得执行但调度器没选它 | warp 太多了 |

**重点**：

- `stall_barrier` 或 `stall_wait` 占比 > 30% → **producer/consumer 节奏不匹配**。要么 producer 太慢搬不上来，要么 consumer 太慢消化不动。这是 "pipeline 没充分使用" 的最常见表现。
- `stall_long_scoreboard` 高 → 数据没到 → **预取不够、stage 不够、TMA issue 太慢**。
- `stall_short_scoreboard` 高 → smem 加载是瓶颈 → s2r pipeline 不够并行。

GUI 的 **Warp State Statistics** 表直接列出各 stall 占百分比 —— **先看这张表**。

### E. Occupancy —— 警告

| 指标 | 含义 |
| --- | --- |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Achieved Occupancy |
| Theoretical Occupancy | 寄存器/smem 决定的上限 |

⚠️ **高 occupancy ≠ 高性能**。Hopper warp specialization kernel 故意只用少量 warp、每 warp 大量寄存器 —— occupancy 可能只有 12.5%，但 Tensor Core 90%+，这才是好。**别盯 occupancy，盯 Tensor Core 利用率 + stall reasons**。

### F. Roofline 图 —— 一图看全局

ncu GUI 自带 Roofline：

```
↑ FLOPS
│     ┌───── Compute Peak (Tensor Core)
│    /
│   /  ← 算力屋顶
│  /
│ /───── Compute Peak (FMA)
│/
├────────────  ← Memory Bandwidth 屋顶
└────────────→ FLOPS / Byte（算术强度）
```

- 点贴近斜线（memory roof）→ memory-bound
- 点贴近水平线（compute roof）→ compute-bound
- 点远低于两条线 → **pipeline 有泡泡，两边都没榨干**

---

## 4. 实战工作流（推荐顺序）

### Step 1：先用 nsys 看时间线

```bash
nsys profile -o timeline --trace=cuda,nvtx your_program
nsys-ui timeline.nsys-rep
```

确认 kernel 占总时间多少。kernel 之间有大 gap 先解决 gap（往往是 cudaMalloc / cudaMemcpy 不在 stream 上）。

### Step 2：用 ncu 看 kernel 内部

```bash
ncu --set full -k humming -o kernel_profile your_program
ncu-ui kernel_profile.ncu-rep
```

**第一眼看 Speed Of Light 页**：会给出 SM vs Memory 利用率 + 白话总结：
- "This kernel exhibits low compute throughput..."
- "Memory is more heavily utilized than compute..."

### Step 3：看 Warp State Statistics

找最大的 stall reason，按 §3.D 对应到设计层面。

### Step 4：看 Source/SASS 关联

编译加 `-lineinfo` 后，GUI 的 **Source** 页能把每个指标关联到源码行：
- 这一行执行了多少次
- 在这一行 stall 了多久、stall 原因

特别有用的查询：
- WGMMA 这一行是不是真的在 stall？
- mbarrier wait 那行等了多久？
- TMA issue 那行的 throughput 是多少？

### Step 5：和 baseline 对比

用同样的 problem size 跑 cuBLAS / CUTLASS：

```bash
ncu --set full -k cublas -o baseline your_baseline_program
```

如果 cuBLAS 跑到 85% TC 利用率而你只有 60%，**就是 25 个点的优化空间**。

---

## 5. Warp Specialization Kernel 特殊关注点

### 5.1 Producer / Consumer 谁是瓶颈

ncu 的 **Memory Workload Analysis** 页可以按 warp 分组看指标。Producer / Consumer 因 `threadIdx.x` 不同会被分开统计。

- Producer warp `stall_long_scoreboard` 高 → TMA 还没回来 → 加 stage 没用，要改预取
- Consumer warp `stall_barrier` / `stall_wait` 高 → 在等 producer → producer 太慢 / stage 不够

### 5.2 mbarrier 等待时长（手动埋 clock64）

ncu 没有 mbarrier 专属指标，但 `stall_barrier` 包含。可以源码埋探针：

```cpp
__device__ uint64_t mbarrier_wait_cycles = 0;   // debug-only

uint64_t t0 = clock64();
consumer.wait_stage(stage_id);
uint64_t t1 = clock64();
atomicAdd(&mbarrier_wait_cycles, t1 - t0);
```

跑完看累加值占总 cycles 比例。**> 20% 就要怀疑 pipeline 设计**。

### 5.3 WGMMA 之间是否真的重叠

WGMMA 异步，发了就能算别的，最后 `wgmma.wait_group` 收尾。看 SASS：

```bash
cuobjdump --dump-sass your.cubin | grep -A2 -B2 WGMMA
```

- 看到 **WGMMA 之后紧跟 WGMMA**（中间穿插 load / transform）→ 流水做得好 ✓
- 看到 **WGMMA 后立刻 wait_group** → 串行 WGMMA，没流水 ✗

### 5.4 TMA 是真在用还是 fallback 到 cp.async

```bash
cuobjdump --dump-sass your.cubin | grep -E "UBULK|cp.async.bulk"
```

应该看到 `UBULK`（TMA bulk load）。只看到普通 `LDG` 说明 TMA 路径没启用。

---

## 6. 自己埋点的最简验证

### 方法 A：注释 transform 看 WGMMA 上限

```cpp
// 临时把 transform 注掉
// mma.transform_b((warp_k_iter_id + 1) % 2);
```

跑一次看 Tensor Core 利用率。跳了 20 个点 → transform 在跟 WGMMA 抢资源。

### 方法 B：扫 stage 数画曲线

`kNumStages` ∈ {2, 3, 4, 5, 6, 8} 各跑一次，画 stage 数 vs 性能：

- 单调上升 → 当前 stage 不够，预取没拉满
- 某值后饱和 → 那就是最优 stage 数
- 某值后下降 → smem 装不下，occupancy 掉了

### 方法 C：torch profiler（最低成本）

```python
import torch.profiler as p

with p.profile(
    activities=[p.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    your_op(...)

prof.export_chrome_trace("trace.json")
```

`trace.json` 拖进 `chrome://tracing` 或 `perfetto.dev`。能看到 kernel 和 memcpy 的并行情况。

---

## 7. Cheat Sheet

### "我该看什么"决策树

```
开始
 │
 ├─ kernel 占总时间多少？───── 不到 50% ── 先用 nsys 解决 host 侧 gap
 │                       └── > 50% ── 进 ncu
 │
 ├─ ncu Speed Of Light：SM 和 Memory 谁高？
 │     │
 │     ├─ SM 高 (>80%) ─────────── compute-bound，看 WGMMA 是否能换更优 shape
 │     ├─ Memory 高 (>80%) ──────── memory-bound，看 prefetch / 数据复用 / L2 命中
 │     └─ 都低 (<50%)  ─────────── pipeline 有泡泡，进下一步
 │
 ├─ Warp State Statistics：最大 stall 是什么？
 │     │
 │     ├─ stall_barrier / stall_wait ── producer/consumer 节奏不匹配
 │     │     │
 │     │     ├─ Producer 慢？────── 检查 TMA / prefetch / stage 数
 │     │     └─ Consumer 慢？────── 检查 WGMMA shape / transform overhead
 │     │
 │     ├─ stall_long_scoreboard ── 内存延迟暴露 → 加 stage / 提前预取
 │     ├─ stall_short_scoreboard ─ smem 加载瓶颈 → 看 s2r pipeline
 │     └─ stall_mio_throttle ─── smem 访问太密 → 看 swizzle / bank conflict
 │
 └─ Roofline：点在哪？───── 看是被哪条 roof 卡住
```

### 关键 ncu 指标速查

```bash
# Tensor Core 利用率（fp16/bf16）
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed

# DRAM 带宽利用率
dram__throughput.avg.pct_of_peak_sustained_elapsed

# L2 命中率
lts__t_sector_hit_rate.pct

# Achieved occupancy（仅参考）
sm__warps_active.avg.pct_of_peak_sustained_active

# stall reasons 一组（看占比）
smsp__average_warps_issue_stalled_*.ratio
```

### 几个反直觉的"知识点"

- **高 occupancy ≠ 高性能**：warp specialization kernel 故意低 occupancy。
- **`__syncthreads()` 不一定贵**：贵的是它前面没事可做的 warp 在等。
- **L2 命中率不是越高越好**：高了说明数据被反复读，可能本身就是低算术强度。
- **WGMMA 是异步的**：单看一条 WGMMA 延迟没意义，看它后面有没有事干填满 gap。
- **stage 数不是越多越好**：smem 是稀缺资源，stage 多了 occupancy 掉，反而慢。

---

## 附录：相关链接

- Nsight Compute 用户手册：<https://docs.nvidia.com/nsight-compute/>
- Nsight Compute Metrics Reference：<https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference>
- CUDA C++ Programming Guide (Warp Specialization / TMA / WGMMA)：<https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- CUTLASS 3.x 文档（学习 SOTA Hopper kernel 设计）：<https://github.com/NVIDIA/cutlass>
