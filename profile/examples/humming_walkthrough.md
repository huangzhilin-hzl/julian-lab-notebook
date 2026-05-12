# 手把手 profile `humming_ws` GEMM kernel

> 用 `open-huming/benchmarks/bench_humming.py` 作为入口，端到端跑一遍 nsys + ncu，
> 学会判断 `humming_ws.cuh` 里那套 warp-specialized GEMM 的 pipeline 是否充分使用。
>
> 节奏建议：每一步跑完先回过头看 §"该看什么"，再进下一步。

---

## Step 0：环境检查（先全部确认 ✓ 再继续）

跑下面这些命令，每条都要有合理输出：

```bash
# 1. CUDA & GPU
nvidia-smi                         # 看到 GPU 型号 (H100/H200/A100/...) 和 driver
nvcc --version                     # NVCC 编译器版本

# 2. Profiling 工具
ncu --version                      # Nsight Compute CLI
nsys --version                     # Nsight Systems CLI
which ncu-ui 2>/dev/null || true   # GUI 客户端 (没有也行，可以拷 .ncu-rep 回本地看)

# 3. humming 装好了
cd /path/to/open-huming
python -c "from humming.layer import HummingLayer; print('ok')"
```

如果 `ncu` 命令找不到：CUDA toolkit 自带，路径通常在 `/usr/local/cuda/bin/ncu`，加进 PATH。

⚠️ **权限注意**：`ncu` 默认需要 GPU 性能计数器权限。报错 `ERR_NVGPUCTRPERM` 的话：

```bash
# 方案 A: 永久放开（需 root）
sudo sh -c "echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' > /etc/modprobe.d/nvidia-profiler.conf"
sudo update-initramfs -u && sudo reboot

# 方案 B: 临时用 sudo 跑 ncu
sudo ncu ...
```

---

## Step 1：先跑一次 baseline，确认能 work

我们选一个**有代表性的 prefill GEMM**：

- `M = 4096`（一个 batch 的总 token 数）
- `N = 4096`, `K = 4096`
- 权重 INT4 + scale BF16，激活 BF16（典型 LLM 量化推理场景）

```bash
cd /path/to/open-huming

python benchmarks/bench_humming.py \
    --shape_n 4096 --shape_k 4096 \
    --a_dtype bfloat16 --b_dtype int4 --bs_dtype bfloat16 --c_dtype bfloat16 \
    --weight_scale_group_size 128 \
    --shape_m_list 4096
```

⚠️ **第一次跑会触发 JIT 编译**，可能要 30s ~ 几分钟（编译 kernel）。**第二次跑就快了**（命中缓存）。

**预期输出**：会打一张 1 行的表格：

```
+----------+--------+--------------+---------------+
|  shape_m |   time |  memory_gbps |   compute_tops |
+==========+========+==============+===============+
|     4096 | x.xxxx |     xxxx.xx |        xxx.xx |
+----------+--------+--------------+---------------+
```

记下 `compute_tops` 和 `memory_gbps`。这俩数是后面的目标 —— 优化前后看它们变化。

### 该看什么
- `compute_tops` ≈ 该硬件 fp16 TC 峰值的 60~85% → kernel 在正常范围
- 远低于 → 这台 kernel 在你这台机器+这个 shape 上**根本没拉满**，profile 一探究竟

---

## Step 2：nsys 看时间线（系统视角）

```bash
nsys profile \
    --trace=cuda,nvtx,osrt \
    -o nsys_humming_baseline \
    --force-overwrite=true \
    python benchmarks/bench_humming.py \
        --shape_n 4096 --shape_k 4096 \
        --a_dtype bfloat16 --b_dtype int4 --bs_dtype bfloat16 --c_dtype bfloat16 \
        --weight_scale_group_size 128 \
        --shape_m_list 4096
```

生成 `nsys_humming_baseline.nsys-rep`。

### 看时间线
**方案 A（有 GUI）**：
```bash
nsys-ui nsys_humming_baseline.nsys-rep
```

**方案 B（无 GUI，命令行摘要）**：
```bash
nsys stats --report cuda_gpu_kern_sum nsys_humming_baseline.nsys-rep | head -30
```

会输出每个 kernel 的 **total time / instances / avg time**。

### 该看什么
1. 找到 `humming` 开头的 kernel（会有几个 jit 出来的实例）。
2. 它的 **total time 占总 GPU 时间的百分比** —— 应该 > 50%。
3. 如果不到 50%，先用 GUI 看 kernel 之间有没有大 gap（往往是 cudaMalloc / 不在 stream 上的 memcpy）—— **先解决 gap 再做 ncu**。

### ✅ Checkpoint 1
告诉我：
- humming kernel 一次跑大概多少 ms？
- kernel 之间有没有 gap？

---

## Step 3：ncu 抓 humming kernel（kernel 视角）

```bash
ncu --set full \
    -k regex:humming \
    --launch-skip 200 --launch-count 1 \
    --target-processes all \
    -o ncu_humming_baseline \
    python benchmarks/bench_humming.py \
        --shape_n 4096 --shape_k 4096 \
        --a_dtype bfloat16 --b_dtype int4 --bs_dtype bfloat16 --c_dtype bfloat16 \
        --weight_scale_group_size 128 \
        --shape_m_list 4096
```

**参数解释**：
- `--set full`：抓最全的指标集（慢，但信息全）
- `-k regex:humming`：只 profile 名字匹配 `humming` 的 kernel
- `--launch-skip 200`：跳过前 200 次 launch（避开 JIT compile + warmup）
- `--launch-count 1`：只抓 1 次（profile 一次能用，多了浪费时间）

⚠️ **ncu 抓一次比 nsys 慢得多**（要重放 kernel 几十次采样不同 counter）。一次几分钟正常。

### 生成 `ncu_humming_baseline.ncu-rep`，用以下任一方式看：

**方案 A（有 GUI）**：
```bash
ncu-ui ncu_humming_baseline.ncu-rep
```

**方案 B（无 GUI，命令行精简）**：
```bash
ncu --import ncu_humming_baseline.ncu-rep --print-summary per-kernel | head -80
```

**方案 C（只挑几个指标，最快）**：
```bash
ncu --import ncu_humming_baseline.ncu-rep \
    --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_qmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
lts__t_sector_hit_rate.pct,\
sm__warps_active.avg.pct_of_peak_sustained_active
```

### ✅ Checkpoint 2
把这几个值发给我：
- `sm__throughput` 百分比
- `dram__throughput` 百分比
- 3 个 TC 指标（HMMA/IMMA/QMMA）哪个非 0？值多少？
- `lts__t_sector_hit_rate`（L2 命中率）

---

## Step 4：Speed Of Light 一图看瓶颈（GUI 里看，没 GUI 看命令行）

ncu GUI 打开后，**第一个看的就是 "Speed Of Light" 页**。

它会用两条柱状图告诉你：

```
SM [Compute]:          ██████████████░░░░  72%
Memory:                ███████░░░░░░░░░░░  35%
```

然后下方有一段白话**自动诊断**：
- "This kernel is utilizing the compute resources well..."  → compute-bound
- "Memory utilization is significantly higher..."          → memory-bound
- "Neither compute nor memory are well-utilized..."        → **pipeline 有泡泡**（重点关注）

### 该看什么
判断属于下面哪一类（决定后续 ncu 怎么继续看）：

| Compute | Memory | 结论 | 下一步看哪里 |
| --- | --- | --- | --- |
| 高 (>80%) | 任何 | compute-bound ✓ | 看 WGMMA shape / `kPartMmaShapeK`，找更优 tile shape |
| 任何 | 高 (>80%) | memory-bound | 看 prefetch、stage 数、L2 复用、TMA 是否启用 |
| 低 | 低 | **pipeline 泡泡** | 直接进 Step 5（看 stall）|

### ✅ Checkpoint 3
告诉我 SOL 页的两个数 + 它自动写的那段话。

---

## Step 5：Warp State Statistics —— pipeline 泡泡的真相

GUI 上找到 **"Warp State Statistics"** 节，会看到一张表，列出 warp 每种 stall 占的周期百分比。

无 GUI 命令行版：

```bash
ncu --import ncu_humming_baseline.ncu-rep \
    --metrics smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio
```

### 对照诊断

| 最大 stall | 翻译 | 在 humming 里很可能是 |
| --- | --- | --- |
| `stall_long_scoreboard` | 等 global memory load | producer 等 TMA 还没回来 → **加 stage 数 / 提前 prefetch** |
| `stall_short_scoreboard` | 等 smem load | s2r pipeline 不够并行 → 看 `s2r_pipeline.cuh` |
| `stall_barrier` / `stall_wait` | 在 mbarrier / __syncthreads 等 | **producer/consumer 节奏不匹配**（详见 Step 6）|
| `stall_mio_throttle` | smem 访问太密 | bank conflict / swizzle 不对 |
| `stall_not_selected` | 调度器没轮到它 | 实际是有得算 → 多半是好事，不优化 |

### ✅ Checkpoint 4
告诉我：占比最大的 3 个 stall reason + 它们的百分比。

---

## Step 6：Source 关联到 `humming_ws.cuh` 具体哪一行

这一步用 GUI 最直观。前提：`humming_ws.cuh` 编译时带 `-lineinfo`（humming JIT 默认应该带）。

GUI 顶上选 **Source** 页，会看到：左边是源码、右边是每行的指标列。

**重点找这几行**：

| `humming_ws.cuh` 行 | 你想确认 |
| --- | --- |
| `consumer.wait_stage<true>(kNumStages)` (~L149) | 等 producer 是否长？ `stall_barrier` 高就是这里 |
| `mma.run(stage_id, warp_k_iter_id)` (~L162) | WGMMA 真正在跑的行；TC active 占比高 |
| `mma.transform_b(...)` (~L151,170) | dequant 是否在跟 WGMMA 抢资源（看这行 inst 量）|
| `s2r_pipe.load_stage_iter(...)` (~L150,161) | smem→reg 加载是否瓶颈 |
| `producer.wait_stage(stage_id)` (~L117) | producer 是否在等 consumer 消化 |

### 怎么用
比对**相邻几行**的 stall 类型 + 占比，找到那条"特别贵"的指令。比如：
- `wait_stage` 行 `stall_barrier` 占 35% → **consumer 经常在等 producer**，producer 跟不上 → 加 stage / 加 producer warp。
- `mma.run` 行 TC active 100%，但全 kernel TC active 只有 50% → **WGMMA 自己快，但 kernel 大部分时间没在跑 WGMMA**（流水有空洞）。

### ✅ Checkpoint 5
告诉我源码里**最贵的 3 行**（行号 + 主要 stall）。

---

## Step 7：改一个参数，对比

到这里你已经有完整 baseline。下面挑一个改动**重新跑一遍**，看指标变化 —— 这是真正学会读 profile 的最快路径。

### 改动 A：换 stage 数

humming 用 heuristics 自动选 `num_stages`。你可以**手动 dump 当前 tuning_config**：

```python
# 在 bench_humming.py 的 tuning_config 那行下加一句临时 print：
print("tuning_config:", tuning_config)
```

记下当前 `num_stages`，然后**手动指定一个不同的值**（修改 `get_heuristics_config` 返回的 config，或直接传 dict 给 layer）。

跑 `num_stages = 3, 4, 5, 6, 8`，每个出一份 .ncu-rep，对比：
- 哪个 stage 数 TC active 最高？
- 增大 stage 后 `stall_long_scoreboard` 有没有下降？
- 哪个 stage 数 occupancy 掉下来了？（smem 装不下）

### 改动 B：换 dtype

把 `--a_dtype bfloat16` 换成 `--a_dtype float8e4m3`（如果 GPU 支持）。
重跑 ncu，对比：
- TC 利用率从 HMMA 变成 QMMA 了吗？
- compute_tops 翻倍了吗？（fp8 峰值理论是 fp16 的 2 倍）

### 改动 C：换 problem size

把 `--shape_m_list 4096` 换成 `--shape_m_list 16` 或 `--shape_m_list 64`，模拟 decode 场景。

- 这时是 memory-bound 了吗？
- humming 在小 M 上 TC 利用率怎样？这通常是 LLM decode 的痛点。

### ✅ Checkpoint 6
报你做的对比 + 结论。

---

## 附录 A：常见踩坑

1. **第一次跑 ncu 巨慢**：正常，要重放 kernel 几十次采指标。后续跑相同 kernel 会快点。
2. **ncu 报 ERR_NVGPUCTRPERM**：见 Step 0 权限说明。
3. **JIT 编译干扰 profile**：`--launch-skip` 调大（>200），确保跳过编译期 launch。
4. **看不到 humming kernel 名字**：humming JIT 出来的 kernel 名字带 hash 后缀，`-k regex:humming` 能匹配。如果还不行，先 nsys 跑一遍看实际 kernel 名再用更精准的 regex。
5. **GUI 装不上**：把 `.ncu-rep` 文件拷回本地 Mac 用 Nsight Compute Mac 客户端打开 —— 文件格式跨平台。
6. **Stream-K 时多个 kernel launch**：humming 可能 launch 多次同名 kernel（reduce 阶段、不同 expert）。用 `--launch-count N` 多抓几个或具体到 `--launch-skip K --launch-count 1`。

## 附录 B：一份"看 ncu 报告"的速查 cheat sheet

| 看到什么 | 想到什么 |
| --- | --- |
| `sm__pipe_tensor_op_hmma_*` > 80% | TC 用爽了，瓶颈不在 MMA |
| 上面那个 < 50%，但 SM throughput 高 | 在跑别的 pipeline（dequant、地址计算）—— transform_b 太重 |
| `dram__throughput` > 80% | memory-bound，看 L2 复用 / tile 大小 |
| `lts__t_sector_hit_rate` < 30% | L2 没复用，调 block swizzle / persistent kernel |
| `stall_barrier` 占 30%+ | producer/consumer 节奏不对 |
| Achieved occupancy 仅 12.5% 但 TC 90% | warp specialization 正常表现，**别动** |

---

## 附录 C：跟我说反馈用的模板

每个 Checkpoint 用这个格式发：

```
Step X 完成。

机器: H100 / A100 / ...
shape: M=4096 N=4096 K=4096

指标:
- sm__throughput: __%
- dram__throughput: __%
- HMMA / IMMA / QMMA active: __% / __% / __%
- L2 hit: __%
- Top-3 stall: __% / __% / __%

观察 / 困惑:
- ...
```

我会根据数字告诉你下一步该看什么、改什么。
