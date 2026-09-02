export const megaMoeReportHtml = String.raw`
<section class="chapter-section article-block mega-chapter" id="section-1">
<div class="section-label">01 / CONFIGURATION</div>
<h2>Flash / Pro 配置</h2>
<h3 id="section-1-1"><span>1.1</span>生产配置</h3>
<div class="mega-table-wrap"><table class="mega-table">
<thead><tr><th class="">配置项</th><th class="num">Flash</th><th class="num">Pro</th></tr></thead><tbody>
<tr><td class="">hidden，H</td><td class="num">4096</td><td class="num">7168</td></tr>
<tr><td class="">intermediate hidden，I</td><td class="num">2048</td><td class="num">3072</td></tr>
<tr><td class="">global experts，E</td><td class="num">256</td><td class="num">384</td></tr>
<tr><td class="">ranks，R</td><td class="num">8</td><td class="num">8</td></tr>
<tr><td class="">local experts/rank，E_local</td><td class="num">32</td><td class="num">48</td></tr>
<tr><td class="">top-k，T</td><td class="num">6</td><td class="num">6</td></tr>
<tr><td class="">benchmark M/rank</td><td class="num">8、16、32、64、128、256、512、1024、2048、4096、8192</td><td class="num">同左</td></tr>
<tr><td class="">symmetric capacity，Mmax</td><td class="num">8192</td><td class="num">8192</td></tr>
<tr><td class="">shared experts，默认</td><td class="num">0</td><td class="num">0</td></tr>
<tr><td class="">activation</td><td class="num">SwiGLU</td><td class="num">SwiGLU</td></tr>
<tr><td class="">activation clamp，benchmark 默认</td><td class="num">10</td><td class="num">10</td></tr>
<tr><td class="">fast math，benchmark 默认</td><td class="num">开启</td><td class="num">开启</td></tr>
<tr><td class="">recipe</td><td class="num">(1, 1, 32)</td><td class="num">(1, 1, 32)</td></tr>
</tbody></table></div>
<p>M 是每个 source rank 的原始 token 数。cluster 内有效 route 总数最多为 R×M×T=48M；均衡路由时每个 destination rank 约接收 M×T=6M 条 route，单 expert 的均值分别为 6M/32 和 6M/48。实际 kernel work 不只取决于 route 数，还取决于每个 expert 的 M64 padding：</p>
<div class="mega-equation">
  <span>ROUTE BLOCKS / RANK</span>
  <math display="block"><mrow><msub><mi>N</mi><mtext>route-block</mtext></msub><mo>(</mo><mi>rank</mi><mo>)</mo><mo>=</mo><munder><mo>∑</mo><mi>e</mi></munder><mrow><mo>⌈</mo><mfrac><msub><mi>R</mi><mi>e</mi></msub><mn>64</mn></mfrac><mo>⌉</mo></mrow></mrow></math>
</div>

<p>其中 R_e 是本 rank 所属 expert e 从所有 source ranks 收到的有效 route 数。负一 expert ID 的 masked route 不进入该求和。</p>
</section>
<section class="chapter-section article-block mega-chapter" id="section-2">
<div class="section-label">02 / TENSOR CONTRACT</div>
<h2>Tensor 与前处理</h2>
<h3 id="section-2-1"><span>2.1</span>Activation、route 和输出</h3>
<div class="mega-table-wrap"><table class="mega-table">
<thead><tr><th class="">Tensor name</th><th class="">原始 dtype / shape</th><th class="">前处理后 dtype / shape</th><th class="">kernel 可见位置</th><th class="">说明</th></tr></thead><tbody>
<tr><td class="">x_fp8</td><td class="">FP8 E4M3 [M,H]</td><td class="">不变</td><td class="">symm.x[0:M,:]</td><td class="">per-token、K128 量化结果</td></tr>
<tr><td class="">x_scale</td><td class="">FP32 [M,H/128]</td><td class="">不变</td><td class="">symm.x_sf[0:M,:]</td><td class="">每 token 每 K128 一个 scale</td></tr>
<tr><td class="">topk_idx</td><td class="">int64 [M,6]</td><td class="">不变</td><td class="">symm.topk_idx</td><td class="">全局 expert ID；masked route 为 -1</td></tr>
<tr><td class="">topk_weights</td><td class="">FP32 [M,6]</td><td class="">不变</td><td class="">symm.topk_weights</td><td class="">当前 benchmark 直接使用 top-k score；masked route 为 0</td></tr>
<tr><td class="">output / y</td><td class="">BF16 [M,H]</td><td class="">无前处理</td><td class="">独立 CUDA tensor</td><td class="">combine 最终 TMA store 目标</td></tr>
<tr><td class="">cumulative_recv_stats</td><td class="">int32 [E_local]</td><td class="">无前处理</td><td class="">独立 CUDA tensor，可选</td><td class="">累加本地 expert 收到的 route 数</td></tr>
</tbody></table></div>
<p>Flash 的 x_scale shape 为 [M,32]；Pro 为 [M,56]。调用前，x_fp8、x_scale、topk_idx、topk_weights 被 copy 到本 rank 的 symmetric allocation；四次 copy 不计入 persistent kernel 时间。</p>
<h3 id="section-2-2"><span>2.2</span>DSV4 HuggingFace routed expert 权重</h3>
<p>这里只描述 DeepSeek-V4-Flash / Pro 官方 HuggingFace checkpoint。每个 expert 分别保存 w1（gate）、w3（up）和 w2（down）；shape 是单 expert 的 safetensors shape。</p>
<div class="mega-table-wrap"><table class="mega-table">
<thead><tr><th class="">HF tensor name</th><th class="">含义</th><th class="">safetensors dtype</th><th class="">Flash shape</th><th class="">Pro shape</th></tr></thead><tbody>
<tr><td class="">layers.{l}.ffn.experts.{e}.w1.weight</td><td class="">gate weight；K 维每字节打包两个 E2M1</td><td class="">I8</td><td class="">[2048,2048]</td><td class="">[3072,3584]</td></tr>
<tr><td class="">layers.{l}.ffn.experts.{e}.w1.scale</td><td class="">gate 的 K32 UE8M0 scale</td><td class="">F8_E8M0</td><td class="">[2048,128]</td><td class="">[3072,224]</td></tr>
<tr><td class="">layers.{l}.ffn.experts.{e}.w3.weight</td><td class="">up weight；K 维每字节打包两个 E2M1</td><td class="">I8</td><td class="">[2048,2048]</td><td class="">[3072,3584]</td></tr>
<tr><td class="">layers.{l}.ffn.experts.{e}.w3.scale</td><td class="">up 的 K32 UE8M0 scale</td><td class="">F8_E8M0</td><td class="">[2048,128]</td><td class="">[3072,224]</td></tr>
<tr><td class="">layers.{l}.ffn.experts.{e}.w2.weight</td><td class="">down weight；K 维每字节打包两个 E2M1</td><td class="">I8</td><td class="">[4096,1024]</td><td class="">[7168,1536]</td></tr>
<tr><td class="">layers.{l}.ffn.experts.{e}.w2.scale</td><td class="">down 的 K32 UE8M0 scale</td><td class="">F8_E8M0</td><td class="">[4096,64]</td><td class="">[7168,96]</td></tr>
</tbody></table></div>
<p>EP load 后，当前 rank 把 E_local 个 expert stack 起来，并沿输出维融合 w1 / w3。safetensors 的 I8 对应 torch.int8，F8_E8M0 对应 torch.float8_e8m0fnu：</p>
<div class="mega-table-wrap"><table class="mega-table">
<thead><tr><th class="">DeepGEMM 输入</th><th class="">来源</th><th class="">torch dtype</th><th class="">Flash shape</th><th class="">Pro shape</th></tr></thead><tbody>
<tr><td class="">l1_packed</td><td class="">stack(concat(w1.weight,w3.weight))</td><td class="">torch.int8</td><td class="">[32,4096,2048]</td><td class="">[48,6144,3584]</td></tr>
<tr><td class="">l1_ue8m0</td><td class="">stack(concat(w1.scale,w3.scale))</td><td class="">torch.float8_e8m0fnu</td><td class="">[32,4096,128]</td><td class="">[48,6144,224]</td></tr>
<tr><td class="">l2_packed</td><td class="">stack(w2.weight)</td><td class="">torch.int8</td><td class="">[32,4096,1024]</td><td class="">[48,7168,1536]</td></tr>
<tr><td class="">l2_ue8m0</td><td class="">stack(w2.scale)</td><td class="">torch.float8_e8m0fnu</td><td class="">[32,4096,64]</td><td class="">[48,7168,96]</td></tr>
</tbody></table></div>
<p>DSV4 官方 HF checkpoint 不含 weight_scale_2。下一节的 weight_scale_2 是 DeepGEMM 按 expert 计算出的 secondary scale，不是 checkpoint 输入。</p>
<h3 id="section-2-3"><span>2.3</span>Routed weight 前处理</h3>
<figure class="mega-weight-preprocess">
  <header>
    <small>MODEL-LOAD TIME</small>
    <strong>DSV4 MXFP4 routed weight preprocessing</strong>
    <span>per layer · per local expert</span>
  </header>
  <div class="mega-preprocess-grid">
    <section class="mega-prep-stage input">
      <div class="mega-prep-index">01</div>
      <h4>HF 输入</h4>
      <div class="mega-prep-tensor">
        <b>packed E2M1</b>
        <code>I8 · [E,N,K/2]</code>
        <small>每 byte 两个 nibble</small>
      </div>
      <div class="mega-prep-tensor">
        <b>UE8M0 scale code</b>
        <code>F8_E8M0 (1 byte) · [E,N,K/32]</code>
        <small>物理 bit pattern = raw_exp；数值语义 = 2^(raw_exp − 127)</small>
      </div>
    </section>
    <div class="mega-prep-arrow" aria-hidden="true">→</div>
    <section class="mega-prep-stage window">
      <div class="mega-prep-index">02</div>
      <h4>求每个 expert 的 base_exp</h4>
      <p class="mega-prep-scope">
        <b>raw_exp = UE8M0 exponent code</b>
        <span>每个 K32 group 一个；scale = 2^(raw_exp − 127)，raw_exp = 127 表示 scale = 1</span>
      </p>
      <p class="mega-exp-scan">在该 expert 的 N × K/32 个 raw_exp 上取最小值和最大值</p>
      <div class="mega-exp-range">
        <span>min_exp</span><i>≤</i><b>base_exp</b><i>≤</i><span>max_exp</span>
      </div>
      <code class="mega-prep-formula">base_exp = max(min_exp, max_exp − 11)</code>
      <div class="mega-exp-window">
        <b>为什么窗口宽度是 11</b>
        <code>E4M3 = E2M1 × 2^(relative_exp − 6)</code>
        <span><i>下界 relative_exp=1</i>0.5 → 2⁻⁶，刚好是 E4M3 最小 normal</span>
        <span><i>上界 relative_exp=12</i>6 → 384；再加 1 得到 768，超过 E4M3 最大值 448</span>
        <strong>relative_exp = raw_exp − base_exp + 1，因此 raw_exp − base_exp 最大为 11</strong>
      </div>
    </section>
    <div class="mega-prep-arrow" aria-hidden="true">→</div>
    <section class="mega-prep-stage rewrite">
      <div class="mega-prep-index">03</div>
      <h4>逐 K32 group 做数值补偿</h4>
      <div class="mega-prep-rule clipped">
        <em>raw_exp &lt; base_exp</em>
        <span><b>δ = base_exp − raw_exp</b>scale code 抬到 base_exp；packed E2M1 用 LUT 按 δ 降阶</span>
      </div>
      <div class="mega-prep-rule kept">
        <em>raw_exp ≥ base_exp</em>
        <span><b>δ = 0</b>scale code 与 packed E2M1 都保持不变</span>
      </div>
      <div class="mega-prep-derived">
        <code>relative_exp = max(raw_exp, base_exp) − base_exp + 1</code>
        <code>secondary = 2^(base_exp − 128)</code>
      </div>
      <div class="mega-prep-layout">
        <b>为 device 读取重排；数值不变</b>
        <span><i>sign bit</i>每 8 个 E2M1 仅把符号槽从 01234567 改为 04152637，magnitude 不动；decoder 因此少做两次 sign PRMT</span>
        <span><i>L1 rows</i>[gate 全部行 | up 全部行] → [gate8, up8, gate8, up8, …]；WGMMA accumulator 中相邻 8-column chunk 可直接配对做 SwiGLU</span>
        <span><i>scale bytes</i>逻辑 shape 仍为 [E,N,K/32]；payload 按 [E,K/128,N,4] 排列，使同一 K128 stage 的 4 个 K32 code 组成 uint32，并沿 N 连续</span>
      </div>
    </section>
    <div class="mega-prep-arrow" aria-hidden="true">→</div>
    <section class="mega-prep-stage output">
      <div class="mega-prep-index">04</div>
      <h4>Kernel triple</h4>
      <div class="mega-prep-result weight">
        <b>processed_e2m1</b>
        <code>int8 · [E,N,K/2]</code>
      </div>
      <div class="mega-prep-result scale">
        <b>relative_ue8m0</b>
        <code>uint8 · [E,N,K/32]</code>
      </div>
      <div class="mega-prep-result secondary">
        <b>weight_scale_2</b>
        <code>FP32 · [E]</code>
      </div>
    </section>
  </div>
  <figcaption><span>L1 / L2 各输出一组 triple。</span>Global memory 保留 compact E2M1；B producer 搬入 tile 后，math warpgroup 再在线展开为 E4M3。</figcaption>
</figure>
<p>两种 DSV4 shape 都满足 H≤8192，因此 scale 的公共 shape 不变，但底层 payload 从逻辑 [E,N,K/32] 变为物理 [E,K/128,N,4]：</p>
<div class="mega-table-wrap"><table class="mega-table">
<thead><tr><th class="">Tensor</th><th class="">Flash 物理 payload</th><th class="">Pro 物理 payload</th></tr></thead><tbody>
<tr><td class="">l1 relative_ue8m0</td><td class="">[32,32,4096,4]</td><td class="">[48,56,6144,4]</td></tr>
<tr><td class="">l2 relative_ue8m0</td><td class="">[32,16,4096,4]</td><td class="">[48,24,7168,4]</td></tr>
</tbody></table></div>
<h3 id="section-2-4"><span>2.4</span>Shared expert 输入</h3>
<p>默认生产 benchmark 不启用 shared expert。启用 S 个 shared experts 时，它们不使用 MXFP4：</p>
<div class="mega-table-wrap"><table class="mega-table">
<thead><tr><th class="">Tensor</th><th class="">dtype / shape</th><th class="">前处理</th></tr></thead><tbody>
<tr><td class="">shared_l1_weight</td><td class="">FP8 E4M3 [2SI,H]</td><td class="">只做 gate/up granularity-8 行交错</td></tr>
<tr><td class="">shared_l1_scale</td><td class="">FP32 [2SI/128,H/128]</td><td class="">保持自然 row-major</td></tr>
<tr><td class="">shared_l2_weight</td><td class="">FP8 E4M3 [H,SI]</td><td class="">不变</td></tr>
<tr><td class="">shared_l2_scale</td><td class="">FP32 [H/128,SI/128]</td><td class="">保持自然 row-major</td></tr>
</tbody></table></div>
<p>shared L1 activation 直接 alias symm.x；调用方需把 x_scale 同时排入 shared_l1_acts_sf 的 column-major view。</p>
</section>
<section class="chapter-section article-block mega-chapter" id="section-3">
<div class="section-label">03 / LAUNCH & TILING</div>
<h2>Kernel 配置与选型</h2>
<h3 id="section-3-1"><span>3.1</span>固定 launch contract</h3>
<div class="mega-launch-summary">
  <div><span>GRID</span><strong>156 CTAs</strong><small>2 × 78 H20 SMs</small></div>
  <div><span>BLOCK</span><strong>256 threads</strong><small>8 warps · 2 warpgroups</small></div>
  <div><span>ROLE SPLIT</span><strong>WG0 / WG1</strong><small>control + producer / math</small></div>
  <div><span>RESIDENCY</span><strong>2 CTA / SM</strong><small>launch_bounds(256, 2)</small></div>
</div>
<h3 id="section-3-2"><span>3.2</span>CTA 内 warp / warpgroup 分工</h3>
<div class="mega-table-wrap"><table class="mega-table">
<thead><tr><th class="">Warp</th><th class="">Warpgroup</th><th class="num">threads</th><th class="num">寄存器配额</th><th class="">角色</th></tr></thead><tbody>
<tr><td class="">warp 0</td><td class="">WG0</td><td class="num">32</td><td class="num">48/thread</td><td class="">dispatch：route 扫描、计数、publish；结束时参与 cleanup</td></tr>
<tr><td class="">warp 1</td><td class="">WG0</td><td class="num">32</td><td class="num">48/thread</td><td class="">dispatch：同上；与 warp 0 各有一个 token pull scratch</td></tr>
<tr><td class="">warp 2</td><td class="">WG0</td><td class="num">32</td><td class="num">48/thread</td><td class="">A+SFA producer；消费 scheduler mailbox</td></tr>
<tr><td class="">warp 3</td><td class="">WG0</td><td class="num">32</td><td class="num">48/thread</td><td class="">scheduler producer + B+SFB producer</td></tr>
<tr><td class="">warp 4–7</td><td class="">WG1</td><td class="num">128</td><td class="num">208/thread</td><td class="">一个 math warpgroup：MXFP4 decode、WGMMA、L1/L2 epilogue、scatter、combine</td></tr>
</tbody></table></div>
<p>WG0 的四个 warps 必须共同执行 setmaxnreg dealloc；WG1 执行 alloc。CTA 寄存器预算为 64×48 + 64×48 + 128×208 = 32,768 registers，恰为 SM register file 的一半，从而保留 2 CTA/SM。</p>
<h3 id="section-3-3"><span>3.3</span>GEMM tile 与 MMA shape</h3>
<div class="mega-table-wrap"><table class="mega-table">
<thead><tr><th class="">层级</th><th class="">Regular path</th><th class="">Small-M swap-AB path</th></tr></thead><tbody>
<tr><td class="">logical CTA task</td><td class="">M64×N128，遍历 K</td><td class="">仍是一个 M64×N128 task</td></tr>
<tr><td class="">pipeline macro-tile</td><td class="">M64×N128×K128</td><td class="">将 N128 weight 切成两个 N64 half</td></tr>
<tr><td class="">WGMMA</td><td class="">m64n128k32</td><td class="">m64n8/16/32/64k32</td></tr>
<tr><td class="">operand 方向</td><td class="">activation 为 M，weight output row 为 N</td><td class="">weight N64 half 映射到 WGMMA M；有效 token bucket 映射到 WGMMA N</td></tr>
<tr><td class="">K128 内指令数</td><td class="">4</td><td class="">每个 weight half 4 条；两 half 合成原 N128</td></tr>
<tr><td class="">bucket</td><td class="">不适用</td><td class="">N_SWAP = align_up(valid_m,8)，截到 8/16/32/64</td></tr>
<tr><td class="">accumulator</td><td class="">默认 FP32 WGMMA fragment；特定大 work 使用 packed FP16 WGMMA</td><td class="">FP32 WGMMA fragment，跨 K promotion 保存为 packed BF16</td></tr>
</tbody></table></div>
<h3 id="section-3-4"><span>3.4</span>TMA descriptor 与每 stage 搬运</h3>
<div class="mega-table-wrap"><table class="mega-table">
<thead><tr><th class="">数据</th><th class="">逻辑 tile → 搬运 box（inner × outer）</th><th class="num">字节/stage</th><th class="">搬运者</th></tr></thead><tbody>
<tr><td class="">A activation</td><td class="">M64×K128 → K128×M64，B128 swizzle</td><td class="num">8192 B</td><td class="">warp 2，TMA</td></tr>
<tr><td class="">L1 SFA</td><td class="">M64×1 → M64×1 FP32；K128 granularity</td><td class="num">256 B</td><td class="">warp 2，TMA</td></tr>
<tr><td class="">L2 SFA</td><td class="">M64×2 → 2 次 M64×1 FP32；K64 granularity</td><td class="num">512 B</td><td class="">warp 2，TMA</td></tr>
<tr><td class="">routed packed-B</td><td class="">N128×K128 → 64 B×N128，B64 swizzle</td><td class="num">8192 B</td><td class="">warp 3，TMA</td></tr>
<tr><td class="">routed SFB</td><td class="">N128×4 B；manual vector load</td><td class="num">512 B</td><td class="">warp 3，32 lanes 各 LDG uint4 + STS.128</td></tr>
<tr><td class="">shared FP8-B</td><td class="">N128×K128 → K128×N128，B128 swizzle</td><td class="num">16,384 B</td><td class="">warp 3，TMA</td></tr>
<tr><td class="">L1 output store</td><td class="">SMEM M64×N64 → 本 rank symm.l2_acts ring；TMA box N64×M64，no swizzle</td><td class="num">4096 B</td><td class="">math WG 的 warp 4 elected lane，TMA store</td></tr>
</tbody></table></div>
<p>routed SFB 不使用 TMA：自然布局下每个 N row 的 K128 scale word 只有 4 B，小于 Hopper TMA 连续维至少 16 B 的要求。预处理后的 [E,K/128,N,4] 物理布局让每 lane 可一次搬连续 16 B。</p>
<p>L1 output 是表中唯一的反向搬运：从 CTA SMEM 写入当前 rank symmetric allocation 内的 l2_acts FP8 ring，而不是最终 y。L2 producer 随后从同一 global 地址 TMA load；对应 l2_acts_sf 由 epilogue 用普通 global store 写入 scale ring。</p>
<h3 id="section-3-5"><span>3.5</span>CTA dynamic SMEM</h3>
<div class="mega-table-wrap"><table class="mega-table">
<thead><tr><th class="">SMEM region</th><th class="">Flash offset / size</th><th class="">Pro offset / size</th><th class="">说明</th></tr></thead><tbody>
<tr><td class="">expert count scratch</td><td class="">0 / 1024 B</td><td class="">0 / 2048 B</td><td class="">按 E_global×4 后向 1 KiB 对齐</td></tr>
<tr><td class="">2 dispatch send buffers</td><td class="">1024 / 8192 B</td><td class="">2048 / 14,336 B</td><td class="">每 dispatch warp 一个 H-byte token</td></tr>
<tr><td class="">C/D epilogue scratch</td><td class="">9216 / 16,384 B</td><td class="">16,384 / 16,384 B</td><td class="">L1 FP8 或 L2 BF16；combine 阶段 alias</td></tr>
<tr><td class="">A，3 stages</td><td class="">25,600 / 24,576 B</td><td class="">32,768 / 24,576 B</td><td class="">3×8192 B</td></tr>
<tr><td class="">expanded E4M3-B</td><td class="">50,176 / 32,768 B</td><td class="">57,344 / 16,384 B</td><td class="">Flash 双 buffer；Pro 第二 slot alias C/D</td></tr>
<tr><td class="">packed MXFP4-B，3 stages</td><td class="">82,944 / 24,576 B</td><td class="">73,728 / 24,576 B</td><td class="">3×8192 B</td></tr>
<tr><td class="">SFA，3 stages</td><td class="">107,520 / 1536 B</td><td class="">98,304 / 1536 B</td><td class="">3×512 B</td></tr>
<tr><td class="">SFB，3 stages</td><td class="">109,056 / 1536 B</td><td class="">99,840 / 1536 B</td><td class="">3×512 B</td></tr>
<tr><td class="">base barriers</td><td class="">110,592 / 128 B</td><td class="">101,376 / 128 B</td><td class="">dispatch、A/B stage、combine</td></tr>
<tr><td class="">scheduler barriers + 2 TaskInfo</td><td class="">110,720 / 96 B</td><td class="">101,504 / 96 B</td><td class="">4×8 B barrier + 2×32 B mailbox</td></tr>
<tr><td class="">launch dynamic SMEM 总计</td><td class="">110,816 B（108.219 KiB）</td><td class="">101,600 B（99.219 KiB）</td><td class="">无 shared expert</td></tr>
</tbody></table></div>
<p>有 shared expert 时增加两个 shared-B empty barrier，即每 CTA 再加 16 B。Flash 保留两个独立 expanded-B slots；Pro 为满足 2 CTA/SM，把第二 routed expanded-B slot alias 到 epilogue 才使用的 C/D 区，phase 切换时通过 barrier 保护生命周期。</p>
</section>
<section class="chapter-section article-block mega-chapter" id="section-4">
<div class="section-label">04 / SYMMETRIC MEMORY</div>
<h2>Symmetric Memory</h2>
<h3 id="section-4-1"><span>4.1</span>容量公式</h3>
<p>默认 8 ranks、Mmax=8192、top-k=6：</p>
<ul class="mega-list">
<li>最大 routed routes：8192×8×6 = 393,216；</li>
<li>logical M64 pool blocks：ceil(393,216/64)+E_local；</li>
<li>full metadata pool：align(393,216+191×E_local,384)；</li>
<li>physical activation ring 只按 scheduler 同时在途上界分配；</li>
<li>source metadata 和 source-token table 仍按 full logical capacity 分配；</li>
<li>combine 固定分配 top-k×Mmax×H×2 bytes。</li>
</ul>
<h3 id="section-4-2"><span>4.2</span>Live ring 推导</h3>
<div class="mega-table-wrap"><table class="mega-table">
<thead><tr><th class="">项目</th><th class="num">Flash</th><th class="num">Pro</th></tr></thead><tbody>
<tr><td class="">logical M64 pool blocks</td><td class="num">6176</td><td class="num">6192</td></tr>
<tr><td class="">L1 N128 tasks/M block</td><td class="num">32</td><td class="num">48</td></tr>
<tr><td class="">L2 N128 tasks/M block</td><td class="num">32</td><td class="num">56</td></tr>
<tr><td class="">worker CTAs</td><td class="num">156</td><td class="num">156</td></tr>
<tr><td class="">deadlock-safe L1 warmup waves</td><td class="num">2</td><td class="num">2</td></tr>
<tr><td class="">live blocks after warmup</td><td class="num">10</td><td class="num">7</td></tr>
<tr><td class="">L1/L2 frontier growth</td><td class="num">0</td><td class="num">885</td></tr>
<tr><td class="">global-wave margin</td><td class="num">5</td><td class="num">4</td></tr>
<tr><td class="">conservative live M64 blocks</td><td class="num">15</td><td class="num">896</td></tr>
<tr><td class="">ring tokens，向 128 对齐</td><td class="num">1024</td><td class="num">57,344</td></tr>
<tr><td class="">实际 M64 ring slots</td><td class="num">16</td><td class="num">896</td></tr>
<tr><td class="">scale ring tokens</td><td class="num">2048</td><td class="num">114,688</td></tr>
<tr><td class="">full metadata tokens</td><td class="num">399,360</td><td class="num">402,432</td></tr>
</tbody></table></div>
<p class="mega-callout">注意：Workspace 内四组 generation counter 沿用通用 kMinCandidateBlockM=8 ABI，因此实际分配的 counter entries 是 ring_tokens/8，即 Flash 128、Pro 7168；SM90 kernel 的 M64 ring modulo 只使用前 ring_tokens/64 个逻辑位置。</p>
<h3 id="section-4-3"><span>4.3</span>每 rank 的精确分配：Flash</h3>
<p>下表 offset 均相对本 rank symmetric allocation 起点；默认无 shared expert。</p>
<div class="mega-table-wrap"><table class="mega-table">
<thead><tr><th class="">Region</th><th class="">dtype / logical shape</th><th class="num">offset，B</th><th class="num">size，B</th><th class="num">可读大小</th></tr></thead><tbody>
<tr><td class="">workspace</td><td class="">internal</td><td class="num">0</td><td class="num">71,911,808</td><td class="num">68.580 MiB</td></tr>
<tr><td class="">x</td><td class="">E4M3 [8192,4096]</td><td class="num">71,911,808</td><td class="num">33,554,432</td><td class="num">32 MiB</td></tr>
<tr><td class="">x_sf</td><td class="">FP32 [8192,32]</td><td class="num">105,466,240</td><td class="num">1,048,576</td><td class="num">1 MiB</td></tr>
<tr><td class="">topk_idx</td><td class="">int64 [8192,6]</td><td class="num">106,514,816</td><td class="num">393,216</td><td class="num">384 KiB</td></tr>
<tr><td class="">topk_weights</td><td class="">FP32 [8192,6]</td><td class="num">106,908,032</td><td class="num">196,608</td><td class="num">192 KiB</td></tr>
<tr><td class="">l1_acts</td><td class="">E4M3 [1024,4096]</td><td class="num">107,104,640</td><td class="num">4,194,304</td><td class="num">4 MiB</td></tr>
<tr><td class="">l1_acts_sf</td><td class="">FP32 column-major [2048,32]</td><td class="num">111,298,944</td><td class="num">262,144</td><td class="num">256 KiB</td></tr>
<tr><td class="">l1_topk_weights</td><td class="">FP32 [1024]</td><td class="num">111,561,088</td><td class="num">4096</td><td class="num">4 KiB</td></tr>
<tr><td class="">l2_acts</td><td class="">E4M3 [1024,2048]</td><td class="num">111,565,184</td><td class="num">2,097,152</td><td class="num">2 MiB</td></tr>
<tr><td class="">l2_acts_sf</td><td class="">FP32 column-major [2048,32]</td><td class="num">113,662,336</td><td class="num">262,144</td><td class="num">256 KiB</td></tr>
<tr><td class="">combine</td><td class="">BF16 [6,8192,4096]</td><td class="num">113,924,480</td><td class="num">402,653,184</td><td class="num">384 MiB</td></tr>
<tr><td class="">总计/rank</td><td class="">int8 allocation</td><td class="num">0</td><td class="num">516,577,664</td><td class="num">492.647 MiB</td></tr>
</tbody></table></div>
<h3 id="section-4-4"><span>4.4</span>每 rank 的精确分配：Pro</h3>
<div class="mega-table-wrap"><table class="mega-table">
<thead><tr><th class="">Region</th><th class="">dtype / logical shape</th><th class="num">offset，B</th><th class="num">size，B</th><th class="num">可读大小</th></tr></thead><tbody>
<tr><td class="">workspace</td><td class="">internal</td><td class="num">0</td><td class="num">105,617,920</td><td class="num">100.725 MiB</td></tr>
<tr><td class="">x</td><td class="">E4M3 [8192,7168]</td><td class="num">105,617,920</td><td class="num">58,720,256</td><td class="num">56 MiB</td></tr>
<tr><td class="">x_sf</td><td class="">FP32 [8192,56]</td><td class="num">164,338,176</td><td class="num">1,835,008</td><td class="num">1.75 MiB</td></tr>
<tr><td class="">topk_idx</td><td class="">int64 [8192,6]</td><td class="num">166,173,184</td><td class="num">393,216</td><td class="num">384 KiB</td></tr>
<tr><td class="">topk_weights</td><td class="">FP32 [8192,6]</td><td class="num">166,566,400</td><td class="num">196,608</td><td class="num">192 KiB</td></tr>
<tr><td class="">l1_acts</td><td class="">E4M3 [57,344,7168]</td><td class="num">166,763,008</td><td class="num">411,041,792</td><td class="num">392 MiB</td></tr>
<tr><td class="">l1_acts_sf</td><td class="">FP32 column-major [114,688,56]</td><td class="num">577,804,800</td><td class="num">25,690,112</td><td class="num">24.5 MiB</td></tr>
<tr><td class="">l1_topk_weights</td><td class="">FP32 [57,344]</td><td class="num">603,494,912</td><td class="num">229,376</td><td class="num">224 KiB</td></tr>
<tr><td class="">l2_acts</td><td class="">E4M3 [57,344,3072]</td><td class="num">603,724,288</td><td class="num">176,160,768</td><td class="num">168 MiB</td></tr>
<tr><td class="">l2_acts_sf</td><td class="">FP32 column-major [114,688,48]</td><td class="num">779,885,056</td><td class="num">22,020,096</td><td class="num">21 MiB</td></tr>
<tr><td class="">combine</td><td class="">BF16 [6,8192,7168]</td><td class="num">801,905,152</td><td class="num">704,643,072</td><td class="num">672 MiB</td></tr>
<tr><td class="">总计/rank</td><td class="">int8 allocation</td><td class="num">0</td><td class="num">1,506,548,224</td><td class="num">1436.756 MiB</td></tr>
</tbody></table></div>
<p>Pro ring 明显大于 Flash，不是因为单 token 更宽这一项，而是 Pro 的 L2 N tasks=56 大于 L1 N tasks=48。交错调度中 L1 在 M-block frontier 上推进得更快，容量模型因此加入 885 个 block 的保守 frontier growth。</p>
<h3 id="section-4-5"><span>4.5</span>Workspace 内部拆分</h3>
<div class="mega-table-wrap"><table class="mega-table">
<thead><tr><th class="">Workspace 子区</th><th class="num">Flash</th><th class="num">Pro</th><th class="">用途</th></tr></thead><tbody>
<tr><td class="">barrier/schedule hot line</td><td class="num">128 B</td><td class="num">128 B</td><td class="">grid sync、NVLink tags、task counters；独占 128 B L2 line</td></tr>
<tr><td class="">expert send/recv counters</td><td class="num">4096 B</td><td class="num">6144 B</td><td class="">E_global×2×uint64</td></tr>
<tr><td class="">local expert recv sums</td><td class="num">256 B</td><td class="num">384 B</td><td class="">E_local×uint64</td></tr>
<tr><td class="">4 组 ring generation counters</td><td class="num">2048 B</td><td class="num">114,688 B</td><td class="">l1_full/l1_empty/l2_full/l2_empty</td></tr>
<tr><td class="">shared L2 full counters</td><td class="num">4096 B</td><td class="num">4096 B</td><td class="">Mmax/8 entries；即使 S=0 仍按 ABI 分配</td></tr>
<tr><td class="">source token-topk table</td><td class="num">67,108,864 B</td><td class="num">100,663,296 B</td><td class="">[E_local,R,R×Mmax] int32</td></tr>
<tr><td class="">full TokenSrcMetadata pool</td><td class="num">4,792,320 B</td><td class="num">4,829,184 B</td><td class="">每 route 12 B：source rank/token/topk slot</td></tr>
<tr><td class="">合计</td><td class="num">71,911,808 B</td><td class="num">105,617,920 B</td><td class="">已向 16 B 对齐</td></tr>
</tbody></table></div>
<h3 id="section-4-6"><span>4.6</span>Shared expert 的额外 symmetric 空间</h3>
<p>Mmax=8192 时，shared scale pool tokens = ceil(8192/8)×128 = 131,072。增加一个 shared expert 的增量：</p>
<div class="mega-table-wrap"><table class="mega-table">
<thead><tr><th class="">Region 增量</th><th class="num">Flash</th><th class="num">Pro</th></tr></thead><tbody>
<tr><td class="">shared_l1_acts_sf</td><td class="num">16 MiB</td><td class="num">28 MiB</td></tr>
<tr><td class="">shared_l2_acts</td><td class="num">16 MiB</td><td class="num">24 MiB</td></tr>
<tr><td class="">shared_l2_acts_sf</td><td class="num">16 MiB</td><td class="num">24 MiB</td></tr>
<tr><td class="">combine 额外 shared contribution slot</td><td class="num">64 MiB</td><td class="num">112 MiB</td></tr>
<tr><td class="">合计增量</td><td class="num">112 MiB</td><td class="num">188 MiB</td></tr>
</tbody></table></div>
<p>前三项随 shared expert 数 S 线性增长；combine 只增加一个 slot，因为所有 shared experts 被拼成一个宽 FFN，并把总结果写入同一个 shared contribution slot。</p>
</section>
<section class="chapter-section article-block mega-chapter" id="section-5">
<div class="section-label">05 / PERSISTENT PIPELINE</div>
<h2>Dispatch、MMA 与 epilogue</h2>
<figure class="mega-pipeline-sketch">
  <div class="sketch-title">One cooperative launch, two specialized warpgroups</div>
  <div class="mega-pipeline-flow">
    <div class="mega-flow-box blue"><small>SYMMETRIC INPUT</small><strong>x / scale / top-k</strong><em>remote rank</em></div>
    <b>→</b>
    <div class="mega-flow-box green"><small>WG0 · W0–1</small><strong>Dispatch + pull</strong><em>route → M64 ring</em></div>
    <b>→</b>
    <div class="mega-flow-box purple"><small>WG1 · W4–7</small><strong>L1 + SwiGLU</strong><em>MXFP4 decode + WGMMA</em></div>
    <b>→</b>
    <div class="mega-flow-box orange"><small>WG1 · W4–7</small><strong>L2 + scatter</strong><em>remote combine slots</em></div>
    <b>→</b>
    <div class="mega-flow-box blue"><small>WG1 · W4–7</small><strong>Combine</strong><em>BF16 output</em></div>
  </div>
  <div class="mega-warp-lanes">
    <span>WG0 / producer</span><div><i class="green">W0–1 · dispatch</i><i class="blue">W2 · A + SFA</i><i class="orange">W3 · scheduler + B + SFB</i></div>
    <span>WG1 / math</span><div><i class="purple wide">W4–7 · decode → WGMMA → promotion → epilogue → scatter / combine</i></div>
  </div>
  <figcaption><span>Figure 1.</span> CTA 内只有一个 math warpgroup。WG0 负责把 route、activation、scale 与 weight tile 持续送入三阶段 pipeline；WG1 在同一 launch 中完成两层 GEMM 和最终归并。</figcaption>
</figure>

<h3 id="section-5-1"><span>5.1</span>端到端状态链</h3>
<div class="mega-table-wrap"><table class="mega-table">
<thead><tr><th class="num">顺序</th><th class="">执行者</th><th class="">状态/数据流</th></tr></thead><tbody>
<tr><td class="num">1</td><td class="">host</td><td class="">权重完成一次性 preprocess；x/routes copy 到本 rank symm</td></tr>
<tr><td class="num">2</td><td class="">warp 0–1</td><td class="">扫本 rank top-k，统计 global expert counts，向 expert owner publish token_topk_idx</td></tr>
<tr><td class="num">3</td><td class="">warp 0–1</td><td class="">grid sync；block 0 汇总并向各 destination rank 发布 recv counts；NVLink barrier tag 1</td></tr>
<tr><td class="num">4</td><td class="">warp 0–1</td><td class="">destination 按 local expert 顺序，从各 source rank pull token/SF/route weight 到 L1 ring</td></tr>
<tr><td class="num">5</td><td class="">warp 3</td><td class="">生成 L1/L2 动态 TaskInfo，并在双 mailbox 中发布</td></tr>
<tr><td class="num">6</td><td class="">warp 2 / warp 3</td><td class="">分别 TMA A+SFA、packed-B；普通 LD/ST 搬 SFB</td></tr>
<tr><td class="num">7</td><td class="">warp 4–7</td><td class="">packed MXFP4 在线展开成 E4M3；WGMMA；按 scale 做跨 K promotion</td></tr>
<tr><td class="num">8</td><td class="">warp 4–7</td><td class="">L1：SwiGLU×top-k weight→K64 dynamic FP8→L2 ring</td></tr>
<tr><td class="num">9</td><td class="">warp 4–7</td><td class="">L2：BF16 cast→按 source metadata 跨 rank scatter 到 combine slot</td></tr>
<tr><td class="num">10</td><td class="">warp 4–7</td><td class="">NVLink barrier tag 2；执行 combine，TMA store y</td></tr>
<tr><td class="num">11</td><td class="">warp 0–1</td><td class="">与 epilogue rendezvous，清空 generation/task/expert counters；NVLink barrier tag 3</td></tr>
</tbody></table></div>
<h3 id="section-5-2"><span>5.2</span>Dispatch：route 计数与 remote publish</h3>
<p>top-k=6 时，kNumTokensPerWarp=32/6=5，一个 dispatch warp 每轮激活 30 lanes，覆盖 5 个 token×6 routes，余下 2 lanes 空闲：</p>
<ol class="mega-list">
<li>warp 0/1 以 grid-stride 扫 topk_idx；</li>
<li>对有效 expert ID 在 CTA shared expert_count 做 block-scope atomic add；</li>
<li>两个 dispatch warps 同步后，每个 global expert 用 64-bit atomic 一次性领取本 CTA 的 slot 区间；</li>
<li>再扫一次 top-k，把 4 B 的 token_topk_idx = token_idx×topk+slot 写到 destination rank 的 source-token table；</li>
<li>全 156 CTAs grid sync；</li>
<li>block 0 把本 source rank 的 per-expert count 写给 expert owner，并用 system-scope atomic 累加完成状态；</li>
<li>全 rank 经过 NVLink barrier tag 1 后，destination 才开始 pull。</li>
</ol>
<p>sparse completion 只改变本地 zero-count atomic；跨 rank publish 的 wire protocol和 completion quorum不变。</p>
<h3 id="section-5-3"><span>5.3</span>Destination pull：从哪里搬到哪里、每 route 搬多少</h3>
<div class="mega-table-wrap"><table class="mega-table">
<thead><tr><th class="">数据</th><th class="">源</th><th class="">中转</th><th class="">目的</th><th class="num">Flash/route</th><th class="num">Pro/route</th><th class="">方式</th></tr></thead><tbody>
<tr><td class="">activation token</td><td class="">remote rank symm.x</td><td class="">本 CTA 对应 dispatch warp 的 send buffer</td><td class="">local l1_acts ring</td><td class="num">4096 B</td><td class="num">7168 B</td><td class="">remote TMA load + local TMA store</td></tr>
<tr><td class="">activation scale</td><td class="">remote rank symm.x_sf</td><td class="">无</td><td class="">local l1_acts_sf ring，column-major</td><td class="num">128 B</td><td class="num">224 B</td><td class="">lanes 普通 mapped LD + local ST</td></tr>
<tr><td class="">route weight</td><td class="">remote rank symm.topk_weights</td><td class="">无</td><td class="">local l1_topk_weights ring</td><td class="num">4 B</td><td class="num">4 B</td><td class="">elected lane load/store</td></tr>
<tr><td class="">source metadata</td><td class="">各 source rank 已写入本 destination 的 source table</td><td class="">registers</td><td class="">full metadata pool</td><td class="num">12 B write</td><td class="num">12 B write</td><td class="">保存 rank、token、top-k slot</td></tr>
</tbody></table></div>
<p>每个 dispatch warp 有一个 H-byte scratch，因此 CTA 共有 2H bytes send buffer。具体过程：</p>
<ul class="mega-list">
<li>scheduler 的 expert recv counts 给出每个 local expert 的 [start,end)；</li>
<li>source-rank selector 优先在 Flash M8/M16/M32 的单 slot common case 用 ballot mask 直接定位，否则按各 source rank count 做通用 round-robin；</li>
<li>logical pool_token_idx 映射到 ring_block_idx = pool_block_idx mod ring_blocks；</li>
<li>覆盖 ring slot 前等待上一 generation 的 l1_empty；</li>
<li>remote TMA token load 完成后，再 TMA store 到 local l1 ring；</li>
<li>metadata 写 full pool，不随 ring wrap；</li>
<li>最后一条不足 M64 的 expert block 用补量使 l1_full 一次达到 64，A producer 只需等固定计数。</li>
</ul>
<h3 id="section-5-4"><span>5.4</span>Dynamic scheduler 与 mailbox</h3>
<p>TaskInfo 固定 32 B，字段为 phase、local_expert、m_block、n_cluster、pool_block、valid_m、shape_n、shape_k。</p>
<p>warp 3 同时是 scheduler producer 和 B loader：</p>
<ol class="mega-list">
<li>用 global task counter 抢下一个 task；</li>
<li>在双 stage CTA mailbox 发布同一个 TaskInfo；</li>
<li>warp 2 和 math WG 分别消费该 mailbox；</li>
<li>warp 3 随即对该 task 发起 B/SFB load。</li>
</ol>
<p>routed schedule 先发最小 deadlock-safe L1 warmup，再按 L2、补一个 L1 wave 的方式交错。L2 task 被发布前还检查其依赖的 L1 task 已全部领取。shared experts 存在时 phase 顺序为 SharedL1 → routed L1/L2 → SharedL2 → sentinel。</p>
<h3 id="section-5-5"><span>5.5</span>Producer warps 的三阶段流水</h3>
<p>warp 2 负责 A + SFA：</p>
<div class="mega-table-wrap"><table class="mega-table">
<thead><tr><th class="">phase</th><th class="num">A TMA</th><th class="num">SFA TMA</th><th class="num">full barrier expected bytes</th></tr></thead><tbody>
<tr><td class="">L1</td><td class="num">8192 B</td><td class="num">256 B</td><td class="num">8448 B</td></tr>
<tr><td class="">L2</td><td class="num">8192 B</td><td class="num">512 B</td><td class="num">8704 B</td></tr>
</tbody></table></div>
<p>warp 3 负责 B + SFB：</p>
<ul class="mega-list">
<li>packed-B：global expert weight → SMEM packed-B stage，TMA 8192 B；</li>
<li>SFB：global relative scale → SMEM SFB stage，普通 load/store 512 B；</li>
<li>barrier 的 transaction byte count只登记 TMA 的 8192 B；SFB 由同一 producer warp 在 arrive 前完成并用 warp ordering 发布；</li>
<li>shared expert B 不解码，直接 TMA 16,384 B E4M3 tile。</li>
</ul>
<p>A/B producers 共用同一个 full mbarrier。每个 routed stage 的合计 TMA transaction bytes 为：L1 8448+8192=16,640 B，L2 8704+8192=16,896 B；SFB 的 512 B 普通 load/store 不计入 transaction bytes。</p>
<p>三组 full/empty mbarrier 把 producer 与 math WG 解耦。Flash 的两个 expanded-B slots允许在当前 WGMMA group flight 中解码下一 stage；Pro 用 C/D alias 获得第二逻辑 slot。</p>
<h3 id="section-5-6"><span>5.6</span>Math warpgroup：MXFP4 decode</h3>
<p>regular 路径中，SFB stage 含 128 个 32-bit scale words；128 个 math threads 先各取一个 word，随后 paired-row decoder 用两 lane 协作一行并通过 shuffle 交换 scale。small-M complete-row decoder 才是一 lane 完整持有一个 N row：</p>
<ol class="mega-list">
<li>每个 N row 的 scale word 包含 4 个 K32 relative exponents；</li>
<li>每个 row、每个 K32 group 有 16 B packed E2M1，即 4 个 32-bit words、32 个 E2M1 values；</li>
<li>每个 32-bit packed word含 8 个 E2M1 值，lookup + sign mapping 生成两个 32-bit E4M3 words；</li>
<li>128 threads 合计把 8192 B packed-B 展开成 16,384 B E4M3 B tile；</li>
<li>fence_view_async_shared 后，128-thread warpgroup barrier 确保 WGMMA async proxy 可见；</li>
<li>expanded B 采用 B128 swizzle；packed B 的 B64 swizzle加 bank-permuted pair ownership，降低 LDS replay。</li>
</ol>
<p>small-M complete-row 路径让 lane 持有一个完整 packed row，并复用同一个 exponent lookup；regular paired-row 路径则在 16-row group 内重新排列相邻 word pair 的 lane ownership。预处理期的 sign reorder 与 coalesced SFB 正是为这两个 decoder 形态服务。</p>
<h3 id="section-5-7"><span>5.7</span>Math warpgroup：WGMMA 与 scale promotion</h3>
<p>regular routed task 对每个 K128 stage 执行：</p>
<ul class="mega-list">
<li>L1：一次 commit group 内 4×m64n128k32；</li>
<li>L2：K0..63 两条 WGMMA → 按第一组 K64 SFA promotion；K64..127 两条 WGMMA → 按第二组 K64 SFA promotion；</li>
<li>每个 thread 的 regular fragment 覆盖 64 个 accumulator elements；</li>
<li>relative UE8M0 已体现在展开后的 E4M3 B byte 中；</li>
<li>每个 promotion 的有效乘数为 activation_scale × expert_secondary × 64；乘 64 是 E2M1→E4M3 exponent bias 补偿；</li>
<li>为避免 UE8M0 低端下溢，代码优先把 64 乘到 secondary；只有高端可能溢出时才改乘 activation scale。</li>
</ul>
<p>scale 与 accumulator 无依赖，因此当前路径先发射异步 WGMMA，再读取 SFA、计算 combined scale，最后在第一次访问 fragment 前 wait。完成最后一次 SMEM 读后可先 release stage，让 producer refill；后续 promotion 只访问 registers。</p>
<p>fast-math regular 默认把跨 K partial 保存在 packed BF16；满足 workload gate 时，WGMMA 本身改为 packed FP16 accumulator，并用 half2 做 promotion。strict path 用 FP32 FMA 做当前 promotion，再按既定 packed state 契约存储。</p>
<p>small-M swap-AB 把每个 N64 weight half 当作 WGMMA M64，把 token bucket 当作 N8/N16/N32/N64；两个 half 完成后再 remap 回标准 M-token/N-output 语义。这样 tensor-core padding 从固定 M64 降到 align_up(valid_m,8)。</p>
<h3 id="section-5-8"><span>5.8</span>L1 epilogue：SwiGLU、top-k 融合和 dynamic FP8</h3>
<p>L1 的一个 N128 task 因 gate/up 交错最终产生 N64：</p>
<ol class="mega-list">
<li>accumulator 中 16 个 8-column chunks 按 gate/up 交替；</li>
<li>对 gate 做上界 clamp，对 up 做双边 clamp；</li>
<li>计算 SiLU(gate)×up；</li>
<li>立即乘该 route 的 top-k weight；因此 L2 输出已带路由权重，combine 只做加法；</li>
<li>每 token、每输出 K64 做 amax；</li>
<li>生成一个 FP32 scale，并把 64 个值量化成 E4M3；</li>
<li>64×64 E4M3 tile，即 4096 B，经 TMA store 到 l2_acts；</li>
<li>64 个 FP32 scales，即 256 B，普通 store 到 l2_acts_sf；</li>
<li>等 TMA store 完成后 release-add l2_full，并为该 L1 N task增加一次 l1_empty。</li>
</ol>
<p>regular lane 映射中 lane/4 给 row_idx，lane mod 4 给 col_idx；每 warp处理两组 8 行，4 warps覆盖完整 M64。partial M 仍 TMA store 完整 64 rows；padding 行可能是旧值，但 L2 scatter 对 valid_m 做硬 gate，因此不会写出。</p>
<h3 id="section-5-9"><span>5.9</span>L2 mainloop、ring 释放与 scatter 前处理</h3>
<p>L2 A producer 必须等同一 M block 的全部 L1 N tasks 已发布，即 l2_full 达到 L1_SHAPE_N/128。所有 L2 math threads完成 A 的最后一次读取后，即可增加 l2_empty，使下一 generation 的 L1 覆盖该 physical ring slot；这一 release 不需要等待 L2 scatter，因为输出已在 registers/CTA C-D scratch。</p>
<p>L2 final_accum 被转换成 BF16，并先写 64×128×2=16,384 B 的 C/D scratch。大 M specialization 对 C/D 地址应用 row/column swizzle；随后进入第 6 章的 remote scatter 和 combine。</p>
</section>
<section class="chapter-section article-block mega-chapter" id="section-6">
<div class="section-label">06 / REMOTE REDUCTION</div>
<h2>Combine 流程</h2>
<h3 id="section-6-1"><span>6.1</span>L2 remote scatter</h3>
<p>每个 L2 N128 task 对每条有效 route 写 128 个 BF16，即 256 B：</p>
<ul class="mega-list">
<li>4 个 math warps 各覆盖 16 rows；</li>
<li>一个 warp 内分成两个 16-lane group，每个 group 同时处理一行；</li>
<li>kNumRowsPerWarp=8 次循环，每次两个 row group，共 16 rows/warp；</li>
<li>每 lane负责 128/16=8 个 BF16，即连续 16 B；</li>
<li>16 lanes 合计一行 256 B；</li>
<li>通过 full TokenSrcMetadata 恢复 dst_rank、dst_token、dst_topk_slot；</li>
<li>目的地址为 remote_symm.combine[dst_topk_slot,dst_token,n:n+128]。</li>
</ul>
<p>shared L2 不查 route metadata，固定写本 rank、本 token、slot=top-k；多个 shared experts 的总输出占一个 slot。</p>
<h3 id="section-6-2"><span>6.2</span>Scatter completion</h3>
<p>所有 L2 tasks完成后，math WG 执行 NVLink barrier tag 2。该 barrier 是 combine 读取 remote scatter 结果的可见性边界；不能用 CTA 或 grid-local barrier 代替。之后 math WG 与 dispatch warps做一次 CTA rendezvous，允许 dispatch 清理 workspace；dispatch 清理完成后再执行 tag 3，保证下一次 launch 不读到旧 generation。</p>
<h3 id="section-6-3"><span>6.3</span>Chunk 选型</h3>
<p>chunk 数同时受 SMEM alias 容量和每 lane 128-register combine buffer 限制。</p>
<div class="mega-table-wrap"><table class="mega-table">
<thead><tr><th class="">项目</th><th class="num">Flash</th><th class="num">Pro</th></tr></thead><tbody>
<tr><td class="">combine chunks</td><td class="num">1</td><td class="num">2</td></tr>
<tr><td class="">elements/chunk</td><td class="num">4096</td><td class="num">3584</td></tr>
<tr><td class="">input bytes/contribution/chunk</td><td class="num">8192 B</td><td class="num">7168 B</td></tr>
<tr><td class="">output bytes/chunk</td><td class="num">8192 B</td><td class="num">7168 B</td></tr>
<tr><td class="">vectors/lane/chunk</td><td class="num">16</td><td class="num">14</td></tr>
<tr><td class="">每 vector</td><td class="num">8 BF16 = 16 B</td><td class="num">同左</td></tr>
<tr><td class="">双 input ping-pong + output 的 CTA alias</td><td class="num">98,304 B</td><td class="num">86,016 B</td></tr>
</tbody></table></div>
<h3 id="section-6-4"><span>6.4</span>每 token 的 reduction</h3>
<p>在 H20 上，每个 epilogue warp 的 token 序列起点为 blockIdx.x×4+epilogue_warp，步长为 156×4=624：</p>
<ol class="mega-list">
<li>lane 0..5 读取该 token 的六个 topk_idx，以 ballot mask 排除 masked slot；</li>
<li>若有 shared expert，lane 6 加入固定 shared slot；</li>
<li>elected lane 对第一个 contribution 发起 1D TMA load；</li>
<li>reduction 当前 buffer 时，提前对下一个 contribution 发起另一个 TMA load，形成双 buffer；</li>
<li>每 lane 每轮读取 uint4，即 8 个 BF16；</li>
<li>fast_math 使用 BF16x2 hadd，意味着每加入一个 contribution 都在 BF16 精度舍入；strict mode 保留 FP32 float2 accumulator；</li>
<li>reduction 结果写入每 warp 的 output SMEM 区；</li>
<li>elected lane 以 1D TMA store 写 y[token,chunk]。</li>
</ol>
<p>无 masked route、无 shared expert时，每 source token 的 combine 输入流量：</p>
<div class="mega-table-wrap"><table class="mega-table">
<thead><tr><th class="">模型</th><th class="num">contribution load</th><th class="num">final y store</th></tr></thead><tbody>
<tr><td class="">Flash</td><td class="num">6×4096×2 = 49,152 B</td><td class="num">8192 B</td></tr>
<tr><td class="">Pro</td><td class="num">6×7168×2 = 86,016 B</td><td class="num">14,336 B</td></tr>
</tbody></table></div>
<p>route weight 已在 L1 epilogue 融合，combine 不再加载 topk_weights。masked route减少一个 contribution TMA；shared expert 增加一个 H×2 B contribution。</p>
</section>
<section class="chapter-section article-block mega-chapter" id="section-7">
<div class="section-label">07 / OPTIMIZE & FALSIFY</div>
<h2>优化与收益</h2>
<h3 id="section-7-1"><span>7.1</span>性能数据口径</h3>
<ul class="mega-list">
<li>8×H20，一 GPU 一进程；</li>
<li>cold L2；</li>
<li>max_rank_median_us，越低越好；</li>
<li>M≤128：50 observations；M≥256：3 observations；</li>
<li>每 observation 20 launches；</li>
<li>profiler只做机制归因，不替代分布式 timing。</li>
</ul>
<h3 id="section-7-2"><span>7.2</span>当前实现保留的优化</h3>
<div class="mega-table-wrap"><table class="mega-table">
<thead><tr><th class="">子系统</th><th class="">当前手段</th><th class="">生效点/selector</th><th class="">主要作用</th></tr></thead><tbody>
<tr><td class="">kernel 架构</td><td class="">dispatch+L1+SwiGLU+L2+scatter+combine 单 persistent kernel</td><td class="">全部</td><td class="">消除中间 kernel launch 和 L1/L2 全局 handoff</td></tr>
<tr><td class="">occupancy</td><td class="">156 CTA，2 CTA/SM，role-based setmaxnreg</td><td class="">全部</td><td class="">提供足够 math WG 并行度；保持 zero-spill 资源边界</td></tr>
<tr><td class="">small-M</td><td class="">swap-AB + N8/16/32/64 bucket</td><td class="">Flash/Pro M≤128，routed-only</td><td class="">把 token padding 从 M64 降到 align8</td></tr>
<tr><td class="">packed frontend</td><td class="">compact E2M1+relative UE8M0 常驻，tile 内在线 E4M3 decode</td><td class="">routed</td><td class="">避免全局 E4M3 双驻留和额外 HBM 流量</td></tr>
<tr><td class="">weight scale</td><td class="">[E,K/128,N,4] coalesced layout，lane-wise uint4 publish</td><td class="">H≤8192</td><td class="">4 轮 strided scalar load/store 合为一次 16 B load/store</td></tr>
<tr><td class="">decoder</td><td class="">paired-word、complete-row lookup reuse、PRMT index/exponent</td><td class="">shape-specific</td><td class="">降低 bit、integer、shuffle 和 lookup 指令</td></tr>
<tr><td class="">SMEM</td><td class="">B64 packed swizzle + bank-permuted LDS.64；L2 C/D swizzle</td><td class="">selector 见 7.3</td><td class="">降低 shared bank conflict/replay</td></tr>
<tr><td class="">overlap</td><td class="">下一 packed stage decode 与当前 WGMMA flight 重叠；scale precompute 放在 wait 前</td><td class="">两个 DSV4 shape</td><td class="">隐藏 decode、SFA 和 promotion latency</td></tr>
<tr><td class="">ring</td><td class="">live ring + generation counter；relaxed poll 后 acquire confirm</td><td class="">Flash M8192 等</td><td class="">减少 full intermediate pool 与轮询开销</td></tr>
<tr><td class="">dispatch</td><td class="">single-slot source lookup；Pro expert-count CTA cache</td><td class="">Flash M8/16/32；Pro M8/M512</td><td class="">缩短 source mapping 与重复 counter load</td></tr>
<tr><td class="">epilogue</td><td class="">packed BF16 direct epilogue、inactive SwiGLU predicate、partial-tile TMA</td><td class="">selected small-M</td><td class="">减少 unpack/remap、无效 SFU 和 scalar store</td></tr>
<tr><td class="">accumulator</td><td class="">workload-gated packed-FP16 WGMMA + half2 promotion</td><td class="">当前意图为 Flash M1024+、Pro M512+</td><td class="">降低 conversion/accumulator 指令</td></tr>
</tbody></table></div>
<h3 id="section-7-3"><span>7.3</span>JIT 选型矩阵</h3>
<div class="mega-table-wrap"><table class="mega-table">
<thead><tr><th class="">选项</th><th class="">Flash</th><th class="">Pro</th><th class="">目的</th></tr></thead><tbody>
<tr><td class="">small-M swap-AB</td><td class="">routed-only，M≤128</td><td class="">routed-only，M≤128</td><td class="">避免 M64 padded tensor-core work</td></tr>
<tr><td class="">local swap bucket</td><td class="">8/16/32/64</td><td class="">8/16/32/64</td><td class="">控制 WGMMA N 与 epilogue 展开</td></tr>
<tr><td class="">cross-rank epilogue storage bound</td><td class="">多 rank 固定 64</td><td class="">多 rank 固定 64</td><td class="">单 local expert 可从 8 ranks 聚成 M64</td></tr>
<tr><td class="">scale-path overlap</td><td class="">全部标准点</td><td class="">全部标准点</td><td class="">SFA load/scale promotion 藏在 WGMMA wait 下</td></tr>
<tr><td class="">PRMT exponent/index</td><td class="">M&lt;1024</td><td class="">M=8/16/32/64</td><td class="">减少 bit/地址指令</td></tr>
<tr><td class="">incremental WGMMA descriptor</td><td class="">M&gt;4096，即标准矩阵 M8192</td><td class="">关闭</td><td class="">减少长循环 descriptor 重建</td></tr>
<tr><td class="">packed-BF16 swap epilogue</td><td class="">M16/M32</td><td class="">M16/M32</td><td class="">避免 packed partial 先展开再重排</td></tr>
<tr><td class="">packed-weight bank permutation</td><td class="">全部 M</td><td class="">M8/M32/M64/M128，及 regular M≥512</td><td class="">消除 LDS.64 bank conflict</td></tr>
<tr><td class="">L2 C/D swizzle</td><td class="">M≥1024</td><td class="">M≥1024</td><td class="">降低大 M L2 epilogue shared replay</td></tr>
<tr><td class="">sparse dispatch completion</td><td class="">M32/M1024</td><td class="">关闭</td><td class="">省略本地 zero-count atomic，wire protocol 不变</td></tr>
<tr><td class="">expert-count CTA cache</td><td class="">关闭</td><td class="">M8/M512</td><td class="">三个 scheduler consumer 共享一次完成计数快照</td></tr>
</tbody></table></div>
<h3 id="section-7-4"><span>7.4</span>有正式对照的增益记录</h3>
<p>下表都是相对各迭代紧邻 control 的增量，不应把各百分比线性相加。</p>
<div class="mega-table-wrap"><table class="mega-table mega-performance-table">
<thead><tr><th class="">优化</th><th class="">正式受影响点</th><th class="num">记录增益</th><th class="">profiler/机制证据</th></tr></thead><tbody>
<tr><td class="">保留 2 CTA/SM</td><td class="">Flash M8/M16</td><td class="num">仅改为 1 CTA/SM 回退 +46.07%/+46.34%</td><td class="">当前两 resident CTA 是并行度下限</td></tr>
<tr><td class="">初版 swap-AB</td><td class="">Flash M8/M16/M32</td><td class="num">-8.01%/-9.83%/-9.17%，GM -9.01%</td><td class="">避免大量 padded M64 WGMMA</td></tr>
<tr><td class="">初版 swap-AB</td><td class="">Pro M8/M16/M32/M64</td><td class="num">-11.66%/-13.68%/-12.98%/-11.35%，GM -12.42%</td><td class="">同上</td></tr>
<tr><td class="">Flash vectorized SFB publish</td><td class="">M8/M32/M64</td><td class="num">双侧 -11.91/-12.67%、-8.56/-11.09%、-14.59/-15.93%</td><td class="">4 次 scalar publish 合为 uint4</td></tr>
<tr><td class="">Pro coalesced scale+uint4 publish</td><td class="">M8/16/32/64/128</td><td class="num">双侧约 -1.98% 到 -4.06%</td><td class="">global-load sectors -87.12%</td></tr>
<tr><td class="">bank-permuted paired load</td><td class="">Flash M16</td><td class="num">-3.99% 到 -5.51%</td><td class="">shared-load conflict -99.80%</td></tr>
<tr><td class="">bank-permuted paired load</td><td class="">Pro selected small buckets</td><td class="num">-1.23% 到 -3.30%</td><td class="">Pro M8 shared-load conflict -99.90%</td></tr>
<tr><td class="">PRMT pair index/exponent</td><td class="">6 个 selected points</td><td class="num">GM -3.89% 到 -4.26%</td><td class="">Flash M8 动态指令约 -10%</td></tr>
<tr><td class="">mature swap 扩到 Flash M128</td><td class="">Flash M128</td><td class="num">-12.67%/-12.41%</td><td class="">经过 scale、decoder、bank 优化后 crossover 已改变</td></tr>
<tr><td class="">complete-row decoder</td><td class="">Flash M16</td><td class="num">-2.94% 到 -7.11%</td><td class="">bit -9.50%、integer -5.18%、inter-thread -15.04%</td></tr>
<tr><td class="">complete-row decoder</td><td class="">Pro M8–M128</td><td class="num">-1.60% 到 -7.65%</td><td class="">selected bucket 中复用 row lookup</td></tr>
<tr><td class="">inactive SwiGLU predicate</td><td class="">Flash M16</td><td class="num">正向 -2.42% 到 -5.03%；反向 -4.28% 到 -6.17%</td><td class="">XU/SFU instructions -85.97%</td></tr>
<tr><td class="">vector/pre-wait activation scale</td><td class="">Pro M8</td><td class="num">-1.99% 到 -3.30%</td><td class="">barrier samples -4.9% 到 -6.5%</td></tr>
<tr><td class="">vector/pre-wait activation scale</td><td class="">Flash M16</td><td class="num">-1.01% 到 -3.31%</td><td class="">动态指令约 -3.15%</td></tr>
<tr><td class="">去 padded-token scale selects</td><td class="">Flash M16</td><td class="num">四个正式对照 -0.65% 到 -3.73%，三次均值 -2.15%</td><td class="">warp/thread instructions -6.88%/-6.59%</td></tr>
<tr><td class="">packed-FP16 WGMMA，当前 HEAD</td><td class="">commit 声明 Flash M1024+、Pro M512+</td><td class="num">初筛 1.3%–2.2%、0.6%–1.6%</td><td class="">尚无 R203 同协议完整 22 点双顺序结果</td></tr>
</tbody></table></div>
<h3 id="section-7-5"><span>7.5</span>从初始版本到 R203</h3>
<p>初始 candidate 相对 PR383：</p>
<div class="mega-table-wrap"><table class="mega-table mega-performance-table">
<thead><tr><th class="">切片</th><th class="num">初始几何平均 gap</th></tr></thead><tbody>
<tr><td class="">全部 22 点</td><td class="num">+20.36%</td></tr>
<tr><td class="">M≤128</td><td class="num">+45.93%</td></tr>
<tr><td class="">M≥256</td><td class="num">+2.51%</td></tr>
</tbody></table></div>
<p>R203 最终矩阵中，gap = R203 / mean(PR383_before,PR383_after) - 1：</p>
<div class="mega-table-wrap"><table class="mega-table mega-performance-table">
<thead><tr><th class="num">M</th><th class="num">Flash R203，µs</th><th class="num">Flash gap</th><th class="num">Pro R203，µs</th><th class="num">Pro gap</th></tr></thead><tbody>
<tr><td class="num">8</td><td class="num">293.9795</td><td class="num">-1.96%</td><td class="num">718.5870</td><td class="num">+1.21%</td></tr>
<tr><td class="num">16</td><td class="num">315.4985</td><td class="num">+1.47%</td><td class="num">922.0145</td><td class="num">-8.48%</td></tr>
<tr><td class="num">32</td><td class="num">324.3600</td><td class="num">-2.68%</td><td class="num">996.3670</td><td class="num">-9.99%</td></tr>
<tr><td class="num">64</td><td class="num">360.9235</td><td class="num">-1.10%</td><td class="num">1023.0000</td><td class="num">-12.30%</td></tr>
<tr><td class="num">128</td><td class="num">412.4505</td><td class="num">-7.42%</td><td class="num">1180.0000</td><td class="num">-6.84%</td></tr>
<tr><td class="num">256</td><td class="num">548.9800</td><td class="num">+8.55%</td><td class="num">1603.0000</td><td class="num">-1.43%</td></tr>
<tr><td class="num">512</td><td class="num">918.2910</td><td class="num">+0.06%</td><td class="num">2543.0000</td><td class="num">+5.93%</td></tr>
<tr><td class="num">1024</td><td class="num">1513.0000</td><td class="num">-1.38%</td><td class="num">3902.0000</td><td class="num">-2.86%</td></tr>
<tr><td class="num">2048</td><td class="num">2740.0000</td><td class="num">+0.26%</td><td class="num">6869.0000</td><td class="num">-2.16%</td></tr>
<tr><td class="num">4096</td><td class="num">5109.0000</td><td class="num">+0.26%</td><td class="num">12,971.0000</td><td class="num">+0.39%</td></tr>
<tr><td class="num">8192</td><td class="num">9944.0000</td><td class="num">+1.24%</td><td class="num">25,288.0000</td><td class="num">+0.98%</td></tr>
</tbody></table></div>
<p>汇总：</p>
<ul class="mega-list">
<li>Flash 11 点几何平均：-0.31%；</li>
<li>Pro 11 点几何平均：-3.38%；</li>
<li>全部 22 点几何平均：-1.86%；</li>
<li>12/22 点同时优于两侧 PR383 control；</li>
<li>从初始 +20.36% 到 R203 -1.86%，aggregate gap 收敛 22.22 个百分点；这是阶段性总体演进，不是单项优化增益。</li>
</ul>
<p>Flash M16 对执行顺序仍较敏感；可稳定归因的是 R203 相对 R191 的 0.65%–3.73%，不能仅凭单一矩阵顺序声称稳定超越 PR383。</p>
<h3 id="section-7-6"><span>7.6</span>当前 HEAD 的 packed-FP16 gate 审计</h3>
<p>当前 gate 使用 num_max_tokens_per_rank，而不是本次实际 num_tokens：</p>
<div class="mega-equation">
  <span>CAPACITY BLOCK ESTIMATE</span>
  <math display="block"><mrow><msub><mi>B</mi><mtext>est</mtext></msub><mo>=</mo><mrow><mo>⌈</mo><mfrac><mrow><msub><mi>M</mi><mtext>max</mtext></msub><mo>×</mo><mi>top-k</mi></mrow><mn>64</mn></mfrac><mo>⌉</mo></mrow></mrow></math>
</div>

<div class="mega-equation">
  <span>PACKED-FP16 WORKLOAD</span>
  <math display="block"><mrow><mi>work</mi><mo>=</mo><msub><mi>B</mi><mtext>est</mtext></msub><mo>×</mo><mo>[</mo><mfrac><mrow><mn>2</mn><mi>I</mi></mrow><mn>128</mn></mfrac><mo>×</mo><mfrac><mi>H</mi><mn>128</mn></mfrac><mo>+</mo><mfrac><mi>H</mi><mn>128</mn></mfrac><mo>×</mo><mfrac><mi>I</mi><mn>128</mn></mfrac><mo>]</mo></mrow></math>
</div>

<div class="mega-equation">
  <span>ENABLE GATE</span>
  <math display="block"><mrow><mi>enable</mi><mo>=</mo><mi>fast_math</mi><mo>∧</mo><mi>regular</mi><mo>∧</mo><mi>routed-only</mi><mo>∧</mo><mi>work</mi><mo>≥</mo><msup><mn>2</mn><mn>17</mn></msup></mrow></math>
</div>

<p>如果 B_est 使用实际 M，则阈值首次命中 Flash M1024、Pro M512，与 commit 描述一致。但 benchmark 默认 Mmax=8192，于是所有 regular、routed-only、fast-math 点都会由 capacity estimate 命中，包括 Flash M256/M512 和 Pro M256。</p>
<p>因此当前实现的正式结论是：</p>
<ul class="mega-list">
<li>packed-FP16 机制已经在当前代码中；</li>
<li>commit 记录的收益是 preliminary same-source screen；</li>
<li>“按实际 workload 排除回退点”的意图与当前 capacity gate 语义尚未闭环；</li>
<li>在修正 gate 或按 capacity 语义完成全矩阵复测前，不把 1.3%–2.2% / 0.6%–1.6% 叠加到 R203 最终矩阵。</li>
</ul>
<h3 id="section-7-7"><span>7.7</span>已验证不应采用的方向</h3>
<div class="mega-table-wrap"><table class="mega-table">
<thead><tr><th class="">方向</th><th class="">结果</th><th class="">设计结论</th></tr></thead><tbody>
<tr><td class="">1 CTA/SM</td><td class="">Flash M8/M16 回退约 46%</td><td class="">不能只缩 grid；若做 1 CTA 必须同时重构 register/SMEM/pipeline</td></tr>
<tr><td class="">全局预解码 E4M3 双驻留</td><td class="">指令 -57.59%，但 DRAM read +85.88%；8-rank 关键点回退</td><td class="">保留 compact MXFP4 global traffic，在 tile 内优化 decoder</td></tr>
<tr><td class="">超过 128-register lifetime 的 decoder state</td><td class="">出现 stack/LDL/STL 或性能回退</td><td class="">persistent 热循环中 zero-spill 是硬门槛</td></tr>
<tr><td class="">省略跨 rank completion step</td><td class="">混合 JIT specialization 时存在协议风险</td><td class="">本地 sparse 优化不能改变全 rank wire protocol</td></tr>
<tr><td class="">未经 phase 证明的 SMEM alias</td><td class="">曾出现 producer 覆盖 epilogue scratch/NaN</td><td class="">alias 必须由 producer prefetch 与 consumer lifetime共同证明</td></tr>
</tbody></table></div>
</section>
<section class="chapter-section article-block mega-chapter" id="section-8">
<div class="section-label">08 / PROFILE &amp; ROOFLINE</div>
<h2>NSYS 与 Roofline（待补）</h2>
<p class="mega-empty-state">本章统一评估当前 HEAD 的执行时间线、硬件利用率与 Roofline。数据尚未按 8×H20、cold-L2 协议重新采集。</p>
<div class="mega-table-wrap"><table class="mega-table">
<thead><tr><th class="">评估项</th><th class="">指标</th><th class="">状态</th></tr></thead><tbody>
<tr><td class="">NSYS timeline</td><td class="">dispatch / L1 / L2 / scatter / combine 时长与重叠</td><td class="">待采集</td></tr>
<tr><td class="">同步等待</td><td class="">NVLink barrier 1/2/3、grid barrier、block tail</td><td class="">待采集</td></tr>
<tr><td class="">计算利用率</td><td class="">achieved FLOPs、WGMMA active、tensor-core throughput</td><td class="">待采集</td></tr>
<tr><td class="">带宽利用率</td><td class="">HBM read/write、SMEM throughput、NVLink traffic</td><td class="">待采集</td></tr>
<tr><td class="">算术强度</td><td class="">实际 FLOPs / 实测数据流量</td><td class="">待计算</td></tr>
<tr><td class="">Roofline</td><td class="">Flash / Pro 各 M 点相对 H20 compute 与 bandwidth ceiling 的位置</td><td class="">待评估</td></tr>
<tr><td class="">机制对照</td><td class="">packed-FP16 on/off matched profile</td><td class="">待 gate 语义闭环</td></tr>
</tbody></table></div>
</section>
`;
