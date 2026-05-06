# Marlin

## Images

| Relative Path | Content |
| --- | --- |
| `images/fetch_to_shared_pipeline_corrected.png` | `gptq_marlin_repack_kernel` 中 `fetch_to_shared` 的修正版数据搬运示意图：明确 `b_q_weight` 的真实 shape 是 `[K/pack_factor, N]` 且元素是 `uint32/int32`，`cp_async4` 只是一次读取 4 个连续 `uint32`；区分 FP4 下 `has_perm=false` 的 `stage_size=32 int4` 和 `has_perm=true` 的 `stage_size=256 int4`，并补充 `perm` shared prefix 与 8-stage ring buffer 行为。 |
| `images/fetch_to_shared_pipeline_corrected.svg` | 上图的可编辑 SVG 源文件，用于后续继续修正变量名、公式和布局。 |
| `images/ptx_mma_m16n8k16_b_fragment_layout.jpg` | PTX ISA 文档中 `mma.m16n8k16` 的 matrix B fragment layout 截图，展示 `.f16` / `.bf16` 类型下 `%laneid` 持有的 B fragment 在 `16x8` B tile 中的 row/col 映射；用于对照 Marlin `repack_tile` 中 `tc_col`、`tc_row`、`tc_offsets` 和 `pack_idx` 的来源。 |
