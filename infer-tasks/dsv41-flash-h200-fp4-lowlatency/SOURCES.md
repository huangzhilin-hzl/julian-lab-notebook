# Sources: dsv41-flash-h200-fp4-lowlatency

Fetched 2026-11-02 12:39 (UTC+8). Local copies: `/workspace/harbor-tasks/_sources/`.

| Key | Source | What it gave |
| --- | --- | --- |
| C1 | `https://lmsysorg.mintlify.app/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.md` (sha256 `01467c1c…915f`, 315 lines) | L17: no sglang release supports V4.1 Flash, use preview image. L23: NVIDIA image `lmsysorg/sglang:dev-dsv41` (H200/B200/B300/GB300). L26-35: `docker run --gpus all --shm-size 32g -p 30000:30000 -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN --ipc=host lmsysorg/sglang:dev-dsv41 sglang serve <args>`. L57: Low-Latency = fastest reply for a single user. |
| C2 | Deploy-panel `config` object embedded in the rendered page HTML (`_sources/cookbook_config_dsv41.js`) | `dockerImages.h200 = lmsysorg/sglang:dev-dsv41`; `modelNames["flash\|fp4"] = deepseek-ai/DeepSeek-V4.1-Flash`; `latencyPercentile = P50`. |
| C3 | Same config, `cells[]` entry `{hw:h200, variant:flash, quant:fp4, strategy:low-latency, nodes:single}` (line 3 of `_sources/cookbook_cells_dsv41.txt`) | `nnodes:1, verified:true`, no env, flags: `--trust-remote-code --model-path {{MODEL_NAME}} --tp 8 --ep-size 8 --mem-fraction-static 0.8 --attention-backend dsv4 --moe-runner-backend flashinfer_mxfp4 --cuda-graph-max-bs-decode 64 --reasoning-parser auto --tool-call-parser auto --host {{HOST_IP}} --port {{PORT}}` |
| C4 | Same config, `placeholders` | `HOST_IP` default `0.0.0.0`, `PORT` default `30000`. |
| B1 | `https://docs.sglang.io/cookbook/base/benchmarks/autoregressive_model_benchmark.md` L16 | Generic quick start only: `python -m sglang.bench_serving --backend sglang --port 30000 --dataset-name random --num-prompts 100`. Not tied to this recipe, so not used for the load. |

The URL hash (`#hw=h200&variant=flash&quant=fp4&strategy=low-latency&nodes=single`) only selects a
cell in the JS Deploy panel; the `.md` export does not contain the generated command, so C3 was read
from the panel's config in the HTML.

## Benchmark load (not in the cookbook)

The page renders `Deployment` without a `benchmarks` prop, so this cell has no benchmark card and no
reference numbers. The load applies InferenceX's fixed-seq-len rules to the cookbook server config.

InferenceX: `SemiAnalysisAI/InferenceX` @ `de8884cda8ba8b5c54576289d4f63098963eb91b` (2026-11-02
14:28 UTC+8). Paths relative to `inferencex-e2e/`.

| Key | File:line | Used for |
| --- | --- | --- |
| IX1 | `infx/bench/fixed_seq.py:41`; `infx/bench_serving/benchmark_serving.py:273, 332-350` | `--dataset-name random`: synthetic ids `(offset + i + j) % vocab_size`, re-encoded until exactly the target length -> `BENCH_DATASET = random-ids` (S7) |
| IX2 | `infx/matrix/generate.py:30` | `"8k1k": (8192, 1024)` -> `BENCH_INPUT_LEN` / `BENCH_OUTPUT_LEN`. All 84 `isl:` entries in `configs/{nvidia,amd}-master.yaml` are 8192/1024 |
| IX3 | `infx/srt_slurm/generate.py:31`; passed at `infx/bench/fixed_seq.py:44` | `RANDOM_RANGE_RATIO = "0.8"`; lengths uniform in `[int(len*0.8), len]` (`infx/bench_serving/benchmark_serving.py:265-269`) |
| IX4 | `infx/bench/fixed_seq.py:103` (single node), `:155` (sweep) | `num_prompts = 10 * conc` -> `BENCH_PROMPTS_PER_CONC = 10` |
| IX5 | `infx/bench/fixed_seq.py:50` | `--num-warmups 2 * conc` -> `BENCH_WARMUPS_PER_CONC = 2`. Warmup uses `input_requests[0]` at full output length under the concurrency limit (`benchmark_serving.py:537-571`) |
| IX6 | `infx/bench/fixed_seq.py:47-48` | `--request-rate inf`, `--ignore-eos` |
| IX7 | `infx/results/fixed_sequence.py:192-206` | `tput_per_gpu = total/num_gpus`, `output_tput_per_gpu`, `input_tput_per_gpu = (total - output)/num_gpus`, `intvty = 1000/tpot_ms` -> `check_result.py` |
| IX8 | `configs/nvidia-master.yaml:7660-7674` | `dsv41flash-fp4-h200-sglang-agentic-dspark`: `conc-list: [1, 2, 4, 8, 16]` (agentic scenario) |
| IX9 | `infx/bench/fixed_seq.py:121-166` | Multi-node `srt-sweep` loops `CONC_LIST` against one live server: precedent for one bench step looping concurrencies |
| IX10 | `docs/MODELS.md:129` | `dsv41flash` active scenarios: agentic coding only. No fixed-seq-len result exists to compare against |
| IX11 | `benchmarks/single_node/srt-slurm-recipes/dsv41flash/sglang/h200-fp4-mtp/agentic.yaml:23,27,32,41` | InferenceX's H200 server: `expert-parallel-size 1`, `moe-runner-backend marlin`, `mem-fraction-static 0.70`, `speculative-algorithm DSPARK`. Not used; differs from C3 |

zhilin's decisions (Z, 2026-11-02): ISL/OSL 8k/1k only; concurrency `1 2 4 8 16`; client = sglang
`bench_serving`, not InferenceX's `infx.bench_serving`.

sglang client: `sgl-project/sglang` @ `4ae54e94588380c45565eb3edee73ad6435d7d9a`,
`python/sglang/benchmark/serving.py` (`python -m sglang.bench_serving` re-exports it,
`python/sglang/bench_serving.py:10-23`).

| Key | Line | Fact |
| --- | --- | --- |
| S1 | 2301, 2314, 2317, 2328, 2379, 2394, 2412, 2418, 2424, 2466, 2478, 2490, 2554, 2697, 2708 | Flags exist: `--backend --host --port --dataset-name --model --num-prompts --random-input-len --random-output-len --random-range-ratio --request-rate --max-concurrency --output-file --seed --flush-cache --warmup-requests` |
| S2 | 1855-1908, 1961 | Result keys incl. `p90_/p99_{ttft,tpot,itl}_ms`, `p90_/p99_e2e_latency_ms`, `max_concurrency`, `completed`; one line appended per run. No p99.9 |
| S3 | 2424-2426; `datasets/common.py:56-65` | `--random-range-ratio` default 0.0; lengths uniform in `[max(int(len*ratio),1), len]`, same as IX3 |
| S4 | 2466-2470, 703, 2556 | `--request-rate` default inf; `ignore_eos = not --disable-ignore-eos` (on by default) |
| S5 | 1468-1501 | Warmup reuses `input_requests[0]` with `output_len = min(output_len, 32)`, under the concurrency semaphore. Differs from IX5 |
| S6 | 1517-1523, 1010-1031, 2697 | `--flush-cache` POSTs `/flush_cache` after warmup, before the measured run |
| S7 | `serving.py:2336-2337`; `datasets/__init__.py:25-26`; `datasets/random.py:39, 65-74, 84-92, 145-163` | `random-ids` is a valid `--dataset-name` and uses the same `RandomDataset` as `random` (same `--random-input-len/-output-len/-range-ratio` via S3); only `random` samples ShareGPT (HF download). `random-ids` builds ids `(offset + i + j) % vocab_size` like IX1, then decodes them to text once without IX1's re-encode loop |

Differences from InferenceX that make numbers non-comparable: server flags (C3 vs IX11), warmup
(S5 vs IX5), prompt length fix-up (S7 vs IX1), client backend (sglang `/generate` vs OpenAI completions,
`fixed_seq.py:37-38`), p99.9 not emitted (S2).

## Not in the cookbook (left as TODO)

- CPU / memory / storage, startup timeout, serve timeout. Bench timeout is a reasoned 3600 s budget
  (comment in `task.toml`), not measured.

## Cookbook differences (C3 plus C5 and Z1)

1. Both arms now follow the prose (C5) by adding DSPARK with block size 5. Prose says the Low-Latency recipe turns DSpark on (C1 L74, L163: `--speculative-algorithm DSPARK
   --speculative-dspark-block-size 5`). The H200 low-latency cell has no DSpark flags; the B200, B300,
   GB300 and MI355X low-latency cells do. H200 low-latency vs high-throughput differ only by
   `--max-running-requests 256`.
2. "Do not override the backends" (C1 L92-96) vs. the H200 cell passing
   `--attention-backend dsv4 --moe-runner-backend flashinfer_mxfp4` explicitly.
3. Docker flags `--shm-size 32g --ipc=host` (C1 L29, L33) have no `task.toml` field; see `../HARBOR_NOTES.md`.

## Related but not used

The page's Agentic section embeds InferenceX data for H200 (`inferencex-e2e/benchmarks/single_node/
srt-slurm-recipes/dsv41flash/sglang/h200-fp4-mtp/agentic.yaml`, sha `5bfd5559…`, run
`39318221321/attempts/1`, status verified). That is a different config (agentic replay with DSpark/MTP),
so none of its server values are used here; only InferenceX's
fixed-seq-len client rules are (above).

## Pinned A/B decisions (2026-11-02)

| Key | Source | Decision |
| --- | --- | --- |
| C5 | [sglang cookbook section 3.3 at 38d9e4a3](https://github.com/sgl-project/sglang/blob/38d9e4a3fe4a493b0ae468477a274e83defda81a/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx) | Both arms add `--speculative-algorithm DSPARK --speculative-dspark-block-size 5` from the Low-Latency recipe. |
| Z1 | zhilin's explicit experiment decisions | Both arms use the pinned latest `dev` image below instead of cookbook `dev-dsv41`. Off uses `flashinfer_mxfp4`; on uses `humming`. No `--quantization humming` or `SGLANG_HUMMING_*` overrides; use defaults. |

Image: `lmsysorg/sglang:dev@sha256:77ff63ee69fd7ebef20df3a6d036ee4b8a4ffe0734efb0028c3b7643386e3ccf` (index digest).
AMD64 manifest: `sha256:c6da252c6837bb062184b01c7aa2df7a2aadb1ed4944f236d7027174c73c45fa`.
Build: `sgl-project/sglang@38d9e4a3fe4a493b0ae468477a274e83defda81a`,
`nightly-dev-20261102-38d9e4a3`. Per the supplied provenance, it contains
[PR #46018](https://github.com/sgl-project/sglang/pull/46018), merge
`ca939407423088dffd33304e0ea7d092cba355d2`, `humming-kernels 0.1.18` and `flashinfer 0.7.2`.
These pins are supplied by zhilin; no image or GPU runtime validation was performed here.

The original cell's `verified:true` does not validate these modifications. Running V4.1 on
this dev image is unverified, as is Humming with TP8/EP8, CUDA graphs and DSPARK together.
Earlier S# client citations describe the previously inspected commit; compatibility with the
pinned build must be checked on the first run. The fixed-load policy is unchanged.
