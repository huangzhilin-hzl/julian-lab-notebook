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

## Not in the cookbook (left as TODO)

- Benchmark load: dataset, ISL/OSL, range ratio, num prompts, max concurrency, request rate.
  The page renders `Deployment` without a `benchmarks` prop, so this cell has no benchmark card and
  no reference numbers (no TTFT/TPOT/throughput to compare against).
- CPU / memory / storage, startup and agent timeouts.
- Image digest (`dev-dsv41` is a mutable preview tag).

## Contradictions inside the page (we follow the C3 cell)

1. Prose says the Low-Latency recipe turns DSpark on (C1 L74, L163: `--speculative-algorithm DSPARK
   --speculative-dspark-block-size 5`). The H200 low-latency cell has no DSpark flags; the B200, B300,
   GB300 and MI355X low-latency cells do. H200 low-latency vs high-throughput differ only by
   `--max-running-requests 256`.
2. "Do not override the backends" (C1 L92-96) vs. the H200 cell passing
   `--attention-backend dsv4 --moe-runner-backend flashinfer_mxfp4` explicitly.
3. Docker flags `--shm-size 32g --ipc=host` (C1 L29, L33) have no `task.toml` field; see NOTES.md.

## Related but not used

The page's Agentic section embeds InferenceX data for H200 (`inferencex-e2e/benchmarks/single_node/
srt-slurm-recipes/dsv41flash/sglang/h200-fp4-mtp/agentic.yaml`, sha `5bfd5559…`, run
`39318221321/attempts/1`, status verified). That is a different config (agentic replay with DSpark/MTP),
so none of its values are used here.
