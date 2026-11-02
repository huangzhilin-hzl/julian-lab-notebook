Start DeepSeek-V4.1-Flash serving with the sglang cookbook's verified **H200 · flash · FP4 ·
Low-Latency · single node** recipe (8× H200, image `lmsysorg/sglang:dev-dsv41`). A later step
benchmarks it, so leave it running.

1. Start the server in the background with exactly this command (also in `$SGLANG_LAUNCH_CMD`),
   detached so it keeps running after you finish (for example `setsid nohup ... < /dev/null &`):

   ```bash
   sglang serve \
     --trust-remote-code \
     --model-path deepseek-ai/DeepSeek-V4.1-Flash \
     --tp 8 \
     --ep-size 8 \
     --mem-fraction-static 0.8 \
     --attention-backend dsv4 \
     --moe-runner-backend flashinfer_mxfp4 \
     --cuda-graph-max-bs-decode 64 \
     --reasoning-parser auto \
     --tool-call-parser auto \
     --host 0.0.0.0 \
     --port 30000
   ```

   Write the full server log to `/app/results/server.log`.

2. Wait until `http://127.0.0.1:30000/health` returns 200.

Do not change server flags. Do not stop the server.
