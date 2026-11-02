<!-- TODO(instruction): restate the source in 1–3 sentences and paste the literal launch command. -->
Start the sglang server for the configuration in this task. A later step benchmarks it, so leave
it running.

1. Start the server in the background with exactly this command (also in `$SGLANG_LAUNCH_CMD`),
   detached so it keeps running after you finish (for example `setsid nohup ... < /dev/null &`):

   ```bash
   TODO_LAUNCH_CMD
   ```

   Write the full server log to `/app/results/server.log`.

2. Wait until `http://127.0.0.1:$SERVER_PORT/health` returns 200.

Do not change server flags. Do not stop the server.
