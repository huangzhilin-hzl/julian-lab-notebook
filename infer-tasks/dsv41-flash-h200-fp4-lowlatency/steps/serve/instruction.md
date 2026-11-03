Start DeepSeek-V4.1-Flash on 8x H200 using exactly `$SGLANG_LAUNCH_CMD` from
`task.toml`. This is the cookbook Low-Latency FP4 configuration with DSPARK
(block size 5), the pinned dev image, and the selected MoE runner. Leave the server running for the bench step.

1. Start `$SGLANG_LAUNCH_CMD` in the background, detached with all standard
   streams redirected (for example `setsid nohup bash -c "exec $SGLANG_LAUNCH_CMD"`).
   Write the full server log to `/app/results/server.log`.
2. Wait until `http://127.0.0.1:30000/health` returns 200.

Do not change server flags. Do not stop the server.
