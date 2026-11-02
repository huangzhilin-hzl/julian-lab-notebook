# molou/dsv41-flash-h200-fp4-lowlatency

sglang cookbook DeepSeek-V4.1 Flash, H200 · FP4 · Low-Latency · single node, as a Harbor task.
Built from `../sglang-bench-template`. Sources and line refs: `SOURCES.md`.

Before running: fill every `TODO(load)` in `task.toml` (and keep `[verifier.env]` in sync), then

```bash
harbor lint .
harbor run -p . -a oracle -e <env>   # needs 8x H200; see NOTES.md for k8s
```
