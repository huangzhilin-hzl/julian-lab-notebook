#!/bin/bash
# Oracle, step "serve": start the server from $SGLANG_LAUNCH_CMD in the background and wait for /health.
# The server must outlive this script so the "bench" step can use it: setsid detaches it from this
# session, and all three std streams are redirected so the exec that runs this script can return.
# All knobs come from [environment.env] in task.toml; nothing is hardcoded here.
set -euo pipefail

OUT=/app/results
mkdir -p "$OUT"
: "${SGLANG_LAUNCH_CMD:?}" "${SERVER_PORT:?}"

{ echo "date: $(date -Is)"; echo "launch: $SGLANG_LAUNCH_CMD"; nvidia-smi -L || true; } > "$OUT/run_meta.txt"

setsid nohup bash -c "exec $SGLANG_LAUNCH_CMD" < /dev/null > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$OUT/server.pid"

python3 - "$SERVER_PID" <<'PY'
import os, sys, time, urllib.request
pid = int(sys.argv[1])
deadline = time.time() + int(os.environ.get("SERVER_READY_TIMEOUT_SEC", "2400"))
url = f"http://127.0.0.1:{os.environ['SERVER_PORT']}/health"
while time.time() < deadline:
    try:
        os.kill(pid, 0)
    except OSError:
        sys.exit("server process exited before becoming healthy; see /app/results/server.log")
    try:
        if urllib.request.urlopen(url, timeout=5).status == 200:
            print("server healthy"); sys.exit(0)
    except Exception:
        pass
    time.sleep(10)
sys.exit("server not healthy before SERVER_READY_TIMEOUT_SEC")
PY
