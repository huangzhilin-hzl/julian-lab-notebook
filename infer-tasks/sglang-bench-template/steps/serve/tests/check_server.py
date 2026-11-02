"""Check that the sglang server started by the "serve" step is up and answering.

Runs in the agent container (shared verifier): it must reach the server on 127.0.0.1.
Endpoints from sglang python/sglang/srt/entrypoints/http_server.py (main d952161):
/health, /server_info, and the deprecated alias /get_server_info.
"""

import json
import os
import sys
import urllib.request

PORT = os.environ.get("SERVER_PORT", "30000")
BASE = f"http://127.0.0.1:{PORT}"


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return r.status, r.read()


def main(info_out: str) -> None:
    try:
        status, _ = get("/health")
    except Exception as e:
        fail(f"/health unreachable: {e}")
    if status != 200:
        fail(f"/health returned {status}")

    info = None
    for path in ("/server_info", "/get_server_info"):  # older images only have the latter
        try:
            status, body = get(path)
        except Exception as e:
            print(f"{path}: {e}")
            continue
        if status == 200:
            info = json.loads(body)
            break
    if not isinstance(info, dict):
        fail("neither /server_info nor /get_server_info returned a JSON object")

    with open(info_out, "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"server healthy; server info saved to {info_out}")
    print("PASS")


if __name__ == "__main__":
    main(sys.argv[1])
