#!/bin/bash
# Reward 1 iff the server answers /health and /server_info. Shared verifier: runs in the agent
# container because the server only listens there. Gates the "bench" step (min_reward = 1.0).
mkdir -p /logs/verifier
if python3 /tests/check_server.py /logs/verifier/server_info.json; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
