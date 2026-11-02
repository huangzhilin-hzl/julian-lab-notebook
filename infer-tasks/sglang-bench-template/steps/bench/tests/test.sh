#!/bin/bash
# Reward 1 iff /app/results/bench.jsonl holds one complete, parseable sglang bench_serving result
# per expected concurrency, produced with the expected load. No performance thresholds: numbers are recorded, not judged.
mkdir -p /logs/verifier
if python3 /tests/check_result.py /app/results/bench.jsonl /logs/verifier/metrics.json; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
