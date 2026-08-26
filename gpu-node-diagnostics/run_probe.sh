#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${PROBE_BUILD_DIR:-/tmp/cu_multicast_bind_mem_probe}"
timeout_seconds="${PROBE_TIMEOUT_SEC:-60}"
binary="${build_dir}/cu_multicast_bind_mem_probe"

mkdir -p "${build_dir}"

echo "BUILD source=${script_dir}/cu_multicast_bind_mem_probe.cu binary=${binary}"
nvcc \
  -std=c++17 \
  -O2 \
  -lineinfo \
  "${script_dir}/cu_multicast_bind_mem_probe.cu" \
  -o "${binary}" \
  -lcuda

echo "RUN timeout_seconds=${timeout_seconds} num_devices=${PROBE_NUM_DEVICES:-all-visible} bytes_mib=${PROBE_BYTES_MIB:-16}"
set +e
timeout \
  --foreground \
  --signal=TERM \
  --kill-after=5s \
  "${timeout_seconds}s" \
  stdbuf -oL -eL "${binary}"
probe_status=$?
set -e

case "${probe_status}" in
  0)
    echo "RESULT PASS: cuMulticastBindMem completed on every selected GPU"
    ;;
  124|137)
    echo "RESULT HANG: probe exceeded ${timeout_seconds}s; inspect the last CALL state=begin line" >&2
    ;;
  *)
    echo "RESULT FAIL: probe exited with status ${probe_status}; inspect the CUDA error above" >&2
    ;;
esac

exit "${probe_status}"
