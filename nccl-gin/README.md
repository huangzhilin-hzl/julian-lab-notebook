# NCCL GIN Probe

This directory keeps a small MPI/CUDA/NCCL probe for checking whether the
current GPU and network environment supports NCCL GIN.

## Files

| Relative Path | Content |
| --- | --- |
| `gin_probe.cu` | MPI probe that initializes an NCCL communicator, queries `ginType` or `railedGinType`, then tries `ncclDevCommCreate` with GIN resources. |

## What It Checks

`gin_probe.cu` validates two separate layers:

1. `ncclCommQueryProperties(comm, &props)` reports a usable GIN type.
2. `ncclDevCommCreate(comm, &reqs, &dev_comm)` can allocate a device
   communicator with GIN contexts, queue depth, traffic class, signal count, and
   the requested GIN connection type.

The two probe modes are:

| Probe Mode | NCCL Property | Connection Type |
| --- | --- | --- |
| default | `props.ginType` | `NCCL_GIN_CONNECTION_FULL` |
| `--rail` | `props.railedGinType` | `NCCL_GIN_CONNECTION_RAIL` |

## Build

Run this on the target GPU node or inside the target GPU container. Use the same
NCCL installation that the target workload will use.

```bash
cd /path/to/julian-lab-notebook

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export NCCL_HOME=${NCCL_HOME:-/path/to/nccl}

# OpenMPI:
nvcc -std=c++17 -O2 nccl-gin/gin_probe.cu -o /tmp/gin_probe \
  $(mpicxx --showme:compile) \
  -I"${NCCL_HOME}/include" \
  -L"${NCCL_HOME}/lib" -Wl,-rpath,"${NCCL_HOME}/lib" -lnccl \
  $(mpicxx --showme:link)
```

For MPICH-style wrappers, replace the two `--showme` substitutions with the
include and link flags shown by:

```bash
mpicxx -show
```

If the build fails on `ncclCommQueryProperties`, `ncclCommProperties`,
`NCCL_GIN_CONNECTION_RAIL`, or `ncclDevCommRequirements_t`, the NCCL headers are
too old for this probe.

## Single-Node Smoke Test

This confirms that the binary, MPI, CUDA, and NCCL runtime are wired together.
It does not prove cross-node RDMA GIN support.

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
mpirun -np 8 /tmp/gin_probe --rail
```

## Multi-Node Rail Mode

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export GIN_CONTEXTS=8
export GIN_QUEUE_DEPTH=1024
export GIN_TRAFFIC_CLASS=0

mpirun -np 16 \
  --map-by ppr:8:node \
  --host node0:8,node1:8 \
  /tmp/gin_probe --rail
```

Expected success signal:

```text
RESULT: GIN supported and device communicator created successfully.
```

## Multi-Node Full Mode

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
mpirun -np 16 \
  --map-by ppr:8:node \
  --host node0:8,node1:8 \
  /tmp/gin_probe
```

If `--rail` passes but full mode fails, the environment likely supports railed
GIN but not full GIN for the current communicator topology.

## Result Interpretation

| Output | Meaning |
| --- | --- |
| `selected=0 supported=no` | The selected property is `NCCL_GIN_TYPE_NONE`; this communicator/config does not expose GIN. |
| `GIN not supported for this NCCL communicator/config` | At least one rank reported no selected GIN support. |
| `ncclDevCommCreate OK` | NCCL could allocate the requested device communicator resources. |
| `ncclDevCommCreate` NCCL error | GIN was advertised, but requested contexts, queue depth, traffic class, signals, or connection type could not be created. |

Regular NCCL collectives can pass while this probe fails. This probe checks the
device-side GIN path, not ordinary host-launched NCCL collectives.
