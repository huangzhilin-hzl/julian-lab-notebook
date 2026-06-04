// Probe whether the current NCCL communicator supports NCCL GIN and whether
// NCCL can create a device communicator with GIN resources.
//
// Usage:
//   mpirun -np <world> ./gin_probe [--rail]
//
// --rail checks props.railedGinType and requests NCCL_GIN_CONNECTION_RAIL.
// Without --rail, the probe checks props.ginType and requests
// NCCL_GIN_CONNECTION_FULL.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#define MPICHECK(cmd) do {                                                     \
  int e = (cmd);                                                               \
  if (e != MPI_SUCCESS) {                                                      \
    std::fprintf(stderr, "MPI error %d at %s:%d\n", e, __FILE__, __LINE__);   \
    MPI_Abort(MPI_COMM_WORLD, e);                                              \
  }                                                                            \
} while (0)

#define CUDACHECK(cmd) do {                                                    \
  cudaError_t e = (cmd);                                                       \
  if (e != cudaSuccess) {                                                      \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n",                          \
                 cudaGetErrorString(e), __FILE__, __LINE__);                   \
    MPI_Abort(MPI_COMM_WORLD, 1);                                              \
  }                                                                            \
} while (0)

#define NCCLCHECK(cmd) do {                                                    \
  ncclResult_t r = (cmd);                                                      \
  if (r != ncclSuccess) {                                                      \
    std::fprintf(stderr, "NCCL error %s at %s:%d\n",                          \
                 ncclGetErrorString(r), __FILE__, __LINE__);                   \
    MPI_Abort(MPI_COMM_WORLD, 1);                                              \
  }                                                                            \
} while (0)

static int env_int(const char* name, int fallback) {
  const char* v = std::getenv(name);
  return v ? std::atoi(v) : fallback;
}

static int local_rank_from_env(int global_rank, int num_devices) {
  const char* names[] = {
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "MV2_COMM_WORLD_LOCAL_RANK",
    "SLURM_LOCALID",
    "LOCAL_RANK"
  };
  for (const char* name : names) {
    const char* v = std::getenv(name);
    if (v) return std::atoi(v);
  }
  return global_rank % num_devices;
}

int main(int argc, char** argv) {
  MPICHECK(MPI_Init(&argc, &argv));

  int rank = 0;
  int world = 0;
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world));

  bool rail = false;
  if (argc >= 2 && std::strcmp(argv[1], "--rail") == 0) {
    rail = true;
  }

  int num_devices = 0;
  CUDACHECK(cudaGetDeviceCount(&num_devices));
  if (num_devices <= 0) {
    std::fprintf(stderr, "[rank %d] no CUDA devices\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int local_rank = local_rank_from_env(rank, num_devices);
  CUDACHECK(cudaSetDevice(local_rank % num_devices));

  ncclUniqueId id;
  if (rank == 0) {
    NCCLCHECK(ncclGetUniqueId(&id));
  }
  MPICHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, world, id, rank));

  int nccl_version = 0;
  NCCLCHECK(ncclGetVersion(&nccl_version));

  ncclCommProperties props = NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCLCHECK(ncclCommQueryProperties(comm, &props));

  int selected_gin_type = rail ? props.railedGinType : props.ginType;
  int supported = selected_gin_type != NCCL_GIN_TYPE_NONE;

  int all_supported = 0;
  MPICHECK(MPI_Allreduce(&supported, &all_supported, 1, MPI_INT, MPI_MIN,
                         MPI_COMM_WORLD));

  if (rank == 0) {
    std::printf("NCCL version: %d.%d.%d\n",
                nccl_version / 10000,
                (nccl_version % 10000) / 100,
                nccl_version % 100);
    std::printf("mode: %s\n", rail ? "rail" : "full");
  }

  std::printf("[rank %d] ginType=%d railedGinType=%d selected=%d supported=%s\n",
              rank,
              static_cast<int>(props.ginType),
              static_cast<int>(props.railedGinType),
              selected_gin_type,
              supported ? "yes" : "no");

  if (!all_supported) {
    if (rank == 0) {
      std::printf("RESULT: GIN not supported for this NCCL communicator/config.\n");
    }
    NCCLCHECK(ncclCommDestroy(comm));
    MPICHECK(MPI_Finalize());
    return 2;
  }

  ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginContextCount = env_int("GIN_CONTEXTS", 1);
  reqs.ginExclusiveContexts = true;
  reqs.ginQueueDepth = env_int("GIN_QUEUE_DEPTH", 1024);
  reqs.ginTrafficClass = env_int("GIN_TRAFFIC_CLASS", 0);
  reqs.ginSignalCount = world + 4;
  reqs.ginConnectionType = rail ? NCCL_GIN_CONNECTION_RAIL
                                : NCCL_GIN_CONNECTION_FULL;

  ncclDevComm_t dev_comm;
  NCCLCHECK(ncclDevCommCreate(comm, &reqs, &dev_comm));

  std::printf("[rank %d] ncclDevCommCreate OK: lsaRank=%d lsaSize=%d "
              "ginContextCount=%d\n",
              rank, dev_comm.lsaRank, dev_comm.lsaSize,
              dev_comm.ginContextCount);

  NCCLCHECK(ncclDevCommDestroy(comm, &dev_comm));
  NCCLCHECK(ncclCommDestroy(comm));
  MPICHECK(MPI_Finalize());

  if (rank == 0) {
    std::printf("RESULT: GIN supported and device communicator created successfully.\n");
  }
  return 0;
}
