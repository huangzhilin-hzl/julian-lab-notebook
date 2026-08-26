#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

const char* result_name(CUresult result) {
  const char* name = "UNKNOWN";
  cuGetErrorName(result, &name);
  return name;
}

const char* result_string(CUresult result) {
  const char* message = "unknown CUDA driver error";
  cuGetErrorString(result, &message);
  return message;
}

[[noreturn]] void fail_cuda(const char* expression, CUresult result) {
  std::cerr << "FAIL api=" << expression << " result=" << result_name(result)
            << " code=" << static_cast<int>(result)
            << " message=\"" << result_string(result) << "\"" << std::endl;
  std::exit(10);
}

#define CUDA_DRV_CHECK(expression)                                  \
  do {                                                              \
    CUresult result__ = (expression);                               \
    if (result__ != CUDA_SUCCESS) fail_cuda(#expression, result__); \
  } while (0)

size_t align_up(size_t value, size_t alignment) {
  if (alignment == 0) return value;
  return ((value + alignment - 1) / alignment) * alignment;
}

long read_positive_env(const char* name, long default_value) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return default_value;
  char* end = nullptr;
  const long value = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || value <= 0) {
    std::cerr << "FAIL invalid " << name << "=\"" << raw << "\"" << std::endl;
    std::exit(2);
  }
  return value;
}

double elapsed_ms(Clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

}  // namespace

int main() {
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "BEGIN cuMulticastBindMem standalone probe" << std::endl;

  CUDA_DRV_CHECK(cuInit(0));

  int driver_version = 0;
  int visible_device_count = 0;
  CUDA_DRV_CHECK(cuDriverGetVersion(&driver_version));
  CUDA_DRV_CHECK(cuDeviceGetCount(&visible_device_count));

  const int requested_devices =
      static_cast<int>(read_positive_env("PROBE_NUM_DEVICES", visible_device_count));
  if (requested_devices > visible_device_count) {
    std::cerr << "FAIL requested_devices=" << requested_devices
              << " visible_devices=" << visible_device_count << std::endl;
    return 2;
  }
  if (requested_devices < 2) {
    std::cerr << "FAIL multicast probe needs at least 2 visible GPUs" << std::endl;
    return 2;
  }

  const size_t requested_bytes =
      static_cast<size_t>(read_positive_env("PROBE_BYTES_MIB", 16)) * 1024 * 1024;
  std::cout << "INFO driver_version=" << driver_version
            << " visible_devices=" << visible_device_count
            << " team_devices=" << requested_devices
            << " requested_bytes=" << requested_bytes << std::endl;

  std::vector<CUdevice> devices(requested_devices);
  std::vector<CUmemAllocationProp> allocation_props(requested_devices);
  size_t common_alignment = 1;

  for (int rank = 0; rank < requested_devices; ++rank) {
    CUDA_DRV_CHECK(cuDeviceGet(&devices[rank], rank));
    char name[256] = {};
    int multicast_supported = 0;
    CUDA_DRV_CHECK(cuDeviceGetName(name, sizeof(name), devices[rank]));
    CUDA_DRV_CHECK(cuDeviceGetAttribute(
        &multicast_supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, devices[rank]));

    auto& prop = allocation_props[rank];
    std::memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = devices[rank];
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

    size_t allocation_granularity = 0;
    CUDA_DRV_CHECK(cuMemGetAllocationGranularity(
        &allocation_granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    common_alignment = std::lcm(common_alignment, allocation_granularity);

    std::cout << "DEVICE rank=" << rank << " ordinal=" << devices[rank]
              << " name=\"" << name << "\""
              << " multicast_supported=" << multicast_supported
              << " allocation_granularity=" << allocation_granularity << std::endl;
    if (multicast_supported != 1) {
      std::cerr << "FAIL device rank=" << rank << " does not advertise multicast support"
                << std::endl;
      return 3;
    }
  }

  CUmulticastObjectProp multicast_prop{};
  multicast_prop.numDevices = requested_devices;
  multicast_prop.size = requested_bytes;
  multicast_prop.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
  multicast_prop.flags = 0;

  size_t multicast_minimum = 0;
  size_t multicast_recommended = 0;
  CUDA_DRV_CHECK(cuMulticastGetGranularity(
      &multicast_minimum, &multicast_prop, CU_MULTICAST_GRANULARITY_MINIMUM));
  CUDA_DRV_CHECK(cuMulticastGetGranularity(
      &multicast_recommended, &multicast_prop, CU_MULTICAST_GRANULARITY_RECOMMENDED));
  common_alignment = std::lcm(common_alignment, multicast_minimum);
  const size_t bind_bytes = align_up(requested_bytes, common_alignment);
  multicast_prop.size = bind_bytes;

  std::cout << "INFO multicast_minimum=" << multicast_minimum
            << " multicast_recommended=" << multicast_recommended
            << " common_alignment=" << common_alignment
            << " bind_bytes=" << bind_bytes << std::endl;

  CUmemGenericAllocationHandle multicast_handle = 0;
  std::vector<CUmemGenericAllocationHandle> memory_handles(requested_devices, 0);
  std::vector<bool> bound(requested_devices, false);

  {
    const auto begin = Clock::now();
    std::cout << "CALL api=cuMulticastCreate state=begin" << std::endl;
    CUDA_DRV_CHECK(cuMulticastCreate(&multicast_handle, &multicast_prop));
    std::cout << "CALL api=cuMulticastCreate state=success elapsed_ms="
              << elapsed_ms(begin) << std::endl;
  }

  for (int rank = 0; rank < requested_devices; ++rank) {
    const auto begin = Clock::now();
    std::cout << "CALL api=cuMulticastAddDevice rank=" << rank << " state=begin"
              << std::endl;
    CUDA_DRV_CHECK(cuMulticastAddDevice(multicast_handle, devices[rank]));
    std::cout << "CALL api=cuMulticastAddDevice rank=" << rank
              << " state=success elapsed_ms=" << elapsed_ms(begin) << std::endl;
  }

  for (int rank = 0; rank < requested_devices; ++rank) {
    const auto begin = Clock::now();
    std::cout << "CALL api=cuMemCreate rank=" << rank << " state=begin bytes="
              << bind_bytes << std::endl;
    CUDA_DRV_CHECK(cuMemCreate(
        &memory_handles[rank], bind_bytes, &allocation_props[rank], 0));
    std::cout << "CALL api=cuMemCreate rank=" << rank
              << " state=success elapsed_ms=" << elapsed_ms(begin) << std::endl;
  }

  for (int rank = 0; rank < requested_devices; ++rank) {
    const auto begin = Clock::now();
    std::cout << "CALL api=cuMulticastBindMem rank=" << rank
              << " state=begin bytes=" << bind_bytes << std::endl;
    const CUresult result = cuMulticastBindMem(
        multicast_handle, 0, memory_handles[rank], 0, bind_bytes, 0);
    if (result != CUDA_SUCCESS) fail_cuda("cuMulticastBindMem", result);
    bound[rank] = true;
    std::cout << "CALL api=cuMulticastBindMem rank=" << rank
              << " state=success elapsed_ms=" << elapsed_ms(begin) << std::endl;
  }

  std::cout << "PASS all cuMulticastBindMem calls completed" << std::endl;

  for (int rank = requested_devices - 1; rank >= 0; --rank) {
    if (bound[rank]) {
      CUDA_DRV_CHECK(cuMulticastUnbind(multicast_handle, devices[rank], 0, bind_bytes));
    }
    if (memory_handles[rank] != 0) CUDA_DRV_CHECK(cuMemRelease(memory_handles[rank]));
  }
  CUDA_DRV_CHECK(cuMemRelease(multicast_handle));
  std::cout << "END cleanup=success" << std::endl;
  return 0;
}
