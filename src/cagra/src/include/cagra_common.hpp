#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuvs/core/c_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

inline constexpr int64_t VECTOR_DIM = 1536;
inline constexpr int K = 10;
inline constexpr int DISPLAY_TOP_K = 10;
inline constexpr int GROUND_TRUTH_TOP_K = 100;
inline constexpr int QUERY_LIMIT = 1000;
inline constexpr int OFFLINE_QUERY_COUNT = 1000;
inline constexpr int ONLINE_QUERY_COUNT = 1;
inline constexpr int SEARCH_WARMUP_RUNS = 3;
inline constexpr int SEARCH_TIMED_RUNS = 10;
inline constexpr double MS_PER_SECOND = 1000.0;
inline constexpr int kDistanceThreads = 256;
inline constexpr int kMaxFinalK = 256;

inline const std::string METRIC = "sqeuclidean";
inline const std::string DISTRIBUTION_MODE = "sharded";
inline const std::string SEARCH_MODE = "load_balancer";
inline const std::string MERGE_MODE = "tree_merge";

std::string getenv_or(const char* name, const std::string& default_value);
int getenv_int_or(const char* name, int default_value);
std::optional<std::vector<int>> optional_int_list_from_env(const char* name);

void check_cuda(cudaError_t status, const std::string& where);
void check_cuda_device(cudaError_t status, const std::string& where, int device_id);
void check_cuvs(cuvsError_t status, const std::string& where);

int64_t checked_mul(int64_t lhs, int64_t rhs, const char* name);
size_t checked_bytes(int64_t count, size_t item_size, const char* name);

class DeviceGuard {
public:
  explicit DeviceGuard(int device_id);
  ~DeviceGuard();

private:
  int previous_device_ = 0;
};

template <typename T>
struct PinnedHostBuffer {
  T* ptr = nullptr;
  size_t count = 0;

  PinnedHostBuffer() = default;

  explicit PinnedHostBuffer(size_t n) { allocate(n); }

  PinnedHostBuffer(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

  PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
  {
    ptr = other.ptr;
    count = other.count;
    other.ptr = nullptr;
    other.count = 0;
  }

  PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept
  {
    if (this != &other) {
      release();
      ptr = other.ptr;
      count = other.count;
      other.ptr = nullptr;
      other.count = 0;
    }
    return *this;
  }

  ~PinnedHostBuffer() { release(); }

  void allocate(size_t n)
  {
    release();
    count = n;
    if (count == 0) return;

    void* raw = nullptr;
    check_cuvs(cuvsRMMHostAlloc(&raw, count * sizeof(T)), "cuvsRMMHostAlloc");
    ptr = static_cast<T*>(raw);
  }

  void release()
  {
    if (ptr != nullptr) {
      check_cuvs(cuvsRMMHostFree(ptr, count * sizeof(T)), "cuvsRMMHostFree");
      ptr = nullptr;
      count = 0;
    }
  }

  T* data() { return ptr; }
  const T* data() const { return ptr; }

  T& operator[](size_t i) { return ptr[i]; }
  const T& operator[](size_t i) const { return ptr[i]; }
};

template <typename T>
struct MatrixViewHost {
  T* data = nullptr;
  int64_t rows = 0;
  int64_t cols = 0;
};
