// run_cagra_optimized_cpp.cpp
//
// A detailed C++ equivalent of src/cagra/run_cagra_optimized.py.
//
// Notes:
// - Uses the cuVS C API from C++ because it mirrors the Python multi-GPU binding.
// - Uses cuvsRMMHostAlloc/cuvsRMMHostFree for pinned host query/output buffers.
// - Uses cuvsMultiGpuResourcesSetMemoryPool to reduce device allocation churn.
// - Loads the Python data cache .npy files:
//     data_cache/default_data_*/dataset_ids.npy
//     data_cache/default_data_*/dataset.npy
//     data_cache/default_data_*/queries.npy
//   This avoids reimplementing the Python Parquet cache builder.
// - Loads precomputed ground truth from openai_large_5m/neighbors.parquet.
// - Exact rerank uses a resident multi-GPU reranker. There is no
//   host-side exact-L2 fallback path in this file.

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/mg_cagra.h>
#include <dlpack/dlpack.h>

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
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

constexpr int64_t VECTOR_DIM = 1536;
constexpr int K = 10;
constexpr int DISPLAY_TOP_K = 10;
constexpr int GROUND_TRUTH_TOP_K = 100;
constexpr int QUERY_LIMIT = 1000;
constexpr int OFFLINE_QUERY_COUNT = 1000;
constexpr int ONLINE_QUERY_COUNT = 1;
constexpr int SEARCH_WARMUP_RUNS = 3;
constexpr int SEARCH_TIMED_RUNS = 10;
constexpr double MS_PER_SECOND = 1000.0;
constexpr int kDistanceThreads = 256;
constexpr int kMaxFinalK = 256;
constexpr int kConvertThreads = 256;
constexpr size_t kHalfUploadChunkBytes = static_cast<size_t>(64) * 1024 * 1024;

const std::string METRIC = "sqeuclidean";
const std::string DISTRIBUTION_MODE = "sharded";
const std::string SEARCH_MODE = "load_balancer";
const std::string MERGE_MODE = "tree_merge";

std::string getenv_or(const char* name, const std::string& default_value)
{
  const char* value = std::getenv(name);
  if (value == nullptr || std::string(value).empty()) return default_value;
  return std::string(value);
}

int getenv_int_or(const char* name, int default_value)
{
  const char* value = std::getenv(name);
  if (value == nullptr || std::string(value).empty()) return default_value;
  return std::stoi(value);
}

std::optional<std::vector<int>> optional_int_list_from_env(const char* name)
{
  const char* raw = std::getenv(name);
  if (raw == nullptr || std::string(raw).empty()) return std::nullopt;

  std::vector<int> values;
  std::stringstream ss(raw);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (!token.empty()) values.push_back(std::stoi(token));
  }
  return values;
}

void check_cuda(cudaError_t status, const std::string& where)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(where + ": " + cudaGetErrorString(status));
  }
}

void check_cuda_device(cudaError_t status, const std::string& where, int device_id)
{
  if (status != cudaSuccess) {
    std::ostringstream message;
    message << where << " on CUDA device " << device_id << ": "
            << cudaGetErrorString(status);

    if (status == cudaErrorMemoryAllocation) {
      message << ". MultiGpuExactReranker could not allocate resident dataset shards "
              << "or rerank buffers; reduce visible devices/index memory pressure, "
              << "reduce rerank batch size or index memory pressure.";
    }

    throw std::runtime_error(message.str());
  }
}

void check_cuvs(cuvsError_t status, const std::string& where)
{
  if (status != CUVS_SUCCESS) {
    const char* text = cuvsGetLastErrorText();
    throw std::runtime_error(where + ": " + (text ? text : "unknown cuVS error"));
  }
}

template <typename T>
T arrow_value_or_throw(arrow::Result<T> result, const std::string& where)
{
  if (!result.ok()) {
    throw std::runtime_error(where + ": " + result.status().ToString());
  }
  return std::move(result).ValueOrDie();
}

void arrow_check(const arrow::Status& status, const std::string& where)
{
  if (!status.ok()) {
    throw std::runtime_error(where + ": " + status.ToString());
  }
}

int64_t checked_mul(int64_t lhs, int64_t rhs, const char* name)
{
  if (lhs < 0 || rhs < 0) {
    throw std::invalid_argument(std::string(name) + " has negative extent");
  }
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs) {
    throw std::overflow_error(std::string(name) + " size overflow");
  }
  return lhs * rhs;
}

size_t checked_bytes(int64_t count, size_t item_size, const char* name)
{
  if (count < 0) {
    throw std::invalid_argument(std::string(name) + " has negative extent");
  }
  const auto ucount = static_cast<size_t>(count);
  if (item_size != 0 && ucount > std::numeric_limits<size_t>::max() / item_size) {
    throw std::overflow_error(std::string(name) + " byte size overflow");
  }
  return ucount * item_size;
}

class DeviceGuard {
public:
  explicit DeviceGuard(int device_id)
  {
    check_cuda(cudaGetDevice(&previous_device_), "cudaGetDevice");
    check_cuda(cudaSetDevice(device_id), "cudaSetDevice");
  }

  ~DeviceGuard() { cudaSetDevice(previous_device_); }

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

  size_t size() const { return static_cast<size_t>(rows * cols); }

  T* row(int64_t r) { return data + r * cols; }
  const T* row(int64_t r) const { return data + r * cols; }

  T& operator()(int64_t r, int64_t c) { return data[r * cols + c]; }
  const T& operator()(int64_t r, int64_t c) const { return data[r * cols + c]; }
};

struct NpyHeader {
  std::string descr;
  bool fortran_order = false;
  std::vector<int64_t> shape;
};

std::string trim(const std::string& s)
{
  const auto begin = s.find_first_not_of(" \t\n\r");
  if (begin == std::string::npos) return "";
  const auto end = s.find_last_not_of(" \t\n\r");
  return s.substr(begin, end - begin + 1);
}

NpyHeader parse_npy_header_text(const std::string& header)
{
  NpyHeader out;

  auto find_quoted_value = [&](const std::string& key) {
    const auto key_pos = header.find(key);
    if (key_pos == std::string::npos) {
      throw std::runtime_error("npy header missing key: " + key);
    }
    const auto colon = header.find(':', key_pos);
    const auto quote1 = header.find_first_of("'\"", colon);
    const auto quote2 = header.find_first_of("'\"", quote1 + 1);
    return header.substr(quote1 + 1, quote2 - quote1 - 1);
  };

  out.descr = find_quoted_value("descr");

  const auto fortran_pos = header.find("fortran_order");
  if (fortran_pos == std::string::npos) {
    throw std::runtime_error("npy header missing fortran_order");
  }
  const auto colon = header.find(':', fortran_pos);
  const auto comma = header.find(',', colon);
  const auto value = trim(header.substr(colon + 1, comma - colon - 1));
  out.fortran_order = (value == "True");

  const auto shape_pos = header.find("shape");
  if (shape_pos == std::string::npos) {
    throw std::runtime_error("npy header missing shape");
  }
  const auto open = header.find('(', shape_pos);
  const auto close = header.find(')', open);
  std::string shape_text = header.substr(open + 1, close - open - 1);

  std::stringstream ss(shape_text);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token = trim(token);
    if (!token.empty()) out.shape.push_back(std::stoll(token));
  }

  return out;
}

NpyHeader read_npy_header(std::ifstream& in)
{
  char magic[6];
  in.read(magic, 6);
  if (!in || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
    throw std::runtime_error("not an npy file");
  }

  uint8_t major = 0;
  uint8_t minor = 0;
  in.read(reinterpret_cast<char*>(&major), 1);
  in.read(reinterpret_cast<char*>(&minor), 1);

  uint32_t header_len = 0;
  if (major == 1) {
    uint16_t h16 = 0;
    in.read(reinterpret_cast<char*>(&h16), 2);
    header_len = h16;
  } else if (major == 2 || major == 3) {
    uint32_t h32 = 0;
    in.read(reinterpret_cast<char*>(&h32), 4);
    header_len = h32;
  } else {
    throw std::runtime_error("unsupported npy version");
  }

  std::string header(header_len, '\0');
  in.read(header.data(), header_len);
  if (!in) throw std::runtime_error("failed to read npy header");

  return parse_npy_header_text(header);
}

template <typename T>
std::vector<T> read_npy(const fs::path& path, const std::string& expected_descr, std::vector<int64_t>* shape_out)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("failed to open " + path.string());

  NpyHeader header = read_npy_header(in);

  if (header.descr != expected_descr) {
    throw std::runtime_error(
      "unexpected dtype in " + path.string() + ": got " + header.descr +
      ", expected " + expected_descr);
  }
  if (header.fortran_order) {
    throw std::runtime_error("Fortran-order npy not supported: " + path.string());
  }

  int64_t count = 1;
  for (int64_t dim : header.shape) count *= dim;

  std::vector<T> data(static_cast<size_t>(count));
  in.read(reinterpret_cast<char*>(data.data()), count * sizeof(T));
  if (!in) throw std::runtime_error("failed to read npy payload: " + path.string());

  if (shape_out != nullptr) *shape_out = header.shape;
  return data;
}

fs::path project_root()
{
  return fs::current_path();
}

fs::path data_cache_root()
{
  return fs::path(getenv_or("CUVS_BENCH_DATA_CACHE_DIR", (project_root() / "data_cache").string()));
}

fs::path default_data_cache_dir()
{
  fs::path root = data_cache_root();

  if (fs::exists(root / "dataset.npy") &&
      fs::exists(root / "dataset_ids.npy") &&
      fs::exists(root / "queries.npy")) {
    return root;
  }

  fs::path best;
  fs::file_time_type best_time{};
  bool found = false;

  for (const auto& entry : fs::directory_iterator(root)) {
    if (!entry.is_directory()) continue;

    const fs::path dir = entry.path();
    if (dir.filename().string().find("default_data_") != 0) continue;
    if (!fs::exists(dir / "dataset.npy")) continue;
    if (!fs::exists(dir / "dataset_ids.npy")) continue;
    if (!fs::exists(dir / "queries.npy")) continue;

    auto t = fs::last_write_time(dir);
    if (!found || t > best_time) {
      best = dir;
      best_time = t;
      found = true;
    }
  }

  if (!found) {
    throw std::runtime_error(
      "Could not find data cache. Run the Python loader once or set "
      "CUVS_BENCH_DATA_CACHE_DIR to a directory containing dataset.npy, "
      "dataset_ids.npy, and queries.npy.");
  }

  return best;
}

struct LoadedData {
  std::vector<int64_t> dataset_ids;
  std::vector<float> dataset;
  std::vector<float> queries;
  int64_t n_rows = 0;
  int64_t dim = 0;
  int64_t n_queries = 0;
};

LoadedData load_default_data_from_npy_cache()
{
  fs::path dir = default_data_cache_dir();

  std::cout << "Found existing data cache. Loaded: " << dir << "\n";

  std::vector<int64_t> dataset_ids_shape;
  std::vector<int64_t> dataset_shape;
  std::vector<int64_t> queries_shape;

  LoadedData out;
  out.dataset_ids = read_npy<int64_t>(dir / "dataset_ids.npy", "<i8", &dataset_ids_shape);
  out.dataset = read_npy<float>(dir / "dataset.npy", "<f4", &dataset_shape);
  out.queries = read_npy<float>(dir / "queries.npy", "<f4", &queries_shape);

  if (dataset_shape.size() != 2 || queries_shape.size() != 2) {
    throw std::runtime_error("dataset.npy and queries.npy must be 2D");
  }
  if (dataset_shape[1] != VECTOR_DIM || queries_shape[1] != VECTOR_DIM) {
    throw std::runtime_error("unexpected embedding dimension");
  }

  out.n_rows = dataset_shape[0];
  out.dim = dataset_shape[1];
  out.n_queries = queries_shape[0];

  std::cout << "Dataset IDs shape: (" << out.dataset_ids.size() << ")\n";
  std::cout << "Dataset shape: (" << out.n_rows << ", " << out.dim << ")\n";
  std::cout << "Dataset dtype: float32\n";
  std::cout << "Queries shape: (" << out.n_queries << ", " << out.dim << ")\n";
  std::cout << "Queries dtype: float32\n";
  std::cout << "First dataset id: " << out.dataset_ids.front() << "\n";
  std::cout << "Last dataset id: " << out.dataset_ids.back() << "\n";
  std::cout << "Embedding dimension: " << out.dim << "\n";

  return out;
}

struct GroundTruth {
  std::vector<int64_t> neighbors;
  int64_t rows = 0;
  int64_t cols = 0;
};

fs::path precomputed_neighbors_path()
{
  const fs::path data_dir = fs::path(getenv_or("CUVS_BENCH_DATA_DIR", (project_root() / "openai_large_5m").string()));
  return data_dir / "neighbors.parquet";
}

GroundTruth load_precomputed_ground_truth_from_parquet(int top_k, int query_limit)
{
  const fs::path path = precomputed_neighbors_path();
  if (!fs::exists(path)) {
    throw std::runtime_error("precomputed neighbors parquet does not exist: " + path.string());
  }

  std::shared_ptr<arrow::io::ReadableFile> infile =
    arrow_value_or_throw(arrow::io::ReadableFile::Open(path.string()), "ReadableFile::Open");

  std::unique_ptr<parquet::arrow::FileReader> reader =
    arrow_value_or_throw(
      parquet::arrow::OpenFile(infile, arrow::default_memory_pool()),
      "parquet::arrow::OpenFile");

  std::shared_ptr<arrow::Table> table =
    arrow_value_or_throw(reader->ReadTable(), "ReadTable");
  table = arrow_value_or_throw(table->CombineChunks(arrow::default_memory_pool()),
                               "CombineChunks(table)");

  auto id_col = table->GetColumnByName("id");
  auto neighbors_col = table->GetColumnByName("neighbors_id");
  if (!id_col || !neighbors_col) {
    throw std::runtime_error("neighbors parquet missing id or neighbors_id column");
  }
  if (table->num_rows() < query_limit) {
    throw std::runtime_error("query_limit exceeds precomputed neighbor rows");
  }

  if (id_col->num_chunks() != 1 || neighbors_col->num_chunks() != 1) {
    throw std::runtime_error("expected one chunk after CombineChunks(table)");
  }

  auto id_array_base = id_col->chunk(0);
  auto ids = std::dynamic_pointer_cast<arrow::Int64Array>(id_array_base);
  if (!ids) throw std::runtime_error("id column is not int64");

  for (int64_t i = 0; i < query_limit; ++i) {
    if (ids->Value(i) != i) {
      throw std::runtime_error("precomputed query ids do not match first query_limit queries");
    }
  }

  auto neighbors_array_base = neighbors_col->chunk(0);

  GroundTruth gt;
  gt.rows = query_limit;
  gt.cols = top_k;
  gt.neighbors.resize(static_cast<size_t>(query_limit * top_k));

  if (neighbors_array_base->type_id() == arrow::Type::LIST) {
    auto lists = std::dynamic_pointer_cast<arrow::ListArray>(neighbors_array_base);
    auto values = std::dynamic_pointer_cast<arrow::Int64Array>(lists->values());
    if (!values) throw std::runtime_error("neighbors_id list values are not int64");

    for (int64_t q = 0; q < query_limit; ++q) {
      const int64_t offset = lists->value_offset(q);
      const int64_t len = lists->value_length(q);
      if (len < top_k) throw std::runtime_error("neighbors_id row has fewer than top_k values");

      for (int j = 0; j < top_k; ++j) {
        gt.neighbors[static_cast<size_t>(q * top_k + j)] = values->Value(offset + j);
      }
    }
  } else if (neighbors_array_base->type_id() == arrow::Type::LARGE_LIST) {
    auto lists = std::dynamic_pointer_cast<arrow::LargeListArray>(neighbors_array_base);
    auto values = std::dynamic_pointer_cast<arrow::Int64Array>(lists->values());
    if (!values) throw std::runtime_error("neighbors_id large-list values are not int64");

    for (int64_t q = 0; q < query_limit; ++q) {
      const int64_t offset = lists->value_offset(q);
      const int64_t len = lists->value_length(q);
      if (len < top_k) throw std::runtime_error("neighbors_id row has fewer than top_k values");

      for (int j = 0; j < top_k; ++j) {
        gt.neighbors[static_cast<size_t>(q * top_k + j)] = values->Value(offset + j);
      }
    }
  } else {
    throw std::runtime_error("neighbors_id column is not a list/large_list");
  }

  std::cout << "Loaded precomputed full-dataset ground truth: " << path << "\n";
  return gt;
}

GroundTruth load_required_ground_truth()
{
  return load_precomputed_ground_truth_from_parquet(GROUND_TRUTH_TOP_K, QUERY_LIMIT);
}

DLDataType dl_float32()
{
  return DLDataType{kDLFloat, 32, 1};
}

DLDataType dl_float16()
{
  return DLDataType{kDLFloat, 16, 1};
}

DLDataType dl_int64()
{
  return DLDataType{kDLInt, 64, 1};
}

DLDataType dl_int32()
{
  return DLDataType{kDLInt, 32, 1};
}

void noop_dl_deleter(DLManagedTensor*) {}

struct DlpackMatrix {
  DLManagedTensor managed{};
  std::vector<int64_t> shape;

  DlpackMatrix(void* data, int64_t rows, int64_t cols, DLDataType dtype)
    : shape{rows, cols}
  {
    managed.dl_tensor.data = data;
    managed.dl_tensor.device = DLDevice{kDLCPU, 0};
    managed.dl_tensor.ndim = 2;
    managed.dl_tensor.dtype = dtype;
    managed.dl_tensor.shape = shape.data();
    managed.dl_tensor.strides = nullptr;
    managed.dl_tensor.byte_offset = 0;
    managed.manager_ctx = nullptr;
    managed.deleter = noop_dl_deleter;
  }

  DLManagedTensor* get() { return &managed; }
};

struct DlpackVector {
  DLManagedTensor managed{};
  std::vector<int64_t> shape;

  DlpackVector(void* data, int64_t rows, DLDataType dtype)
    : shape{rows}
  {
    managed.dl_tensor.data = data;
    managed.dl_tensor.device = DLDevice{kDLCPU, 0};
    managed.dl_tensor.ndim = 1;
    managed.dl_tensor.dtype = dtype;
    managed.dl_tensor.shape = shape.data();
    managed.dl_tensor.strides = nullptr;
    managed.dl_tensor.byte_offset = 0;
    managed.manager_ctx = nullptr;
    managed.deleter = noop_dl_deleter;
  }

  DLManagedTensor* get() { return &managed; }
};

struct MultiGpuResources {
  cuvsResources_t handle = 0;

  explicit MultiGpuResources(const std::optional<std::vector<int>>& device_ids)
  {
    if (device_ids.has_value()) {
      std::vector<int32_t> ids(device_ids->begin(), device_ids->end());
      DlpackVector ids_tensor(ids.data(), static_cast<int64_t>(ids.size()), dl_int32());
      check_cuvs(cuvsMultiGpuResourcesCreateWithDeviceIds(&handle, ids_tensor.get()),
                 "cuvsMultiGpuResourcesCreateWithDeviceIds");
    } else {
      check_cuvs(cuvsMultiGpuResourcesCreate(&handle), "cuvsMultiGpuResourcesCreate");
    }

    const int pool_percent = getenv_int_or("CUVS_BENCH_CAGRA_MEMORY_POOL_PERCENT", 30);
    if (pool_percent > 0) {
      check_cuvs(cuvsMultiGpuResourcesSetMemoryPool(handle, pool_percent),
                 "cuvsMultiGpuResourcesSetMemoryPool");
      std::cout << "device memory pool: " << pool_percent << "% of free memory\n";
    } else {
      std::cout << "device memory pool: disabled\n";
    }
  }

  ~MultiGpuResources()
  {
    if (handle != 0) {
      try {
        check_cuvs(cuvsMultiGpuResourcesDestroy(handle), "cuvsMultiGpuResourcesDestroy");
      } catch (...) {
      }
    }
  }

  MultiGpuResources(const MultiGpuResources&) = delete;
  MultiGpuResources& operator=(const MultiGpuResources&) = delete;

  void sync() const
  {
    check_cuvs(cuvsStreamSync(handle), "cuvsStreamSync");
  }
};

struct MultiGpuCagraIndex {
  cuvsMultiGpuCagraIndex_t index = nullptr;

  MultiGpuCagraIndex()
  {
    check_cuvs(cuvsMultiGpuCagraIndexCreate(&index), "cuvsMultiGpuCagraIndexCreate");
  }

  ~MultiGpuCagraIndex()
  {
    if (index != nullptr) {
      try {
        check_cuvs(cuvsMultiGpuCagraIndexDestroy(index), "cuvsMultiGpuCagraIndexDestroy");
      } catch (...) {
      }
    }
  }

  MultiGpuCagraIndex(const MultiGpuCagraIndex&) = delete;
  MultiGpuCagraIndex& operator=(const MultiGpuCagraIndex&) = delete;
};

struct CagraConfig {
  int graph_degree = 32;
  int intermediate_graph_degree = 64;
  std::string build_algo = "ivf_pq";

  bool compression_enabled = true;
  int compression_pq_bits = 8;
  int compression_pq_dim = 384;

  bool enable_exact_rerank = true;
  int rerank_candidate_k = 48;

  int itopk_size = 48;
  int max_queries = 256;
  int max_iterations = 0;
  std::string algo = "auto";
  int team_size = 0;
  int search_width = 1;
  int min_iterations = 0;
  int thread_block_size = 0;
  std::string hashmap_mode = "auto";
  int hashmap_min_bitlen = 0;
  float hashmap_max_fill_rate = 0.5f;
  uint32_t num_random_samplings = 1;
  uint64_t rand_xor_mask = 0x128394;
  int64_t n_rows_per_batch = 256;
};

struct BuildKey {
  int graph_degree;
  int intermediate_graph_degree;
  std::string build_algo;
  bool compression_enabled;
  int compression_pq_bits;
  int compression_pq_dim;

  bool operator<(const BuildKey& other) const
  {
    return std::tie(graph_degree,
                    intermediate_graph_degree,
                    build_algo,
                    compression_enabled,
                    compression_pq_bits,
                    compression_pq_dim) <
           std::tie(other.graph_degree,
                    other.intermediate_graph_degree,
                    other.build_algo,
                    other.compression_enabled,
                    other.compression_pq_bits,
                    other.compression_pq_dim);
  }
};

BuildKey build_key_from_config(const CagraConfig& c)
{
  return BuildKey{
    c.graph_degree,
    c.intermediate_graph_degree,
    c.build_algo,
    c.compression_enabled,
    c.compression_pq_bits,
    c.compression_pq_dim,
  };
}

bool exact_rerank_enabled(const CagraConfig& c)
{
  const bool default_enabled =
    getenv_or("CUVS_BENCH_CAGRA_ENABLE_EXACT_RERANK", "1") != "0";
  return c.enable_exact_rerank && default_enabled;
}

int get_search_k(const CagraConfig& c)
{
  if (!exact_rerank_enabled(c)) return K;
  const int default_candidate_k = getenv_int_or("CUVS_BENCH_CAGRA_RERANK_CANDIDATE_K", 48);
  return std::max(K, c.rerank_candidate_k > 0 ? c.rerank_candidate_k : default_candidate_k);
}

cuvsDistanceType distance_from_metric(const std::string& metric)
{
  if (metric == "sqeuclidean") return L2Expanded;
  if (metric == "inner_product") return InnerProduct;
  throw std::runtime_error("Multi-GPU CAGRA supports sqeuclidean and inner_product here");
}

cuvsCagraSearchAlgo search_algo_from_string(const std::string& name)
{
  if (name == "auto") return AUTO;
  if (name == "single_cta") return SINGLE_CTA;
  if (name == "multi_cta") return MULTI_CTA;
  if (name == "multi_kernel") return MULTI_KERNEL;
  throw std::runtime_error("unsupported CAGRA search algo: " + name);
}

cuvsCagraHashMode hash_mode_from_string(const std::string& name)
{
  if (name == "auto") return AUTO_HASH;
  if (name == "hash") return HASH;
  if (name == "small") return SMALL;
  throw std::runtime_error("unsupported CAGRA hashmap mode: " + name);
}

cuvsMultiGpuDistributionMode distribution_mode_from_string(const std::string& mode)
{
  if (mode == "sharded") return CUVS_NEIGHBORS_MG_SHARDED;
  if (mode == "replicated") return CUVS_NEIGHBORS_MG_REPLICATED;
  throw std::runtime_error("unsupported distribution mode: " + mode);
}

cuvsMultiGpuReplicatedSearchMode search_mode_from_string(const std::string& mode)
{
  if (mode == "load_balancer") return CUVS_NEIGHBORS_MG_LOAD_BALANCER;
  if (mode == "round_robin") return CUVS_NEIGHBORS_MG_ROUND_ROBIN;
  throw std::runtime_error("unsupported search mode: " + mode);
}

cuvsMultiGpuShardedMergeMode merge_mode_from_string(const std::string& mode)
{
  if (mode == "merge_on_root_rank") return CUVS_NEIGHBORS_MG_MERGE_ON_ROOT_RANK;
  if (mode == "tree_merge") return CUVS_NEIGHBORS_MG_TREE_MERGE;
  throw std::runtime_error("unsupported merge mode: " + mode);
}

struct IndexParamsOwner {
  cuvsMultiGpuCagraIndexParams_t mg = nullptr;
  cuvsCagraCompressionParams_t compression = nullptr;

  explicit IndexParamsOwner(const CagraConfig& c)
  {
    check_cuvs(cuvsMultiGpuCagraIndexParamsCreate(&mg), "cuvsMultiGpuCagraIndexParamsCreate");
    mg->mode = distribution_mode_from_string(DISTRIBUTION_MODE);

    cuvsCagraIndexParams_t base = mg->base_params;
    base->metric = distance_from_metric(METRIC);
    base->graph_degree = static_cast<size_t>(c.graph_degree);
    base->intermediate_graph_degree = static_cast<size_t>(c.intermediate_graph_degree);

    if (c.build_algo == "ivf_pq") {
      base->build_algo = IVF_PQ;
    } else if (c.build_algo == "auto") {
      base->build_algo = AUTO_SELECT;
    } else if (c.build_algo == "nn_descent") {
      base->build_algo = NN_DESCENT;
    } else {
      throw std::runtime_error("unsupported build_algo: " + c.build_algo);
    }

    if (c.compression_enabled) {
      check_cuvs(cuvsCagraCompressionParamsCreate(&compression),
                 "cuvsCagraCompressionParamsCreate");
      compression->pq_bits = static_cast<uint32_t>(c.compression_pq_bits);
      compression->pq_dim = static_cast<uint32_t>(c.compression_pq_dim);
      base->compression = compression;
    } else {
      base->compression = nullptr;
    }
  }

  ~IndexParamsOwner()
  {
    if (compression != nullptr) {
      try {
        check_cuvs(cuvsCagraCompressionParamsDestroy(compression),
                   "cuvsCagraCompressionParamsDestroy");
      } catch (...) {
      }
    }
    if (mg != nullptr) {
      try {
        check_cuvs(cuvsMultiGpuCagraIndexParamsDestroy(mg),
                   "cuvsMultiGpuCagraIndexParamsDestroy");
      } catch (...) {
      }
    }
  }

  IndexParamsOwner(const IndexParamsOwner&) = delete;
  IndexParamsOwner& operator=(const IndexParamsOwner&) = delete;
};

struct SearchParamsOwner {
  cuvsMultiGpuCagraSearchParams_t mg = nullptr;

  explicit SearchParamsOwner(const CagraConfig& c)
  {
    const int search_k = get_search_k(c);
    if (c.itopk_size < search_k) {
      throw std::runtime_error("CAGRA itopk_size should be at least search_k");
    }

    check_cuvs(cuvsMultiGpuCagraSearchParamsCreate(&mg), "cuvsMultiGpuCagraSearchParamsCreate");

    mg->search_mode = search_mode_from_string(SEARCH_MODE);
    mg->merge_mode = merge_mode_from_string(MERGE_MODE);
    mg->n_rows_per_batch = c.n_rows_per_batch;

    cuvsCagraSearchParams_t base = mg->base_params;
    base->itopk_size = static_cast<size_t>(c.itopk_size);
    base->max_queries = static_cast<size_t>(c.max_queries);
    base->max_iterations = static_cast<size_t>(c.max_iterations);
    base->algo = search_algo_from_string(c.algo);
    base->team_size = static_cast<size_t>(c.team_size);
    base->search_width = static_cast<size_t>(c.search_width);
    base->min_iterations = static_cast<size_t>(c.min_iterations);
    base->thread_block_size = static_cast<size_t>(c.thread_block_size);
    base->hashmap_mode = hash_mode_from_string(c.hashmap_mode);
    base->hashmap_min_bitlen = static_cast<size_t>(c.hashmap_min_bitlen);
    base->hashmap_max_fill_rate = c.hashmap_max_fill_rate;
    base->num_random_samplings = c.num_random_samplings;
    base->rand_xor_mask = c.rand_xor_mask;
  }

  ~SearchParamsOwner()
  {
    if (mg != nullptr) {
      try {
        check_cuvs(cuvsMultiGpuCagraSearchParamsDestroy(mg),
                   "cuvsMultiGpuCagraSearchParamsDestroy");
      } catch (...) {
      }
    }
  }

  SearchParamsOwner(const SearchParamsOwner&) = delete;
  SearchParamsOwner& operator=(const SearchParamsOwner&) = delete;
};

template <typename T>
DLDataType dtype_for_dlpack();

template <>
DLDataType dtype_for_dlpack<float>()
{
  return dl_float32();
}

template <>
DLDataType dtype_for_dlpack<__half>()
{
  return dl_float16();
}

template <typename T>
std::vector<T> convert_float_dataset(const std::vector<float>& src)
{
  std::vector<T> dst(src.size());
  for (size_t i = 0; i < src.size(); ++i) dst[i] = static_cast<T>(src[i]);
  return dst;
}

template <>
std::vector<__half> convert_float_dataset<__half>(const std::vector<float>& src)
{
  std::vector<__half> dst(src.size());
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < src.size(); ++i) dst[i] = __float2half(src[i]);
  return dst;
}

enum class RequestedStorageMode {
  Float32Resident,
  Float16Resident,
};

enum class ActiveDatasetMode {
  ResidentFloat32,
  ResidentFloat16,
};

RequestedStorageMode parse_storage_mode(const std::string& storage_dtype)
{
  if (storage_dtype == "float32") return RequestedStorageMode::Float32Resident;
  if (storage_dtype == "float16") return RequestedStorageMode::Float16Resident;
  throw std::invalid_argument("rerank storage dtype must be either 'float32' or 'float16'");
}

std::vector<int> default_device_ids()
{
  int device_count = 0;
  check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (device_count <= 0) throw std::runtime_error("No CUDA devices are visible");

  std::vector<int> ids;
  ids.reserve(static_cast<size_t>(device_count));
  for (int device_id = 0; device_id < device_count; ++device_id) ids.push_back(device_id);
  return ids;
}

std::vector<int> resolve_device_ids(const std::optional<std::vector<int>>& requested)
{
  std::vector<int> ids = requested.has_value() ? *requested : default_device_ids();
  if (ids.empty()) ids = default_device_ids();

  int device_count = 0;
  check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");

  for (size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] < 0 || ids[i] >= device_count) {
      throw std::invalid_argument("rerank device ids contain an invalid CUDA device id");
    }
    for (size_t j = 0; j < i; ++j) {
      if (ids[i] == ids[j]) {
        throw std::invalid_argument("rerank device ids contain a duplicate CUDA device id");
      }
    }
  }

  return ids;
}

struct SlotBuffers {
  cudaStream_t stream = nullptr;
  float* d_queries = nullptr;
  int64_t* d_candidates = nullptr;
  float* d_candidate_distances = nullptr;
  float* d_partial_distances = nullptr;
  int64_t* d_partial_rows = nullptr;
};

struct DeviceState {
  int device_id = 0;
  int64_t shard_start = 0;
  int64_t shard_end = 0;
  float* d_dataset_float = nullptr;
  __half* d_dataset_half = nullptr;
  SlotBuffers slots[2];

  DeviceState(int device, int64_t start, int64_t end)
    : device_id(device), shard_start(start), shard_end(end)
  {
  }

  DeviceState(const DeviceState&) = delete;
  DeviceState& operator=(const DeviceState&) = delete;

  ~DeviceState() { release(); }

  int64_t shard_rows() const { return shard_end - shard_start; }

  void release() noexcept
  {
    int previous_device = 0;
    cudaGetDevice(&previous_device);
    cudaSetDevice(device_id);

    for (auto& slot : slots) {
      if (slot.stream != nullptr) cudaStreamSynchronize(slot.stream);
    }

    if (d_dataset_float != nullptr) {
      cudaFree(d_dataset_float);
      d_dataset_float = nullptr;
    }
    if (d_dataset_half != nullptr) {
      cudaFree(d_dataset_half);
      d_dataset_half = nullptr;
    }

    for (auto& slot : slots) {
      if (slot.d_queries != nullptr) {
        cudaFree(slot.d_queries);
        slot.d_queries = nullptr;
      }
      if (slot.d_candidates != nullptr) {
        cudaFree(slot.d_candidates);
        slot.d_candidates = nullptr;
      }
      if (slot.d_candidate_distances != nullptr) {
        cudaFree(slot.d_candidate_distances);
        slot.d_candidate_distances = nullptr;
      }
      if (slot.d_partial_distances != nullptr) {
        cudaFree(slot.d_partial_distances);
        slot.d_partial_distances = nullptr;
      }
      if (slot.d_partial_rows != nullptr) {
        cudaFree(slot.d_partial_rows);
        slot.d_partial_rows = nullptr;
      }
      if (slot.stream != nullptr) {
        cudaStreamDestroy(slot.stream);
        slot.stream = nullptr;
      }
    }

    cudaSetDevice(previous_device);
  }
};

struct HostSlot {
  PinnedHostBuffer<float> queries;
  PinnedHostBuffer<int64_t> candidates;
  std::vector<PinnedHostBuffer<float>> partial_distances;
  std::vector<PinnedHostBuffer<int64_t>> partial_rows;
  int64_t start = 0;
  int64_t batch_queries = 0;
  bool active = false;
};

template <typename T>
void device_malloc(T** ptr, int64_t count, const std::string& what, int device_id)
{
  *ptr = nullptr;
  if (count == 0) return;

  check_cuda_device(
    cudaMalloc(reinterpret_cast<void**>(ptr), checked_bytes(count, sizeof(T), what.c_str())),
    "cudaMalloc " + what,
    device_id);
}

__global__ void compute_candidate_l2_kernel(
  const float* __restrict__ dataset_shard,
  int64_t shard_start,
  int64_t shard_end,
  int64_t dim,
  const float* __restrict__ queries,
  const int64_t* __restrict__ candidates,
  int64_t candidate_k,
  float* __restrict__ candidate_distances)
{
  const int64_t candidate_idx = blockIdx.x;
  const int64_t query_idx = blockIdx.y;
  const int thread_idx = threadIdx.x;
  const int64_t row = candidates[query_idx * candidate_k + candidate_idx];

  extern __shared__ float partial_sums[];
  float sum = 0.0f;

  if (row >= shard_start && row < shard_end) {
    const int64_t local_row = row - shard_start;
    const float* dataset_row = dataset_shard + local_row * dim;
    const float* query_row = queries + query_idx * dim;

    for (int64_t d = thread_idx; d < dim; d += blockDim.x) {
      const float diff = query_row[d] - dataset_row[d];
      sum += diff * diff;
    }
  } else {
    sum = (thread_idx == 0) ? INFINITY : 0.0f;
  }

  partial_sums[thread_idx] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (thread_idx < stride) partial_sums[thread_idx] += partial_sums[thread_idx + stride];
    __syncthreads();
  }

  if (thread_idx == 0) {
    candidate_distances[query_idx * candidate_k + candidate_idx] = partial_sums[0];
  }
}

__global__ void compute_candidate_l2_half_kernel(
  const __half* __restrict__ dataset_shard,
  int64_t shard_start,
  int64_t shard_end,
  int64_t dim,
  const float* __restrict__ queries,
  const int64_t* __restrict__ candidates,
  int64_t candidate_k,
  float* __restrict__ candidate_distances)
{
  const int64_t candidate_idx = blockIdx.x;
  const int64_t query_idx = blockIdx.y;
  const int thread_idx = threadIdx.x;
  const int64_t row = candidates[query_idx * candidate_k + candidate_idx];

  extern __shared__ float partial_sums[];
  float sum = 0.0f;

  if (row >= shard_start && row < shard_end) {
    const int64_t local_row = row - shard_start;
    const __half* dataset_row = dataset_shard + local_row * dim;
    const float* query_row = queries + query_idx * dim;

    for (int64_t d = thread_idx; d < dim; d += blockDim.x) {
      const float diff = query_row[d] - __half2float(dataset_row[d]);
      sum += diff * diff;
    }
  } else {
    sum = (thread_idx == 0) ? INFINITY : 0.0f;
  }

  partial_sums[thread_idx] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (thread_idx < stride) partial_sums[thread_idx] += partial_sums[thread_idx + stride];
    __syncthreads();
  }

  if (thread_idx == 0) {
    candidate_distances[query_idx * candidate_k + candidate_idx] = partial_sums[0];
  }
}

__global__ void float_to_half_kernel(
  const float* __restrict__ input,
  __half* __restrict__ output,
  int64_t count)
{
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) output[idx] = __float2half(input[idx]);
}

__global__ void select_local_topk_kernel(
  const float* __restrict__ candidate_distances,
  const int64_t* __restrict__ candidates,
  int64_t candidate_k,
  int64_t final_k,
  float* __restrict__ partial_distances,
  int64_t* __restrict__ partial_rows)
{
  if (threadIdx.x != 0) return;

  const int64_t query_idx = blockIdx.x;
  int selected_indices[kMaxFinalK];

  for (int64_t out_idx = 0; out_idx < final_k; ++out_idx) {
    float best_distance = INFINITY;
    int64_t best_row = -1;
    int best_candidate_idx = -1;

    for (int64_t candidate_idx = 0; candidate_idx < candidate_k; ++candidate_idx) {
      bool already_selected = false;
      for (int64_t previous = 0; previous < out_idx; ++previous) {
        if (selected_indices[previous] == candidate_idx) {
          already_selected = true;
          break;
        }
      }
      if (already_selected) continue;

      const float distance = candidate_distances[query_idx * candidate_k + candidate_idx];
      const int64_t row = candidates[query_idx * candidate_k + candidate_idx];

      if (distance < best_distance ||
          (distance == best_distance && row >= 0 && (best_row < 0 || row < best_row))) {
        best_distance = distance;
        best_row = row;
        best_candidate_idx = static_cast<int>(candidate_idx);
      }
    }

    selected_indices[out_idx] = best_candidate_idx;
    partial_distances[query_idx * final_k + out_idx] = best_distance;
    partial_rows[query_idx * final_k + out_idx] =
      best_candidate_idx >= 0 ? best_row : -1;
  }
}

class MultiGpuExactReranker {
public:
  MultiGpuExactReranker(
    const LoadedData& data,
    int64_t final_k,
    int64_t candidate_k,
    int64_t batch_size,
    const std::optional<std::vector<int>>& device_ids,
    const std::string& storage_dtype)
    : final_k_(final_k),
      candidate_k_(candidate_k),
      batch_size_(batch_size),
      requested_storage_mode_(parse_storage_mode(storage_dtype)),
      host_dataset_ptr_(data.dataset.data()),
      device_ids_(resolve_device_ids(device_ids)),
      dataset_ids_(data.dataset_ids)
  {
    n_rows_ = data.n_rows;
    dim_ = data.dim;

    if (n_rows_ <= 0) throw std::invalid_argument("rerank dataset must have at least one row");
    if (dim_ <= 0) throw std::invalid_argument("rerank dataset dimension must be positive");
    if (static_cast<int64_t>(data.dataset_ids.size()) != n_rows_) {
      throw std::invalid_argument("dataset_ids length must match dataset rows");
    }
    if (static_cast<int64_t>(data.dataset.size()) != checked_mul(n_rows_, dim_, "dataset")) {
      throw std::invalid_argument("dataset vector size must match dataset shape");
    }
    if (candidate_k_ <= 0) throw std::invalid_argument("candidate_k must be positive");
    if (final_k_ <= 0) throw std::invalid_argument("final_k must be positive");
    if (final_k_ > candidate_k_) {
      throw std::invalid_argument("final_k cannot be larger than candidate_k");
    }
    if (final_k_ > kMaxFinalK) {
      throw std::invalid_argument("final_k is larger than the CUDA local top-k limit");
    }
    if (batch_size_ <= 0) throw std::invalid_argument("rerank batch_size must be positive");

    if (requested_storage_mode_ == RequestedStorageMode::Float16Resident) {
      if (!can_use_resident_dataset(sizeof(__half))) {
        throw std::runtime_error(
          "float16 resident rerank dataset does not fit in visible GPU memory. "
          "Reduce visible devices/index memory pressure or reduce rerank batch size.");
      }
      active_dataset_mode_ = ActiveDatasetMode::ResidentFloat16;
    } else {
      if (!can_use_resident_dataset(sizeof(float))) {
        throw std::runtime_error(
          "float32 resident rerank dataset does not fit in visible GPU memory. "
          "Reduce visible devices/index memory pressure or reduce rerank batch size.");
      }
      active_dataset_mode_ = ActiveDatasetMode::ResidentFloat32;
    }

    const int64_t device_count = static_cast<int64_t>(device_ids_.size());
    for (int64_t device_index = 0; device_index < device_count; ++device_index) {
      const int device_id = device_ids_[static_cast<size_t>(device_index)];
      const int64_t shard_start = n_rows_ * device_index / device_count;
      const int64_t shard_end = n_rows_ * (device_index + 1) / device_count;

      auto state = std::make_unique<DeviceState>(device_id, shard_start, shard_end);
      DeviceGuard guard(device_id);

      if (active_dataset_mode_ == ActiveDatasetMode::ResidentFloat32) {
        device_malloc(
          &state->d_dataset_float,
          checked_mul(state->shard_rows(), dim_, "dataset shard"),
          "dataset shard",
          device_id);

        if (state->shard_rows() > 0) {
          check_cuda_device(
            cudaMemcpy(
              state->d_dataset_float,
              host_dataset_ptr_ + shard_start * dim_,
              checked_bytes(
                checked_mul(state->shard_rows(), dim_, "dataset shard"),
                sizeof(float),
                "dataset shard"),
              cudaMemcpyHostToDevice),
            "copy dataset shard to device",
            device_id);
        }
      } else if (active_dataset_mode_ == ActiveDatasetMode::ResidentFloat16) {
        device_malloc(
          &state->d_dataset_half,
          checked_mul(state->shard_rows(), dim_, "dataset shard"),
          "float16 dataset shard",
          device_id);
        upload_half_dataset_shard(*state, shard_start);
      }

      allocate_slots(*state);
      devices_.push_back(std::move(state));
    }

    allocate_host_slots();
  }

  void rerank(
    const float* queries_ptr,
    const int64_t* candidates_ptr,
    int64_t n_queries,
    float* out_distances_ptr,
    int64_t* out_neighbors_ptr)
  {
    if (n_queries < 0) throw std::invalid_argument("n_queries cannot be negative");
    if (n_queries == 0) return;
    if (queries_ptr == nullptr || candidates_ptr == nullptr ||
        out_distances_ptr == nullptr || out_neighbors_ptr == nullptr) {
      throw std::invalid_argument("rerank received a null buffer pointer");
    }

    validate_candidates(candidates_ptr, n_queries);

    int next_slot = 0;
    try {
      for (int64_t start = 0; start < n_queries; start += batch_size_) {
        if (host_slots_[next_slot].active) {
          synchronize_slot(next_slot, out_distances_ptr, out_neighbors_ptr);
        }

        const int64_t batch_queries = std::min(batch_size_, n_queries - start);
        launch_batch(next_slot, start, batch_queries, queries_ptr, candidates_ptr);
        next_slot = 1 - next_slot;
      }

      synchronize_slot(0, out_distances_ptr, out_neighbors_ptr);
      synchronize_slot(1, out_distances_ptr, out_neighbors_ptr);
    } catch (...) {
      synchronize_active_slots_no_merge();
      throw;
    }
  }

  std::string mode() const
  {
    if (active_dataset_mode_ == ActiveDatasetMode::ResidentFloat16) return "resident_float16";
    return "resident_float32";
  }

private:
  bool can_use_resident_dataset(size_t dataset_item_size) const
  {
    constexpr size_t reserve_bytes = static_cast<size_t>(512) * 1024 * 1024;
    const int64_t device_count = static_cast<int64_t>(device_ids_.size());
    const int64_t query_values = checked_mul(batch_size_, dim_, "resident query buffer");
    const int64_t candidate_values =
      checked_mul(batch_size_, candidate_k_, "resident candidate buffer");
    const int64_t output_values =
      checked_mul(batch_size_, final_k_, "resident output buffer");

    const size_t slot_bytes =
      2 * (checked_bytes(query_values, sizeof(float), "query buffer") +
           checked_bytes(candidate_values, sizeof(int64_t), "candidate buffer") +
           checked_bytes(candidate_values, sizeof(float), "candidate distance buffer") +
           checked_bytes(output_values, sizeof(float), "partial distance buffer") +
           checked_bytes(output_values, sizeof(int64_t), "partial row buffer"));

    for (int64_t device_index = 0; device_index < device_count; ++device_index) {
      const int device_id = device_ids_[static_cast<size_t>(device_index)];
      const int64_t shard_start = n_rows_ * device_index / device_count;
      const int64_t shard_end = n_rows_ * (device_index + 1) / device_count;
      const int64_t shard_rows = shard_end - shard_start;
      const int64_t shard_values = checked_mul(shard_rows, dim_, "dataset shard");
      const size_t shard_bytes =
        checked_bytes(shard_values, dataset_item_size, "dataset shard");
      const size_t upload_staging_bytes = dataset_item_size == sizeof(__half)
        ? std::min(
            checked_bytes(shard_values, sizeof(float), "float16 upload staging"),
            kHalfUploadChunkBytes)
        : 0;

      DeviceGuard guard(device_id);
      size_t free_bytes = 0;
      size_t total_bytes = 0;
      check_cuda_device(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo", device_id);

      if (shard_bytes + slot_bytes + upload_staging_bytes + reserve_bytes > free_bytes) {
        return false;
      }
    }

    return true;
  }

  void upload_half_dataset_shard(DeviceState& state, int64_t shard_start)
  {
    const int device_id = state.device_id;
    DeviceGuard guard(device_id);
    const int64_t total_values = checked_mul(state.shard_rows(), dim_, "float16 dataset shard");
    if (total_values == 0) return;

    const int64_t max_chunk_values =
      static_cast<int64_t>(std::max<size_t>(1, kHalfUploadChunkBytes / sizeof(float)));
    const int64_t chunk_values = std::min(total_values, max_chunk_values);

    cudaStream_t stream = nullptr;
    float* d_upload = nullptr;

    try {
      check_cuda_device(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags float16 upload",
        device_id);
      device_malloc(&d_upload, chunk_values, "float16 upload staging", device_id);

      const float* host_shard = host_dataset_ptr_ + shard_start * dim_;
      for (int64_t offset = 0; offset < total_values; offset += chunk_values) {
        const int64_t values_this_chunk = std::min(chunk_values, total_values - offset);

        check_cuda_device(
          cudaMemcpyAsync(
            d_upload,
            host_shard + offset,
            checked_bytes(values_this_chunk, sizeof(float), "float16 upload chunk"),
            cudaMemcpyHostToDevice,
            stream),
          "copy float16 upload chunk to device",
          device_id);

        const int blocks =
          static_cast<int>((values_this_chunk + kConvertThreads - 1) / kConvertThreads);
        float_to_half_kernel<<<blocks, kConvertThreads, 0, stream>>>(
          d_upload, state.d_dataset_half + offset, values_this_chunk);
        check_cuda_device(cudaGetLastError(), "launch float_to_half_kernel", device_id);
      }

      check_cuda_device(cudaStreamSynchronize(stream), "sync float16 upload stream", device_id);
    } catch (...) {
      if (d_upload != nullptr) cudaFree(d_upload);
      if (stream != nullptr) cudaStreamDestroy(stream);
      throw;
    }

    check_cuda_device(cudaFree(d_upload), "cudaFree float16 upload staging", device_id);
    check_cuda_device(cudaStreamDestroy(stream), "cudaStreamDestroy float16 upload", device_id);
  }

  void allocate_slots(DeviceState& state)
  {
    const int device_id = state.device_id;
    const int64_t query_values = checked_mul(batch_size_, dim_, "query buffer");
    const int64_t candidate_values = checked_mul(batch_size_, candidate_k_, "candidate buffer");
    const int64_t output_values = checked_mul(batch_size_, final_k_, "output buffer");

    for (auto& slot : state.slots) {
      check_cuda_device(
        cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags",
        device_id);

      device_malloc(&slot.d_queries, query_values, "queries", device_id);
      device_malloc(&slot.d_candidates, candidate_values, "candidates", device_id);
      device_malloc(&slot.d_candidate_distances, candidate_values, "candidate distances", device_id);
      device_malloc(&slot.d_partial_distances, output_values, "partial distances", device_id);
      device_malloc(&slot.d_partial_rows, output_values, "partial rows", device_id);
    }
  }

  void allocate_host_slots()
  {
    const int64_t query_values = checked_mul(batch_size_, dim_, "host query buffer");
    const int64_t candidate_values =
      checked_mul(batch_size_, candidate_k_, "host candidate buffer");
    const int64_t output_values = checked_mul(batch_size_, final_k_, "host output buffer");

    for (auto& slot : host_slots_) {
      slot.queries.allocate(static_cast<size_t>(query_values));
      slot.candidates.allocate(static_cast<size_t>(candidate_values));
      slot.partial_distances.reserve(devices_.size());
      slot.partial_rows.reserve(devices_.size());

      for (size_t device_index = 0; device_index < devices_.size(); ++device_index) {
        slot.partial_distances.emplace_back(static_cast<size_t>(output_values));
        slot.partial_rows.emplace_back(static_cast<size_t>(output_values));
      }
    }
  }

  void validate_candidates(const int64_t* candidates_ptr, int64_t n_queries) const
  {
    const int64_t candidate_values = checked_mul(n_queries, candidate_k_, "candidate values");
    for (int64_t i = 0; i < candidate_values; ++i) {
      const int64_t row = candidates_ptr[i];
      if (row < 0 || row >= n_rows_) {
        throw std::out_of_range("candidate_neighbors contains an invalid dataset row index");
      }
    }
  }

  void launch_batch(
    int slot_index,
    int64_t start,
    int64_t batch_queries,
    const float* queries_ptr,
    const int64_t* candidates_ptr)
  {
    HostSlot& host_slot = host_slots_[slot_index];
    host_slot.start = start;
    host_slot.batch_queries = batch_queries;
    host_slot.active = false;

    const int64_t query_values = checked_mul(batch_queries, dim_, "query values");
    const int64_t candidate_values =
      checked_mul(batch_queries, candidate_k_, "candidate values");
    const int64_t output_values = checked_mul(batch_queries, final_k_, "output values");

    std::memcpy(
      host_slot.queries.data(),
      queries_ptr + start * dim_,
      checked_bytes(query_values, sizeof(float), "query copy"));
    std::memcpy(
      host_slot.candidates.data(),
      candidates_ptr + start * candidate_k_,
      checked_bytes(candidate_values, sizeof(int64_t), "candidate copy"));

    for (size_t device_index = 0; device_index < devices_.size(); ++device_index) {
      DeviceState& device = *devices_[device_index];
      DeviceGuard guard(device.device_id);
      SlotBuffers& slot = device.slots[slot_index];

      check_cuda_device(
        cudaMemcpyAsync(
          slot.d_queries,
          host_slot.queries.data(),
          checked_bytes(query_values, sizeof(float), "query H2D"),
          cudaMemcpyHostToDevice,
          slot.stream),
        "copy queries to device",
        device.device_id);

      check_cuda_device(
        cudaMemcpyAsync(
          slot.d_candidates,
          host_slot.candidates.data(),
          checked_bytes(candidate_values, sizeof(int64_t), "candidate H2D"),
          cudaMemcpyHostToDevice,
          slot.stream),
        "copy candidates to device",
        device.device_id);

      const dim3 distance_grid(
        static_cast<unsigned int>(candidate_k_),
        static_cast<unsigned int>(batch_queries));
      if (active_dataset_mode_ == ActiveDatasetMode::ResidentFloat32) {
        compute_candidate_l2_kernel<<<
          distance_grid,
          kDistanceThreads,
          sizeof(float) * kDistanceThreads,
          slot.stream>>>(
          device.d_dataset_float,
          device.shard_start,
          device.shard_end,
          dim_,
          slot.d_queries,
          slot.d_candidates,
          candidate_k_,
          slot.d_candidate_distances);
        check_cuda_device(cudaGetLastError(), "launch compute_candidate_l2_kernel", device.device_id);
      } else if (active_dataset_mode_ == ActiveDatasetMode::ResidentFloat16) {
        compute_candidate_l2_half_kernel<<<
          distance_grid,
          kDistanceThreads,
          sizeof(float) * kDistanceThreads,
          slot.stream>>>(
          device.d_dataset_half,
          device.shard_start,
          device.shard_end,
          dim_,
          slot.d_queries,
          slot.d_candidates,
          candidate_k_,
          slot.d_candidate_distances);
        check_cuda_device(
          cudaGetLastError(), "launch compute_candidate_l2_half_kernel", device.device_id);
      }

      select_local_topk_kernel<<<
        static_cast<unsigned int>(batch_queries),
        1,
        0,
        slot.stream>>>(
        slot.d_candidate_distances,
        slot.d_candidates,
        candidate_k_,
        final_k_,
        slot.d_partial_distances,
        slot.d_partial_rows);
      check_cuda_device(cudaGetLastError(), "launch select_local_topk_kernel", device.device_id);

      check_cuda_device(
        cudaMemcpyAsync(
          host_slot.partial_distances[device_index].data(),
          slot.d_partial_distances,
          checked_bytes(output_values, sizeof(float), "partial distance D2H"),
          cudaMemcpyDeviceToHost,
          slot.stream),
        "copy partial distances to host",
        device.device_id);

      check_cuda_device(
        cudaMemcpyAsync(
          host_slot.partial_rows[device_index].data(),
          slot.d_partial_rows,
          checked_bytes(output_values, sizeof(int64_t), "partial row D2H"),
          cudaMemcpyDeviceToHost,
          slot.stream),
        "copy partial rows to host",
        device.device_id);
    }

    host_slot.active = true;
  }

  void synchronize_slot(int slot_index, float* out_distances_ptr, int64_t* out_neighbors_ptr)
  {
    HostSlot& host_slot = host_slots_[slot_index];
    if (!host_slot.active) return;

    for (const auto& device : devices_) {
      DeviceGuard guard(device->device_id);
      check_cuda_device(
        cudaStreamSynchronize(device->slots[slot_index].stream),
        "sync rerank stream",
        device->device_id);
    }

    merge_slot(host_slot, out_distances_ptr, out_neighbors_ptr);
    host_slot.active = false;
  }

  void synchronize_active_slots_no_merge()
  {
    for (int slot_index = 0; slot_index < 2; ++slot_index) {
      HostSlot& host_slot = host_slots_[slot_index];
      if (!host_slot.active) continue;

      for (const auto& device : devices_) {
        DeviceGuard guard(device->device_id);
        cudaStreamSynchronize(device->slots[slot_index].stream);
      }
      host_slot.active = false;
    }
  }

  bool row_already_selected(
    const int64_t* out_neighbors_ptr,
    int64_t query_offset,
    int64_t out_idx,
    int64_t dataset_id) const
  {
    for (int64_t previous = 0; previous < out_idx; ++previous) {
      if (out_neighbors_ptr[query_offset + previous] == dataset_id) return true;
    }
    return false;
  }

  void merge_slot(HostSlot& host_slot, float* out_distances_ptr, int64_t* out_neighbors_ptr)
  {
    for (int64_t local_q = 0; local_q < host_slot.batch_queries; ++local_q) {
      const int64_t global_q = host_slot.start + local_q;
      const int64_t output_base = global_q * final_k_;

      for (int64_t out_idx = 0; out_idx < final_k_; ++out_idx) {
        float best_distance = INFINITY;
        int64_t best_row = -1;
        int64_t best_dataset_id = -1;

        for (size_t device_index = 0; device_index < devices_.size(); ++device_index) {
          const float* partial_distances =
            host_slot.partial_distances[device_index].data() + local_q * final_k_;
          const int64_t* partial_rows =
            host_slot.partial_rows[device_index].data() + local_q * final_k_;

          for (int64_t candidate_idx = 0; candidate_idx < final_k_; ++candidate_idx) {
            const int64_t row = partial_rows[candidate_idx];
            if (row < 0) continue;

            const int64_t dataset_id = dataset_ids_[static_cast<size_t>(row)];
            if (row_already_selected(out_neighbors_ptr, output_base, out_idx, dataset_id)) {
              continue;
            }

            const float distance = partial_distances[candidate_idx];
            if (distance < best_distance ||
                (distance == best_distance && (best_row < 0 || row < best_row))) {
              best_distance = distance;
              best_row = row;
              best_dataset_id = dataset_id;
            }
          }
        }

        if (best_row < 0 || !std::isfinite(best_distance)) {
          throw std::runtime_error("not enough valid candidates to produce final_k rerank results");
        }

        out_distances_ptr[output_base + out_idx] = best_distance;
        out_neighbors_ptr[output_base + out_idx] = best_dataset_id;
      }
    }
  }

  int64_t n_rows_ = 0;
  int64_t dim_ = 0;
  int64_t final_k_ = 0;
  int64_t candidate_k_ = 0;
  int64_t batch_size_ = 0;
  RequestedStorageMode requested_storage_mode_ = RequestedStorageMode::Float16Resident;
  ActiveDatasetMode active_dataset_mode_ = ActiveDatasetMode::ResidentFloat16;
  const float* host_dataset_ptr_ = nullptr;
  std::vector<int> device_ids_;
  std::vector<int64_t> dataset_ids_;
  std::vector<std::unique_ptr<DeviceState>> devices_;
  HostSlot host_slots_[2];
};

struct SearchOutput {
  std::vector<float> distances;
  std::vector<int64_t> neighbors;
  int64_t rows = 0;
  int64_t cols = 0;
};

template <typename T>
struct SearchRunner {
  const MultiGpuResources& resources;
  cuvsMultiGpuCagraIndex_t index;
  const SearchParamsOwner& params;
  MatrixViewHost<const T> queries;
  MatrixViewHost<const float> rerank_queries;
  const LoadedData& data;
  int final_k = K;
  int search_k = K;
  bool exact_rerank = true;
  MultiGpuExactReranker* reranker = nullptr;

  PinnedHostBuffer<int64_t> row_neighbors;
  PinnedHostBuffer<float> candidate_distances;

  SearchOutput last_output;

  SearchRunner(
    const MultiGpuResources& resources_,
    cuvsMultiGpuCagraIndex_t index_,
    const SearchParamsOwner& params_,
    MatrixViewHost<const T> queries_,
    MatrixViewHost<const float> rerank_queries_,
    const LoadedData& data_,
    int final_k_,
    int search_k_,
    bool exact_rerank_,
    MultiGpuExactReranker* reranker_)
    : resources(resources_),
      index(index_),
      params(params_),
      queries(queries_),
      rerank_queries(rerank_queries_),
      data(data_),
      final_k(final_k_),
      search_k(search_k_),
      exact_rerank(exact_rerank_),
      reranker(reranker_),
      row_neighbors(static_cast<size_t>(queries_.rows * search_k_)),
      candidate_distances(static_cast<size_t>(queries_.rows * search_k_))
  {
    last_output.rows = queries.rows;
    last_output.cols = final_k;
    last_output.distances.resize(static_cast<size_t>(queries.rows * final_k));
    last_output.neighbors.resize(static_cast<size_t>(queries.rows * final_k));
  }

  void run()
  {
    DlpackMatrix query_tensor(
      const_cast<T*>(queries.data), queries.rows, queries.cols, dtype_for_dlpack<T>());
    DlpackMatrix neighbor_tensor(row_neighbors.data(), queries.rows, search_k, dl_int64());
    DlpackMatrix distance_tensor(candidate_distances.data(), queries.rows, search_k, dl_float32());

    check_cuvs(cuvsMultiGpuCagraSearch(resources.handle,
                                       params.mg,
                                       index,
                                       query_tensor.get(),
                                       neighbor_tensor.get(),
                                       distance_tensor.get()),
               "cuvsMultiGpuCagraSearch");

    resources.sync();

    if (exact_rerank) {
      if (reranker == nullptr) {
        throw std::runtime_error("exact CAGRA rerank requested without a multi-GPU reranker");
      }
      reranker->rerank(
        rerank_queries.data,
        row_neighbors.data(),
        queries.rows,
        last_output.distances.data(),
        last_output.neighbors.data());
      return;
    }

    for (int64_t q = 0; q < queries.rows; ++q) {
      for (int k = 0; k < final_k; ++k) {
        const int64_t row = row_neighbors[static_cast<size_t>(q * search_k + k)];
        if (row < 0 || row >= data.n_rows) {
          throw std::runtime_error("CAGRA returned row neighbor outside dataset range");
        }

        last_output.distances[static_cast<size_t>(q * final_k + k)] =
          candidate_distances[static_cast<size_t>(q * search_k + k)];
        last_output.neighbors[static_cast<size_t>(q * final_k + k)] =
          data.dataset_ids[static_cast<size_t>(row)];
      }
    }
  }
};

struct TimingSummary {
  int runs = 0;
  double min_sec = 0.0;
  double mean_sec = 0.0;
  double median_sec = 0.0;
  double p95_sec = 0.0;
  std::vector<double> all_sec;
};

TimingSummary summarize_times(std::vector<double> times)
{
  std::vector<double> sorted = times;
  std::sort(sorted.begin(), sorted.end());

  TimingSummary s;
  s.runs = static_cast<int>(times.size());
  s.min_sec = sorted.front();
  s.mean_sec = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
  s.median_sec = sorted[sorted.size() / 2];
  const size_t p95_index =
    std::min(sorted.size() - 1, static_cast<size_t>(0.95 * static_cast<double>(sorted.size() - 1)));
  s.p95_sec = sorted[p95_index];
  s.all_sec = std::move(times);
  return s;
}

template <typename Fn>
TimingSummary measure_synchronized_wall_time(
  Fn&& fn,
  int warmup_runs,
  int timed_runs,
  const MultiGpuResources& resources)
{
  for (int i = 0; i < warmup_runs; ++i) {
    fn();
    resources.sync();
  }

  std::vector<double> times;
  times.reserve(static_cast<size_t>(timed_runs));

  for (int i = 0; i < timed_runs; ++i) {
    resources.sync();
    auto start = std::chrono::steady_clock::now();
    fn();
    resources.sync();
    auto end = std::chrono::steady_clock::now();

    times.push_back(std::chrono::duration<double>(end - start).count());
  }

  return summarize_times(std::move(times));
}

struct RecallResult {
  double recall = 0.0;
  int64_t total_correct = 0;
  int64_t total_possible = 0;
};

RecallResult calculate_recall_at_k(
  const SearchOutput& retrieved,
  const GroundTruth& ground_truth,
  int k)
{
  RecallResult r;
  r.total_possible = retrieved.rows * k;

  for (int64_t q = 0; q < retrieved.rows; ++q) {
    for (int i = 0; i < k; ++i) {
      const int64_t got = retrieved.neighbors[static_cast<size_t>(q * retrieved.cols + i)];

      for (int j = 0; j < k; ++j) {
        const int64_t expected = ground_truth.neighbors[static_cast<size_t>(q * ground_truth.cols + j)];
        if (got == expected) {
          r.total_correct += 1;
          break;
        }
      }
    }
  }

  r.recall = static_cast<double>(r.total_correct) / static_cast<double>(r.total_possible);
  return r;
}

struct ResultRow {
  CagraConfig config;
  int search_k = 0;
  bool exact_rerank = false;
  std::string rerank_backend = "none";
  std::string rerank_storage_dtype = "none";
  double build_time = 0.0;
  double search_time = 0.0;
  TimingSummary online_summary;
  double queries_per_second = 0.0;
  double latency_per_query = 0.0;
  double recall_at_10 = 0.0;
  int64_t total_correct = 0;
  int64_t total_possible = 0;
};

void write_results_csv(const std::vector<ResultRow>& results, const fs::path& output_path)
{
  fs::create_directories(output_path.parent_path());
  std::ofstream out(output_path);
  if (!out) throw std::runtime_error("failed to open output CSV: " + output_path.string());

  out << "graph_degree,intermediate_graph_degree,build_algo,"
      << "compression_enabled,compression_pq_bits,compression_pq_dim,"
      << "enable_exact_rerank,rerank_candidate_k,search_k,exact_rerank,"
      << "rerank_backend,rerank_storage_dtype,itopk_size,max_queries,"
      << "max_iterations,algo,team_size,search_width,min_iterations,"
      << "thread_block_size,hashmap_mode,hashmap_min_bitlen,"
      << "hashmap_max_fill_rate,num_random_samplings,rand_xor_mask,"
      << "n_rows_per_batch,build_time,search_time,queries_per_second,"
      << "latency_per_query,online_median_latency_ms,online_p95_latency_ms,"
      << "recall_at_10,total_correct,total_possible\n";

  out << std::setprecision(10);

  for (const ResultRow& r : results) {
    const CagraConfig& c = r.config;
    const double online_median_ms =
      r.online_summary.median_sec * MS_PER_SECOND / ONLINE_QUERY_COUNT;
    const double online_p95_ms =
      r.online_summary.p95_sec * MS_PER_SECOND / ONLINE_QUERY_COUNT;

    out << c.graph_degree << ','
        << c.intermediate_graph_degree << ','
        << c.build_algo << ','
        << (c.compression_enabled ? "True" : "False") << ','
        << c.compression_pq_bits << ','
        << c.compression_pq_dim << ','
        << (c.enable_exact_rerank ? "True" : "False") << ','
        << c.rerank_candidate_k << ','
        << r.search_k << ','
        << (r.exact_rerank ? "True" : "False") << ','
        << r.rerank_backend << ','
        << r.rerank_storage_dtype << ','
        << c.itopk_size << ','
        << c.max_queries << ','
        << c.max_iterations << ','
        << c.algo << ','
        << c.team_size << ','
        << c.search_width << ','
        << c.min_iterations << ','
        << c.thread_block_size << ','
        << c.hashmap_mode << ','
        << c.hashmap_min_bitlen << ','
        << c.hashmap_max_fill_rate << ','
        << c.num_random_samplings << ','
        << c.rand_xor_mask << ','
        << c.n_rows_per_batch << ','
        << r.build_time << ','
        << r.search_time << ','
        << r.queries_per_second << ','
        << r.latency_per_query << ','
        << online_median_ms << ','
        << online_p95_ms << ','
        << r.recall_at_10 << ','
        << r.total_correct << ','
        << r.total_possible << '\n';
  }
}

template <typename T>
PinnedHostBuffer<T> make_pinned_slice(const std::vector<T>& src, int64_t row_start, int64_t rows, int64_t cols)
{
  PinnedHostBuffer<T> out(static_cast<size_t>(rows * cols));
  const size_t offset = static_cast<size_t>(row_start * cols);
  std::copy(src.begin() + offset, src.begin() + offset + out.count, out.data());
  return out;
}

PinnedHostBuffer<float> make_pinned_float_slice(
  const std::vector<float>& src,
  int64_t row_start,
  int64_t rows,
  int64_t cols)
{
  PinnedHostBuffer<float> out(static_cast<size_t>(rows * cols));
  const size_t offset = static_cast<size_t>(row_start * cols);
  std::copy(src.begin() + offset, src.begin() + offset + out.count, out.data());
  return out;
}

std::unique_ptr<MultiGpuExactReranker> create_cagra_exact_reranker(
  const LoadedData& data,
  int candidate_k,
  int batch_size,
  const std::optional<std::vector<int>>& device_ids,
  const std::string& requested_storage_dtype,
  std::string* active_storage_dtype)
{
  auto reranker = std::make_unique<MultiGpuExactReranker>(
    data, K, candidate_k, batch_size, device_ids, requested_storage_dtype);
  if (active_storage_dtype != nullptr) *active_storage_dtype = requested_storage_dtype;
  return reranker;
}

template <typename T>
void run_cagra_configs_typed(
  const std::vector<CagraConfig>& configs,
  const fs::path& output_path)
{
  LoadedData data = load_default_data_from_npy_cache();
  GroundTruth ground_truth = load_required_ground_truth();

  const std::vector<T> index_dataset = convert_float_dataset<T>(data.dataset);
  const std::vector<T> search_queries = convert_float_dataset<T>(data.queries);

  const auto device_ids = optional_int_list_from_env("CUVS_BENCH_CAGRA_DEVICE_IDS");
  auto rerank_device_ids = optional_int_list_from_env("CUVS_BENCH_CAGRA_RERANK_DEVICE_IDS");
  if (!rerank_device_ids.has_value()) rerank_device_ids = device_ids;
  MultiGpuResources resources(device_ids);

  const int64_t offline_rows = std::min<int64_t>(OFFLINE_QUERY_COUNT, data.n_queries);
  const int64_t online_rows = std::min<int64_t>(ONLINE_QUERY_COUNT, data.n_queries);

  PinnedHostBuffer<T> benchmark_queries =
    make_pinned_slice(search_queries, 0, offline_rows, data.dim);
  PinnedHostBuffer<T> online_queries =
    make_pinned_slice(search_queries, 0, online_rows, data.dim);

  PinnedHostBuffer<float> benchmark_rerank_queries =
    make_pinned_float_slice(data.queries, 0, offline_rows, data.dim);
  PinnedHostBuffer<float> online_rerank_queries =
    make_pinned_float_slice(data.queries, 0, online_rows, data.dim);

  std::cout << "\nRunning CAGRA benchmark\n";
  std::cout << "Dataset shape: (" << data.n_rows << ", " << data.dim << ")\n";
  std::cout << "Dataset dtype: " << getenv_or("CUVS_BENCH_CAGRA_DATASET_DTYPE", "float16") << "\n";
  std::cout << "Queries shape: (" << offline_rows << ", " << data.dim << ")\n";
  std::cout << "Queries dtype: " << getenv_or("CUVS_BENCH_CAGRA_QUERY_DTYPE", "float16") << "\n";
  std::cout << "k: " << K << "\n";
  std::cout << "distribution mode: " << DISTRIBUTION_MODE << "\n";
  std::cout << "search mode: " << SEARCH_MODE << "\n";
  std::cout << "merge mode: " << MERGE_MODE << "\n";
  if (device_ids.has_value()) {
    std::cout << "device ids:";
    for (int id : *device_ids) std::cout << ' ' << id;
    std::cout << "\n";
  } else {
    std::cout << "device ids: all visible\n";
  }

  const std::string rerank_backend = getenv_or("CUVS_BENCH_CAGRA_RERANK_BACKEND", "multi_gpu");
  if (rerank_backend != "multi_gpu") {
    throw std::runtime_error(
      "CUVS_BENCH_CAGRA_RERANK_BACKEND must be 'multi_gpu'. "
      "Only multi-GPU exact rerank is supported by this runner.");
  }
  const int rerank_batch_size = getenv_int_or("CUVS_BENCH_CAGRA_RERANK_BATCH_SIZE", 512);
  const std::string requested_rerank_storage_dtype =
    getenv_or("CUVS_BENCH_CAGRA_RERANK_STORAGE_DTYPE", "float16");

  std::cout << "exact rerank default: "
            << (getenv_or("CUVS_BENCH_CAGRA_ENABLE_EXACT_RERANK", "1") != "0") << "\n";
  std::cout << "rerank backend: " << rerank_backend << "\n";
  std::cout << "rerank storage dtype: " << requested_rerank_storage_dtype << "\n";
  std::cout << "rerank batch size: " << rerank_batch_size << "\n";
  if (rerank_device_ids.has_value()) {
    std::cout << "rerank device ids:";
    for (int id : *rerank_device_ids) std::cout << ' ' << id;
    std::cout << "\n";
  } else {
    std::cout << "rerank device ids: all visible\n";
  }
  std::cout << "pinned host search I/O: enabled\n";

  std::map<BuildKey, std::vector<CagraConfig>> grouped;
  for (const CagraConfig& c : configs) grouped[build_key_from_config(c)].push_back(c);

  std::vector<ResultRow> results;
  int config_number = 0;
  const int total_configs = static_cast<int>(configs.size());

  for (const auto& [key, search_configs] : grouped) {
    CagraConfig build_config = search_configs.front();
    std::unique_ptr<MultiGpuExactReranker> reranker;
    int reranker_candidate_k = -1;
    std::string reranker_storage_dtype = "none";

    std::cout << "\nBuilding CAGRA index with "
              << "graph_degree=" << build_config.graph_degree << ", "
              << "intermediate_graph_degree=" << build_config.intermediate_graph_degree << ", "
              << "build_algo=" << build_config.build_algo << "...\n";

    IndexParamsOwner index_params(build_config);
    MultiGpuCagraIndex index;

    DlpackMatrix dataset_tensor(
      const_cast<T*>(index_dataset.data()), data.n_rows, data.dim, dtype_for_dlpack<T>());

    resources.sync();
    auto build_start = std::chrono::steady_clock::now();

    check_cuvs(cuvsMultiGpuCagraBuild(resources.handle,
                                      index_params.mg,
                                      dataset_tensor.get(),
                                      index.index),
               "cuvsMultiGpuCagraBuild");

    resources.sync();
    auto build_end = std::chrono::steady_clock::now();
    const double build_time = std::chrono::duration<double>(build_end - build_start).count();

    std::cout << "CAGRA index built in " << std::fixed << std::setprecision(2)
              << build_time << " seconds\n";

    for (const CagraConfig& config : search_configs) {
      config_number += 1;
      const bool exact_rerank = exact_rerank_enabled(config);
      const int search_k = get_search_k(config);

      std::cout << "\n[" << config_number << "/" << total_configs << "] "
                << "itopk_size=" << config.itopk_size << ", "
                << "search_k=" << search_k << ", "
                << "search_width=" << config.search_width << ", "
                << "max_iterations=" << config.max_iterations << "\n";

      if (exact_rerank) {
        if (reranker_candidate_k != search_k) {
          resources.sync();
          reranker.reset();
          resources.sync();

          std::cout << "Creating exact CAGRA reranker "
                    << "(candidate_k=" << search_k
                    << ", storage=" << requested_rerank_storage_dtype << ")...\n";
          reranker = create_cagra_exact_reranker(
            data,
            search_k,
            rerank_batch_size,
            rerank_device_ids,
            requested_rerank_storage_dtype,
            &reranker_storage_dtype);
          reranker_candidate_k = search_k;
          std::cout << "Exact reranker mode: " << reranker->mode()
                    << " (storage=" << reranker_storage_dtype << ")\n";
        }
      }

      SearchParamsOwner search_params(config);

      MatrixViewHost<const T> offline_query_view{
        benchmark_queries.data(), offline_rows, data.dim};
      MatrixViewHost<const T> online_query_view{
        online_queries.data(), online_rows, data.dim};
      MatrixViewHost<const float> offline_rerank_query_view{
        benchmark_rerank_queries.data(), offline_rows, data.dim};
      MatrixViewHost<const float> online_rerank_query_view{
        online_rerank_queries.data(), online_rows, data.dim};

      SearchRunner<T> offline_search(
        resources,
        index.index,
        search_params,
        offline_query_view,
        offline_rerank_query_view,
        data,
        K,
        search_k,
        exact_rerank,
        reranker.get());

      SearchRunner<T> online_search(
        resources,
        index.index,
        search_params,
        online_query_view,
        online_rerank_query_view,
        data,
        K,
        search_k,
        exact_rerank,
        reranker.get());

      TimingSummary offline_summary = measure_synchronized_wall_time(
        [&]() { offline_search.run(); },
        SEARCH_WARMUP_RUNS,
        SEARCH_TIMED_RUNS,
        resources);

      const SearchOutput offline_output = offline_search.last_output;
      const double search_time = offline_summary.median_sec;

      TimingSummary online_summary = measure_synchronized_wall_time(
        [&]() { online_search.run(); },
        SEARCH_WARMUP_RUNS,
        SEARCH_TIMED_RUNS,
        resources);

      const double queries_per_second = static_cast<double>(offline_rows) / search_time;
      const double latency_per_query = search_time * MS_PER_SECOND / static_cast<double>(offline_rows);

      RecallResult recall = calculate_recall_at_k(offline_output, ground_truth, K);

      std::cout << "First query top neighbors:";
      for (int i = 0; i < DISPLAY_TOP_K; ++i) {
        std::cout << ' ' << offline_output.neighbors[static_cast<size_t>(i)];
      }
      std::cout << "\n";

      std::cout << "First query top distances:";
      for (int i = 0; i < DISPLAY_TOP_K; ++i) {
        std::cout << ' ' << offline_output.distances[static_cast<size_t>(i)];
      }
      std::cout << "\n";

      std::cout << std::fixed << std::setprecision(4);
      std::cout << "Offline median: " << search_time << " seconds\n";
      std::cout << "Throughput: " << std::setprecision(2) << queries_per_second
                << " queries/second\n";
      std::cout << "Latency per query: " << std::setprecision(4)
                << latency_per_query << " ms\n";
      std::cout << "Recall@10: " << recall.recall << " ("
                << recall.total_correct << "/" << recall.total_possible << ")\n";

      ResultRow row;
      row.config = config;
      row.search_k = search_k;
      row.exact_rerank = exact_rerank;
      row.rerank_backend = exact_rerank ? rerank_backend : "none";
      row.rerank_storage_dtype = exact_rerank ? reranker_storage_dtype : "none";
      row.build_time = build_time;
      row.search_time = search_time;
      row.online_summary = online_summary;
      row.queries_per_second = queries_per_second;
      row.latency_per_query = latency_per_query;
      row.recall_at_10 = recall.recall;
      row.total_correct = recall.total_correct;
      row.total_possible = recall.total_possible;
      results.push_back(std::move(row));

      resources.sync();
    }

    resources.sync();
  }

  write_results_csv(results, output_path);
  std::cout << "\nSaved CAGRA results to: " << output_path << "\n";
}

int main()
{
  try {
    CagraConfig optimized;

    const std::string dataset_dtype =
      getenv_or("CUVS_BENCH_CAGRA_DATASET_DTYPE", "float16");
    const std::string query_dtype =
      getenv_or("CUVS_BENCH_CAGRA_QUERY_DTYPE", dataset_dtype);

    if (dataset_dtype != query_dtype) {
      throw std::runtime_error(
        "This standalone C++ translation expects CAGRA dataset/query dtype to match. "
        "The Python binding may dispatch more flexibly.");
    }

    std::vector<CagraConfig> configs = {optimized};
    fs::path output_path = project_root() / "results" / "cagra_optimized_results.csv";

    if (dataset_dtype == "float16") {
      run_cagra_configs_typed<__half>(configs, output_path);
    } else if (dataset_dtype == "float32") {
      run_cagra_configs_typed<float>(configs, output_path);
    } else {
      throw std::runtime_error("Only float16 and float32 are implemented in this C++ translation");
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}