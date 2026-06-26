#include "cagra_reranker.hpp"

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


class MultiGpuExactReranker::Impl {
public:
  Impl(
    const LoadedData& data,
    int64_t final_k,
    int64_t candidate_k,
    int64_t batch_size,
    const float* pinned_float_dataset,
    const __half* pinned_half_dataset,
    const std::optional<std::vector<int>>& device_ids,
    const std::string& storage_dtype)
    : final_k_(final_k),
      candidate_k_(candidate_k),
      batch_size_(batch_size),
      requested_storage_mode_(parse_storage_mode(storage_dtype)),
      pinned_float_dataset_(pinned_float_dataset),
      pinned_half_dataset_(pinned_half_dataset),
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
      if (pinned_half_dataset_ == nullptr) {
        throw std::invalid_argument(
          "float16 rerank storage requires a pinned float16 host dataset. "
          "Set CUVS_BENCH_CAGRA_DATASET_DTYPE=float16.");
      }
      if (!can_use_resident_dataset(sizeof(__half))) {
        throw std::runtime_error(
          "float16 resident rerank dataset does not fit in visible GPU memory. "
          "Reduce visible devices/index memory pressure or reduce rerank batch size.");
      }
      active_dataset_mode_ = ActiveDatasetMode::ResidentFloat16;
    } else {
      if (pinned_float_dataset_ == nullptr) {
        throw std::invalid_argument(
          "float32 rerank storage requires a pinned float32 host dataset. "
          "Set CUVS_BENCH_CAGRA_DATASET_DTYPE=float32.");
      }
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

      } else if (active_dataset_mode_ == ActiveDatasetMode::ResidentFloat16) {
        device_malloc(
          &state->d_dataset_half,
          checked_mul(state->shard_rows(), dim_, "dataset shard"),
          "float16 dataset shard",
          device_id);
      }

      allocate_slots(*state);
      devices_.push_back(std::move(state));
    }

    upload_dataset_shards();
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
      DeviceGuard guard(device_id);
      size_t free_bytes = 0;
      size_t total_bytes = 0;
      check_cuda_device(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo", device_id);

      if (shard_bytes + slot_bytes + reserve_bytes > free_bytes) {
        return false;
      }
    }

    return true;
  }

  void upload_dataset_shards()
  {
    std::vector<cudaStream_t> upload_streams(devices_.size(), nullptr);

    try {
      for (size_t device_index = 0; device_index < devices_.size(); ++device_index) {
        DeviceState& state = *devices_[device_index];
        if (state.shard_rows() == 0) continue;

        DeviceGuard guard(state.device_id);
        check_cuda_device(
          cudaStreamCreateWithFlags(&upload_streams[device_index], cudaStreamNonBlocking),
          "cudaStreamCreateWithFlags dataset upload",
          state.device_id);

        const int64_t shard_values =
          checked_mul(state.shard_rows(), dim_, "dataset shard");
        if (active_dataset_mode_ == ActiveDatasetMode::ResidentFloat16) {
          check_cuda_device(
            cudaMemcpyAsync(
              state.d_dataset_half,
              pinned_half_dataset_ + state.shard_start * dim_,
              checked_bytes(shard_values, sizeof(__half), "float16 dataset shard"),
              cudaMemcpyHostToDevice,
              upload_streams[device_index]),
            "copy pinned float16 dataset shard to device",
            state.device_id);
        } else {
          check_cuda_device(
            cudaMemcpyAsync(
              state.d_dataset_float,
              pinned_float_dataset_ + state.shard_start * dim_,
              checked_bytes(shard_values, sizeof(float), "float32 dataset shard"),
              cudaMemcpyHostToDevice,
              upload_streams[device_index]),
            "copy pinned float32 dataset shard to device",
            state.device_id);
        }
      }

      for (size_t device_index = 0; device_index < devices_.size(); ++device_index) {
        if (upload_streams[device_index] == nullptr) continue;
        DeviceState& state = *devices_[device_index];
        DeviceGuard guard(state.device_id);
        check_cuda_device(
          cudaStreamSynchronize(upload_streams[device_index]),
          "sync dataset upload stream",
          state.device_id);
      }
    } catch (...) {
      for (size_t device_index = 0; device_index < devices_.size(); ++device_index) {
        if (upload_streams[device_index] == nullptr) continue;
        DeviceState& state = *devices_[device_index];
        DeviceGuard guard(state.device_id);
        cudaStreamDestroy(upload_streams[device_index]);
      }
      throw;
    }

    for (size_t device_index = 0; device_index < devices_.size(); ++device_index) {
      if (upload_streams[device_index] == nullptr) continue;
      DeviceState& state = *devices_[device_index];
      DeviceGuard guard(state.device_id);
      check_cuda_device(
        cudaStreamDestroy(upload_streams[device_index]),
        "cudaStreamDestroy dataset upload",
        state.device_id);
    }
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
  const float* pinned_float_dataset_ = nullptr;
  const __half* pinned_half_dataset_ = nullptr;
  std::vector<int> device_ids_;
  std::vector<int64_t> dataset_ids_;
  std::vector<std::unique_ptr<DeviceState>> devices_;
  HostSlot host_slots_[2];
};


MultiGpuExactReranker::MultiGpuExactReranker(
  const LoadedData& data,
  int64_t final_k,
  int64_t candidate_k,
  int64_t batch_size,
  const float* pinned_float_dataset,
  const __half* pinned_half_dataset,
  const std::optional<std::vector<int>>& device_ids,
  const std::string& storage_dtype)
  : impl_(std::make_unique<Impl>(
      data,
      final_k,
      candidate_k,
      batch_size,
      pinned_float_dataset,
      pinned_half_dataset,
      device_ids,
      storage_dtype))
{
}

MultiGpuExactReranker::~MultiGpuExactReranker() = default;

void MultiGpuExactReranker::rerank(
  const float* queries_ptr,
  const int64_t* candidates_ptr,
  int64_t n_queries,
  float* out_distances_ptr,
  int64_t* out_neighbors_ptr)
{
  impl_->rerank(queries_ptr, candidates_ptr, n_queries, out_distances_ptr, out_neighbors_ptr);
}

std::string MultiGpuExactReranker::mode() const
{
  return impl_->mode();
}

std::unique_ptr<MultiGpuExactReranker> create_cagra_exact_reranker(
  const LoadedData& data,
  int candidate_k,
  int batch_size,
  const float* pinned_float_dataset,
  const __half* pinned_half_dataset,
  const std::optional<std::vector<int>>& device_ids,
  const std::string& requested_storage_dtype,
  std::string* active_storage_dtype)
{
  auto reranker = std::make_unique<MultiGpuExactReranker>(
    data,
    K,
    candidate_k,
    batch_size,
    pinned_float_dataset,
    pinned_half_dataset,
    device_ids,
    requested_storage_dtype);
  if (active_storage_dtype != nullptr) *active_storage_dtype = requested_storage_dtype;
  return reranker;
}
