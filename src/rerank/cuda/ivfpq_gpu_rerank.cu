#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <nvtx3/nvToolsExt.h>
#include <raft/core/device_resources_snmg.hpp>
#include <raft/core/host_mdspan.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

constexpr int kDistanceThreads = 256;
constexpr int kMaxFinalK = 256;
constexpr int kConvertThreads = 256;
constexpr size_t kHalfUploadChunkBytes = static_cast<size_t>(64) * 1024 * 1024;

enum class RequestedStorageMode {
    Float32Auto,
    Float16Resident,
};

enum class ActiveDatasetMode {
    StagedFloat32,
    ResidentFloat32,
    ResidentFloat16,
};

void check_cuda(cudaError_t status, const std::string& what)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(what + ": " + cudaGetErrorString(status));
    }
}

void check_cuda_device(cudaError_t status, const std::string& what, int device_id)
{
    if (status != cudaSuccess) {
        std::ostringstream message;
        message << what << " on CUDA device " << device_id << ": "
                << cudaGetErrorString(status);

        if (status == cudaErrorMemoryAllocation) {
            message << ". MultiGpuExactReranker could not allocate resident dataset shards "
                    << "or rerank buffers; reduce visible devices/index memory pressure, "
                    << "reduce rerank batch size, or use the CPU backend.";
        }

        throw std::runtime_error(message.str());
    }
}

RequestedStorageMode parse_storage_mode(const std::string& storage_dtype)
{
    if (storage_dtype == "float32") {
        return RequestedStorageMode::Float32Auto;
    }
    if (storage_dtype == "float16") {
        return RequestedStorageMode::Float16Resident;
    }

    throw std::invalid_argument(
        "storage_dtype must be either 'float32' or 'float16'");
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

    ~DeviceGuard()
    {
        cudaSetDevice(previous_device_);
    }

private:
    int previous_device_ = 0;
};

class ScopedNvtxRange {
public:
    explicit ScopedNvtxRange(const char* name)
    {
        nvtxRangePushA(name);
    }

    ScopedNvtxRange(const ScopedNvtxRange&) = delete;
    ScopedNvtxRange& operator=(const ScopedNvtxRange&) = delete;

    ~ScopedNvtxRange()
    {
        nvtxRangePop();
    }
};

template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;

    explicit PinnedBuffer(int64_t count)
    {
        allocate(count);
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    PinnedBuffer(PinnedBuffer&& other) noexcept
    {
        ptr_ = other.ptr_;
        count_ = other.count_;
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept
    {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    ~PinnedBuffer()
    {
        release();
    }

    void allocate(int64_t count)
    {
        release();
        count_ = count;
        if (count_ == 0) {
            return;
        }

        check_cuda(
            cudaHostAlloc(
                reinterpret_cast<void**>(&ptr_),
                checked_bytes(count_, sizeof(T), "pinned buffer"),
                cudaHostAllocPortable),
            "cudaHostAlloc pinned buffer");
    }

    void release()
    {
        if (ptr_ != nullptr) {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
    }

    T* data()
    {
        return ptr_;
    }

    const T* data() const
    {
        return ptr_;
    }

private:
    T* ptr_ = nullptr;
    int64_t count_ = 0;
};

struct SlotBuffers {
    cudaStream_t stream = nullptr;
    float* d_queries = nullptr;
    __half* d_queries_half = nullptr;
    int64_t* d_candidates = nullptr;
    float* d_compact_dataset = nullptr;
    int64_t* d_compact_offsets = nullptr;
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

    ~DeviceState()
    {
        release();
    }

    int64_t shard_rows() const
    {
        return shard_end - shard_start;
    }

    void release() noexcept
    {
        int previous_device = 0;
        cudaGetDevice(&previous_device);
        cudaSetDevice(device_id);

        for (auto& slot : slots) {
            if (slot.stream != nullptr) {
                cudaStreamSynchronize(slot.stream);
            }
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
            if (slot.d_queries_half != nullptr) {
                cudaFree(slot.d_queries_half);
                slot.d_queries_half = nullptr;
            }
            if (slot.d_candidates != nullptr) {
                cudaFree(slot.d_candidates);
                slot.d_candidates = nullptr;
            }
            if (slot.d_compact_dataset != nullptr) {
                cudaFree(slot.d_compact_dataset);
                slot.d_compact_dataset = nullptr;
            }
            if (slot.d_compact_offsets != nullptr) {
                cudaFree(slot.d_compact_offsets);
                slot.d_compact_offsets = nullptr;
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
    PinnedBuffer<float> queries;
    PinnedBuffer<__half> queries_half;
    PinnedBuffer<int64_t> candidates;
    std::vector<PinnedBuffer<float>> compact_datasets;
    std::vector<PinnedBuffer<int64_t>> compact_offsets;
    std::vector<int64_t> compact_rows;
    std::vector<PinnedBuffer<float>> partial_distances;
    std::vector<PinnedBuffer<int64_t>> partial_rows;
    int64_t start = 0;
    int64_t batch_queries = 0;
    bool active = false;
};

template <typename T>
void device_malloc(T** ptr, int64_t count, const std::string& what, int device_id)
{
    *ptr = nullptr;
    if (count == 0) {
        return;
    }
    check_cuda_device(
        cudaMalloc(reinterpret_cast<void**>(ptr), checked_bytes(count, sizeof(T), what.c_str())),
        "cudaMalloc " + what,
        device_id);
}

std::vector<int> default_device_ids()
{
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
        throw std::runtime_error("No CUDA devices are visible");
    }

    std::vector<int> ids;
    ids.reserve(static_cast<size_t>(device_count));
    for (int device_id = 0; device_id < device_count; ++device_id) {
        ids.push_back(device_id);
    }
    return ids;
}

std::vector<int> parse_device_ids(const py::object& device_ids)
{
    if (device_ids.is_none()) {
        return default_device_ids();
    }

    std::vector<int> ids = device_ids.cast<std::vector<int>>();
    if (ids.empty()) {
        return default_device_ids();
    }

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");

    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0 || ids[i] >= device_count) {
            throw std::invalid_argument("device_ids contains an invalid CUDA device id");
        }
        for (size_t j = 0; j < i; ++j) {
            if (ids[i] == ids[j]) {
                throw std::invalid_argument("device_ids contains duplicate CUDA device ids");
            }
        }
    }

    return ids;
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
        if (thread_idx < stride) {
            partial_sums[thread_idx] += partial_sums[thread_idx + stride];
        }
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
            const float dataset_value = __half2float(dataset_row[d]);
            const float diff = query_row[d] - dataset_value;
            sum += diff * diff;
        }
    } else {
        sum = (thread_idx == 0) ? INFINITY : 0.0f;
    }

    partial_sums[thread_idx] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_idx < stride) {
            partial_sums[thread_idx] += partial_sums[thread_idx + stride];
        }
        __syncthreads();
    }

    if (thread_idx == 0) {
        candidate_distances[query_idx * candidate_k + candidate_idx] = partial_sums[0];
    }
}

__global__ void compute_candidate_l2_half_query_half_dataset_kernel(
    const __half* __restrict__ dataset_shard,
    int64_t shard_start,
    int64_t shard_end,
    int64_t dim,
    const __half* __restrict__ queries,
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
        const __half* query_row = queries + query_idx * dim;

        for (int64_t d = thread_idx; d < dim; d += blockDim.x) {
            const float diff = __half2float(query_row[d]) - __half2float(dataset_row[d]);
            sum += diff * diff;
        }
    } else {
        sum = (thread_idx == 0) ? INFINITY : 0.0f;
    }

    partial_sums[thread_idx] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_idx < stride) {
            partial_sums[thread_idx] += partial_sums[thread_idx + stride];
        }
        __syncthreads();
    }

    if (thread_idx == 0) {
        candidate_distances[query_idx * candidate_k + candidate_idx] = partial_sums[0];
    }
}

__global__ void compute_candidate_l2_staged_kernel(
    const float* __restrict__ compact_dataset,
    const int64_t* __restrict__ compact_offsets,
    int64_t dim,
    const float* __restrict__ queries,
    int64_t candidate_k,
    float* __restrict__ candidate_distances)
{
    const int64_t candidate_idx = blockIdx.x;
    const int64_t query_idx = blockIdx.y;
    const int thread_idx = threadIdx.x;
    const int64_t compact_offset = compact_offsets[query_idx * candidate_k + candidate_idx];

    extern __shared__ float partial_sums[];
    float sum = 0.0f;

    if (compact_offset >= 0) {
        const float* dataset_row = compact_dataset + compact_offset * dim;
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
        if (thread_idx < stride) {
            partial_sums[thread_idx] += partial_sums[thread_idx + stride];
        }
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
    if (idx < count) {
        output[idx] = __float2half(input[idx]);
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
    if (threadIdx.x != 0) {
        return;
    }

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
            if (already_selected) {
                continue;
            }

            const float distance =
                candidate_distances[query_idx * candidate_k + candidate_idx];
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
        partial_rows[query_idx * final_k + out_idx] = best_candidate_idx >= 0 ? best_row : -1;
    }
}

class MultiGpuExactReranker {
public:
    MultiGpuExactReranker(
        py::array_t<float, py::array::c_style | py::array::forcecast> dataset,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> dataset_ids,
        int64_t final_k,
        int64_t candidate_k,
        int64_t batch_size,
        py::object device_ids,
        const std::string& storage_dtype)
        : final_k_(final_k),
          candidate_k_(candidate_k),
          batch_size_(batch_size),
          requested_storage_mode_(parse_storage_mode(storage_dtype)),
          device_ids_(parse_device_ids(device_ids))
    {
        if (dataset.ndim() != 2) {
            throw std::invalid_argument("dataset must be 2D");
        }
        if (dataset_ids.ndim() != 1) {
            throw std::invalid_argument("dataset_ids must be 1D");
        }

        n_rows_ = dataset.shape(0);
        dim_ = dataset.shape(1);

        if (n_rows_ <= 0) {
            throw std::invalid_argument("dataset must have at least one row");
        }
        if (dim_ <= 0) {
            throw std::invalid_argument("dataset dimension must be positive");
        }
        if (dataset_ids.shape(0) != n_rows_) {
            throw std::invalid_argument("dataset_ids length must match dataset rows");
        }
        if (candidate_k_ <= 0) {
            throw std::invalid_argument("candidate_k must be positive");
        }
        if (final_k_ <= 0) {
            throw std::invalid_argument("final_k must be positive");
        }
        if (final_k_ > candidate_k_) {
            throw std::invalid_argument("final_k cannot be larger than candidate_k");
        }
        if (final_k_ > kMaxFinalK) {
            throw std::invalid_argument("final_k is larger than the CUDA local top-k limit");
        }
        if (batch_size_ <= 0) {
            throw std::invalid_argument("batch_size must be positive");
        }

        dataset_owner_ = dataset;
        host_dataset_ptr_ = dataset_owner_.data();
        dataset_ids_.assign(dataset_ids.data(), dataset_ids.data() + n_rows_);
        if (requested_storage_mode_ == RequestedStorageMode::Float16Resident) {
            if (!can_use_resident_dataset(sizeof(__half))) {
                throw std::runtime_error(
                    "float16 resident rerank dataset does not fit in visible GPU memory. "
                    "Reduce visible devices/index memory pressure, reduce rerank batch size, "
                    "or use CUVS_BENCH_IVFPQ_RERANK_STORAGE_DTYPE=float32.");
            }
            active_dataset_mode_ = ActiveDatasetMode::ResidentFloat16;
        } else {
            active_dataset_mode_ = can_use_resident_dataset(sizeof(float))
                ? ActiveDatasetMode::ResidentFloat32
                : ActiveDatasetMode::StagedFloat32;
        }

        const int64_t device_count = static_cast<int64_t>(device_ids_.size());

        {
            py::gil_scoped_release release;
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
    }

    py::tuple rerank(
        py::array_t<float, py::array::c_style | py::array::forcecast> queries,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> candidate_neighbors)
    {
        if (queries.ndim() != 2) {
            throw std::invalid_argument("queries must be 2D");
        }
        if (candidate_neighbors.ndim() != 2) {
            throw std::invalid_argument("candidate_neighbors must be 2D");
        }

        const int64_t n_queries = queries.shape(0);
        const int64_t query_dim = queries.shape(1);
        const int64_t candidate_queries = candidate_neighbors.shape(0);
        const int64_t candidate_k = candidate_neighbors.shape(1);

        if (query_dim != dim_) {
            throw std::invalid_argument("queries dimension must match dataset dimension");
        }
        if (candidate_queries != n_queries) {
            throw std::invalid_argument("candidate_neighbors rows must match queries rows");
        }
        if (candidate_k != candidate_k_) {
            throw std::invalid_argument("candidate_neighbors columns must match candidate_k");
        }

        py::array_t<float> out_distances({n_queries, final_k_});
        py::array_t<int64_t> out_neighbors({n_queries, final_k_});

        const float* queries_ptr = queries.data();
        const int64_t* candidates_ptr = candidate_neighbors.data();
        float* out_distances_ptr = out_distances.mutable_data();
        int64_t* out_neighbors_ptr = out_neighbors.mutable_data();

        {
            py::gil_scoped_release release;
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

        return py::make_tuple(out_distances, out_neighbors);
    }

    void rerank_raw_half_queries(
        const __half* queries_ptr,
        const int64_t* candidates_ptr,
        int64_t n_queries,
        float* out_distances_ptr,
        int64_t* out_neighbors_ptr)
    {
        if (active_dataset_mode_ != ActiveDatasetMode::ResidentFloat16) {
            throw std::runtime_error(
                "rerank_raw_half_queries requires resident_float16 dataset mode");
        }
        if (n_queries < 0) {
            throw std::invalid_argument("n_queries cannot be negative");
        }

        validate_candidates(candidates_ptr, n_queries);

        int next_slot = 0;
        try {
            for (int64_t start = 0; start < n_queries; start += batch_size_) {
                if (host_slots_[next_slot].active) {
                    synchronize_slot(next_slot, out_distances_ptr, out_neighbors_ptr);
                }

                const int64_t batch_queries = std::min(batch_size_, n_queries - start);
                launch_batch_half_queries(next_slot, start, batch_queries, queries_ptr, candidates_ptr);
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
        if (active_dataset_mode_ == ActiveDatasetMode::ResidentFloat16) {
            return "resident_float16";
        }
        if (active_dataset_mode_ == ActiveDatasetMode::ResidentFloat32) {
            return "resident_float32";
        }
        return "staged_float32";
    }

private:
    bool uses_staged_dataset() const
    {
        return active_dataset_mode_ == ActiveDatasetMode::StagedFloat32;
    }

    bool can_use_resident_dataset(size_t dataset_item_size) const
    {
        constexpr size_t reserve_bytes = static_cast<size_t>(512) * 1024 * 1024;
        const int64_t device_count = static_cast<int64_t>(device_ids_.size());
        const int64_t query_values = checked_mul(batch_size_, dim_, "resident query buffer");
        const int64_t candidate_values =
            checked_mul(batch_size_, candidate_k_, "resident candidate buffer");
        const int64_t output_values = checked_mul(batch_size_, final_k_, "resident output buffer");

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
            const size_t shard_bytes = checked_bytes(
                shard_values,
                dataset_item_size,
                "dataset shard");
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
        const int64_t total_values = checked_mul(state.shard_rows(), dim_, "float16 dataset shard");
        if (total_values == 0) {
            return;
        }

        const int64_t max_chunk_values = static_cast<int64_t>(
            std::max<size_t>(1, kHalfUploadChunkBytes / sizeof(float)));
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

                const int blocks = static_cast<int>(
                    (values_this_chunk + kConvertThreads - 1) / kConvertThreads);
                float_to_half_kernel<<<blocks, kConvertThreads, 0, stream>>>(
                    d_upload,
                    state.d_dataset_half + offset,
                    values_this_chunk);
                check_cuda_device(
                    cudaGetLastError(),
                    "launch float_to_half_kernel",
                    device_id);
            }

            check_cuda_device(
                cudaStreamSynchronize(stream),
                "sync float16 upload stream",
                device_id);
        } catch (...) {
            if (d_upload != nullptr) {
                cudaFree(d_upload);
            }
            if (stream != nullptr) {
                cudaStreamDestroy(stream);
            }
            throw;
        }

        check_cuda_device(cudaFree(d_upload), "cudaFree float16 upload staging", device_id);
        check_cuda_device(
            cudaStreamDestroy(stream),
            "cudaStreamDestroy float16 upload",
            device_id);
    }

    void allocate_slots(DeviceState& state)
    {
        const int device_id = state.device_id;
        const int64_t query_values = checked_mul(batch_size_, dim_, "query buffer");
        const int64_t candidate_values =
            checked_mul(batch_size_, candidate_k_, "candidate buffer");
        const int64_t output_values = checked_mul(batch_size_, final_k_, "output buffer");

        for (auto& slot : state.slots) {
            check_cuda_device(
                cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking),
                "cudaStreamCreateWithFlags",
                device_id);

            device_malloc(&slot.d_queries, query_values, "queries", device_id);
            if (active_dataset_mode_ == ActiveDatasetMode::ResidentFloat16) {
                device_malloc(&slot.d_queries_half, query_values, "half queries", device_id);
            }
            device_malloc(&slot.d_candidates, candidate_values, "candidates", device_id);
            if (uses_staged_dataset()) {
                device_malloc(
                    &slot.d_compact_dataset,
                    checked_mul(candidate_values, dim_, "compact dataset"),
                    "compact dataset",
                    device_id);
                device_malloc(
                    &slot.d_compact_offsets,
                    candidate_values,
                    "compact offsets",
                    device_id);
            }
            device_malloc(
                &slot.d_candidate_distances,
                candidate_values,
                "candidate distances",
                device_id);
            device_malloc(
                &slot.d_partial_distances,
                output_values,
                "partial distances",
                device_id);
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
            slot.queries.allocate(query_values);
            if (active_dataset_mode_ == ActiveDatasetMode::ResidentFloat16) {
                slot.queries_half.allocate(query_values);
            }
            slot.candidates.allocate(candidate_values);
            slot.compact_rows.assign(devices_.size(), 0);
            slot.compact_datasets.reserve(devices_.size());
            slot.compact_offsets.reserve(devices_.size());
            slot.partial_distances.reserve(devices_.size());
            slot.partial_rows.reserve(devices_.size());

            for (size_t device_index = 0; device_index < devices_.size(); ++device_index) {
                if (uses_staged_dataset()) {
                    slot.compact_datasets.emplace_back(
                        checked_mul(candidate_values, dim_, "host compact dataset"));
                    slot.compact_offsets.emplace_back(candidate_values);
                }
                slot.partial_distances.emplace_back(output_values);
                slot.partial_rows.emplace_back(output_values);
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

    size_t owner_for_row(int64_t row) const
    {
        size_t owner = static_cast<size_t>(
            (row * static_cast<int64_t>(devices_.size())) / n_rows_);
        if (owner >= devices_.size()) {
            owner = devices_.size() - 1;
        }
        return owner;
    }

    void fill_staged_compact_buffers(HostSlot& host_slot, int64_t batch_queries)
    {
        const int64_t candidate_values =
            checked_mul(batch_queries, candidate_k_, "candidate values");

        for (size_t device_index = 0; device_index < devices_.size(); ++device_index) {
            host_slot.compact_rows[device_index] = 0;
            std::fill(
                host_slot.compact_offsets[device_index].data(),
                host_slot.compact_offsets[device_index].data() + candidate_values,
                int64_t{-1});
        }

        for (int64_t local_q = 0; local_q < batch_queries; ++local_q) {
            for (int64_t candidate_idx = 0; candidate_idx < candidate_k_; ++candidate_idx) {
                const int64_t flat_idx = local_q * candidate_k_ + candidate_idx;
                const int64_t row = host_slot.candidates.data()[flat_idx];
                const size_t owner = owner_for_row(row);
                const int64_t compact_row = host_slot.compact_rows[owner]++;

                host_slot.compact_offsets[owner].data()[flat_idx] = compact_row;
                std::memcpy(
                    host_slot.compact_datasets[owner].data() + compact_row * dim_,
                    host_dataset_ptr_ + row * dim_,
                    checked_bytes(dim_, sizeof(float), "compact row copy"));
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

        if (uses_staged_dataset()) {
            fill_staged_compact_buffers(host_slot, batch_queries);
        }

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

            if (uses_staged_dataset()) {
                check_cuda_device(
                    cudaMemcpyAsync(
                        slot.d_compact_offsets,
                        host_slot.compact_offsets[device_index].data(),
                        checked_bytes(candidate_values, sizeof(int64_t), "compact offset H2D"),
                        cudaMemcpyHostToDevice,
                        slot.stream),
                    "copy compact offsets to device",
                    device.device_id);

                const int64_t compact_values =
                    checked_mul(host_slot.compact_rows[device_index], dim_, "compact values");
                if (compact_values > 0) {
                    check_cuda_device(
                        cudaMemcpyAsync(
                            slot.d_compact_dataset,
                            host_slot.compact_datasets[device_index].data(),
                            checked_bytes(compact_values, sizeof(float), "compact dataset H2D"),
                            cudaMemcpyHostToDevice,
                            slot.stream),
                        "copy compact dataset to device",
                        device.device_id);
                }
            }

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
                    cudaGetLastError(),
                    "launch compute_candidate_l2_half_kernel",
                    device.device_id);
            } else {
                compute_candidate_l2_staged_kernel<<<
                    distance_grid,
                    kDistanceThreads,
                    sizeof(float) * kDistanceThreads,
                    slot.stream>>>(
                    slot.d_compact_dataset,
                    slot.d_compact_offsets,
                    dim_,
                    slot.d_queries,
                    candidate_k_,
                    slot.d_candidate_distances);
                check_cuda_device(
                    cudaGetLastError(),
                    "launch compute_candidate_l2_staged_kernel",
                    device.device_id);
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

    void launch_batch_half_queries(
        int slot_index,
        int64_t start,
        int64_t batch_queries,
        const __half* queries_ptr,
        const int64_t* candidates_ptr)
    {
        HostSlot& host_slot = host_slots_[slot_index];
        host_slot.start = start;
        host_slot.batch_queries = batch_queries;
        host_slot.active = false;

        const int64_t query_values = checked_mul(batch_queries, dim_, "half query values");
        const int64_t candidate_values =
            checked_mul(batch_queries, candidate_k_, "candidate values");
        const int64_t output_values = checked_mul(batch_queries, final_k_, "output values");

        std::memcpy(
            host_slot.queries_half.data(),
            queries_ptr + start * dim_,
            checked_bytes(query_values, sizeof(__half), "half query copy"));
        std::memcpy(
            host_slot.candidates.data(),
            candidates_ptr + start * candidate_k_,
            checked_bytes(candidate_values, sizeof(int64_t), "candidate copy"));

        for (size_t device_index = 0; device_index < devices_.size(); ++device_index) {
            DeviceState& device = *devices_[device_index];
            DeviceGuard guard(device.device_id);
            SlotBuffers& slot = device.slots[slot_index];

            {
                ScopedNvtxRange range("session.rerank_h2d");
                check_cuda_device(
                    cudaMemcpyAsync(
                        slot.d_queries_half,
                        host_slot.queries_half.data(),
                        checked_bytes(query_values, sizeof(__half), "half query H2D"),
                        cudaMemcpyHostToDevice,
                        slot.stream),
                    "copy half queries to device",
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
            }

            {
                ScopedNvtxRange range("session.rerank_kernel");
                const dim3 distance_grid(
                    static_cast<unsigned int>(candidate_k_),
                    static_cast<unsigned int>(batch_queries));
                compute_candidate_l2_half_query_half_dataset_kernel<<<
                    distance_grid,
                    kDistanceThreads,
                    sizeof(float) * kDistanceThreads,
                    slot.stream>>>(
                    device.d_dataset_half,
                    device.shard_start,
                    device.shard_end,
                    dim_,
                    slot.d_queries_half,
                    slot.d_candidates,
                    candidate_k_,
                    slot.d_candidate_distances);
                check_cuda_device(
                    cudaGetLastError(),
                    "launch compute_candidate_l2_half_query_half_dataset_kernel",
                    device.device_id);

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
            }

            {
                ScopedNvtxRange range("session.rerank_d2h");
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
        }

        host_slot.active = true;
    }

    void synchronize_slot(
        int slot_index,
        float* out_distances_ptr,
        int64_t* out_neighbors_ptr)
    {
        HostSlot& host_slot = host_slots_[slot_index];
        if (!host_slot.active) {
            return;
        }

        for (const auto& device : devices_) {
            DeviceGuard guard(device->device_id);
            check_cuda_device(
                cudaStreamSynchronize(device->slots[slot_index].stream),
                "sync rerank stream",
                device->device_id);
        }

        {
            ScopedNvtxRange range("session.merge");
            merge_slot(host_slot, out_distances_ptr, out_neighbors_ptr);
        }
        host_slot.active = false;
    }

    void synchronize_active_slots_no_merge()
    {
        for (auto& host_slot : host_slots_) {
            if (!host_slot.active) {
                continue;
            }
            const int slot_index = (&host_slot == &host_slots_[0]) ? 0 : 1;
            for (const auto& device : devices_) {
                DeviceGuard guard(device->device_id);
                cudaStreamSynchronize(device->slots[slot_index].stream);
            }
            host_slot.active = false;
        }
    }

    bool row_already_selected(const int64_t* out_neighbors_ptr, int64_t query_offset, int64_t out_idx, int64_t dataset_id) const
    {
        for (int64_t previous = 0; previous < out_idx; ++previous) {
            if (out_neighbors_ptr[query_offset + previous] == dataset_id) {
                return true;
            }
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
                        if (row < 0) {
                            continue;
                        }

                        const int64_t dataset_id = dataset_ids_[static_cast<size_t>(row)];
                        if (row_already_selected(out_neighbors_ptr, output_base, out_idx, dataset_id)) {
                            continue;
                        }

                        const float distance = partial_distances[candidate_idx];
                        if (distance < best_distance ||
                            (distance == best_distance && row < best_row)) {
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
    RequestedStorageMode requested_storage_mode_ = RequestedStorageMode::Float32Auto;
    ActiveDatasetMode active_dataset_mode_ = ActiveDatasetMode::StagedFloat32;
    py::array_t<float, py::array::c_style | py::array::forcecast> dataset_owner_;
    const float* host_dataset_ptr_ = nullptr;
    std::vector<int> device_ids_;
    std::vector<int64_t> dataset_ids_;
    std::vector<std::unique_ptr<DeviceState>> devices_;
    HostSlot host_slots_[2];
};

class IvfPqSearchRerankSession {
public:
    using IndexT =
        cuvs::neighbors::mg_index<cuvs::neighbors::ivf_pq::index<int64_t>, half, int64_t>;

    IvfPqSearchRerankSession(
        py::array index_dataset,
        py::array_t<float, py::array::c_style | py::array::forcecast> rerank_dataset,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> dataset_ids,
        int64_t final_k,
        int64_t candidate_k,
        int64_t batch_size,
        py::object device_ids,
        int64_t n_lists,
        int64_t pq_bits,
        int64_t pq_dim,
        int64_t n_probes)
        : device_ids_(parse_device_ids(device_ids)),
          clique_(device_ids_),
          final_k_(final_k),
          candidate_k_(candidate_k),
          batch_size_(batch_size)
    {
        py::buffer_info index_info = index_dataset.request();
        if (index_info.ndim != 2) {
            throw std::invalid_argument("index_dataset must be 2D");
        }
        if (index_info.itemsize != sizeof(half)) {
            throw std::invalid_argument(
                "IvfPqSearchRerankSession currently requires float16 index_dataset");
        }
        if (rerank_dataset.ndim() != 2) {
            throw std::invalid_argument("rerank_dataset must be 2D");
        }
        if (dataset_ids.ndim() != 1) {
            throw std::invalid_argument("dataset_ids must be 1D");
        }

        n_rows_ = index_info.shape[0];
        dim_ = index_info.shape[1];
        if (n_rows_ <= 0 || dim_ <= 0) {
            throw std::invalid_argument("index_dataset must have positive shape");
        }
        if (rerank_dataset.shape(0) != n_rows_ || rerank_dataset.shape(1) != dim_) {
            throw std::invalid_argument("rerank_dataset shape must match index_dataset shape");
        }
        if (dataset_ids.shape(0) != n_rows_) {
            throw std::invalid_argument("dataset_ids length must match dataset rows");
        }
        if (final_k_ <= 0 || candidate_k_ <= 0 || final_k_ > candidate_k_) {
            throw std::invalid_argument("final_k and candidate_k must satisfy 0 < final_k <= candidate_k");
        }

        index_dataset_owner_ = index_dataset;

        configure_search_params(n_probes);

        auto index_params = make_index_params(n_lists, pq_bits, pq_dim);
        const half* index_ptr = reinterpret_cast<const half*>(index_info.ptr);
        auto index_view = raft::make_host_matrix_view<const half, int64_t, raft::row_major>(
            index_ptr,
            n_rows_,
            dim_);

        {
            py::gil_scoped_release release;
            auto built_index = cuvs::neighbors::ivf_pq::build(clique_, index_params, index_view);
            index_ = std::make_unique<IndexT>(std::move(built_index));
            sync_session_devices();
        }

        reranker_ = std::make_unique<MultiGpuExactReranker>(
            rerank_dataset,
            dataset_ids,
            final_k_,
            candidate_k_,
            batch_size_,
            py::cast(device_ids_),
            "float16");
    }

    py::tuple search_rerank(py::array queries)
    {
        ScopedNvtxRange total_range("session.search_rerank");

        py::buffer_info query_info = queries.request();
        if (query_info.ndim != 2) {
            throw std::invalid_argument("queries must be 2D");
        }
        if (query_info.itemsize != sizeof(half)) {
            throw std::invalid_argument(
                "IvfPqSearchRerankSession.search_rerank currently requires float16 queries");
        }

        const int64_t n_queries = query_info.shape[0];
        const int64_t query_dim = query_info.shape[1];
        if (query_dim != dim_) {
            throw std::invalid_argument("queries dimension must match index dimension");
        }

        py::array_t<float> out_distances({n_queries, final_k_});
        py::array_t<int64_t> out_neighbors({n_queries, final_k_});

        const half* query_ptr = reinterpret_cast<const half*>(query_info.ptr);
        float* out_distances_ptr = out_distances.mutable_data();
        int64_t* out_neighbors_ptr = out_neighbors.mutable_data();

        {
            py::gil_scoped_release release;
            ensure_search_buffers(n_queries);

            {
                ScopedNvtxRange search_range("session.search");
                auto query_view = raft::make_host_matrix_view<const half, int64_t, raft::row_major>(
                    query_ptr,
                    n_queries,
                    dim_);
                auto neighbor_view = raft::make_host_matrix_view<int64_t, int64_t, raft::row_major>(
                    search_neighbors_.data(),
                    n_queries,
                    candidate_k_);
                auto distance_view = raft::make_host_matrix_view<float, int64_t, raft::row_major>(
                    search_distances_.data(),
                    n_queries,
                    candidate_k_);

                cuvs::neighbors::ivf_pq::search(
                    clique_,
                    *index_,
                    search_params_,
                    query_view,
                    neighbor_view,
                    distance_view);
                sync_session_devices();
            }

            {
                ScopedNvtxRange rerank_range("session.rerank");
                reranker_->rerank_raw_half_queries(
                    query_ptr,
                    search_neighbors_.data(),
                    n_queries,
                    out_distances_ptr,
                    out_neighbors_ptr);
            }
        }

        return py::make_tuple(out_distances, out_neighbors);
    }

    void set_n_probes(int64_t n_probes)
    {
        if (n_probes <= 0) {
            throw std::invalid_argument("n_probes must be positive");
        }
        search_params_.n_probes = static_cast<uint32_t>(n_probes);
    }

    std::string mode() const
    {
        return "cuvs_cpp_search_resident_float16_rerank";
    }

private:
    static cuvs::neighbors::mg_index_params<cuvs::neighbors::ivf_pq::index_params>
    make_index_params(int64_t n_lists, int64_t pq_bits, int64_t pq_dim)
    {
        if (n_lists <= 0 || pq_bits <= 0 || pq_dim <= 0) {
            throw std::invalid_argument("n_lists, pq_bits, and pq_dim must be positive");
        }

        cuvs::neighbors::mg_index_params<cuvs::neighbors::ivf_pq::index_params> params;
        params.mode = cuvs::neighbors::SHARDED;
        params.metric = cuvs::distance::DistanceType::L2Expanded;
        params.n_lists = static_cast<uint32_t>(n_lists);
        params.pq_bits = static_cast<uint32_t>(pq_bits);
        params.pq_dim = static_cast<uint32_t>(pq_dim);
        params.kmeans_n_iters = 20;
        params.kmeans_trainset_fraction = 0.5;
        params.codebook_kind = cuvs::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE;
        params.codes_layout = cuvs::neighbors::ivf_pq::list_layout::INTERLEAVED;
        params.force_random_rotation = false;
        params.conservative_memory_allocation = true;
        params.max_train_points_per_pq_code = 256;
        return params;
    }

    void configure_search_params(int64_t n_probes)
    {
        if (n_probes <= 0) {
            throw std::invalid_argument("n_probes must be positive");
        }
        search_params_.n_probes = static_cast<uint32_t>(n_probes);
        search_params_.search_mode = cuvs::neighbors::LOAD_BALANCER;
        search_params_.merge_mode = cuvs::neighbors::MERGE_ON_ROOT_RANK;
        search_params_.n_rows_per_batch = 1000;
        search_params_.lut_dtype = CUDA_R_16F;
        search_params_.internal_distance_dtype = CUDA_R_16F;
        search_params_.coarse_search_dtype = CUDA_R_16F;
        search_params_.max_internal_batch_size = 4096;
    }

    void ensure_search_buffers(int64_t n_queries)
    {
        if (n_queries <= search_buffer_capacity_queries_) {
            return;
        }
        const int64_t values = checked_mul(n_queries, candidate_k_, "session search output");
        search_neighbors_.allocate(values);
        search_distances_.allocate(values);
        search_buffer_capacity_queries_ = n_queries;
    }

    void sync_session_devices() const
    {
        int previous_device = 0;
        check_cuda(cudaGetDevice(&previous_device), "cudaGetDevice");
        for (int device_id : device_ids_) {
            check_cuda(cudaSetDevice(device_id), "cudaSetDevice");
            check_cuda_device(cudaDeviceSynchronize(), "sync session device", device_id);
        }
        check_cuda(cudaSetDevice(previous_device), "cudaSetDevice");
    }

    std::vector<int> device_ids_;
    raft::device_resources_snmg clique_;
    int64_t n_rows_ = 0;
    int64_t dim_ = 0;
    int64_t final_k_ = 0;
    int64_t candidate_k_ = 0;
    int64_t batch_size_ = 0;
    int64_t search_buffer_capacity_queries_ = 0;
    py::array index_dataset_owner_;
    cuvs::neighbors::mg_search_params<cuvs::neighbors::ivf_pq::search_params> search_params_;
    std::unique_ptr<IndexT> index_;
    std::unique_ptr<MultiGpuExactReranker> reranker_;
    PinnedBuffer<int64_t> search_neighbors_;
    PinnedBuffer<float> search_distances_;
};

py::tuple rerank_ivf_pq_candidates_exact_l2_gpu(
    py::array_t<float, py::array::c_style | py::array::forcecast> dataset,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> dataset_ids,
    py::array_t<float, py::array::c_style | py::array::forcecast> queries,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> candidate_neighbors,
    int64_t final_k,
    int64_t batch_size,
    int device_id)
{
    if (candidate_neighbors.ndim() != 2) {
        throw std::invalid_argument("candidate_neighbors must be 2D");
    }

    py::list device_ids;
    device_ids.append(device_id);

    MultiGpuExactReranker reranker(
        dataset,
        dataset_ids,
        final_k,
        candidate_neighbors.shape(1),
        batch_size,
        device_ids,
        "float32");

    return reranker.rerank(queries, candidate_neighbors);
}

}  // namespace

PYBIND11_MODULE(ivfpq_gpu_rerank, m)
{
    m.doc() = "CUDA exact rerank for multi-GPU IVF-PQ candidates";

    py::class_<MultiGpuExactReranker>(m, "MultiGpuExactReranker")
        .def(
            py::init<
                py::array_t<float, py::array::c_style | py::array::forcecast>,
                py::array_t<int64_t, py::array::c_style | py::array::forcecast>,
                int64_t,
                int64_t,
                int64_t,
                py::object,
                std::string>(),
            py::arg("dataset"),
            py::arg("dataset_ids"),
            py::arg("final_k"),
            py::arg("candidate_k"),
            py::arg("batch_size") = 512,
            py::arg("device_ids") = py::none(),
            py::arg("storage_dtype") = "float32")
        .def_property_readonly("mode", &MultiGpuExactReranker::mode)
        .def("rerank", &MultiGpuExactReranker::rerank, py::arg("queries"), py::arg("candidate_neighbors"));

    py::class_<IvfPqSearchRerankSession>(m, "IvfPqSearchRerankSession")
        .def(
            py::init<
                py::array,
                py::array_t<float, py::array::c_style | py::array::forcecast>,
                py::array_t<int64_t, py::array::c_style | py::array::forcecast>,
                int64_t,
                int64_t,
                int64_t,
                py::object,
                int64_t,
                int64_t,
                int64_t,
                int64_t>(),
            py::arg("index_dataset"),
            py::arg("rerank_dataset"),
            py::arg("dataset_ids"),
            py::arg("final_k"),
            py::arg("candidate_k"),
            py::arg("batch_size") = 512,
            py::arg("device_ids") = py::none(),
            py::arg("n_lists") = 4096,
            py::arg("pq_bits") = 4,
            py::arg("pq_dim") = 384,
            py::arg("n_probes") = 32)
        .def_property_readonly("mode", &IvfPqSearchRerankSession::mode)
        .def("set_n_probes", &IvfPqSearchRerankSession::set_n_probes, py::arg("n_probes"))
        .def("search_rerank", &IvfPqSearchRerankSession::search_rerank, py::arg("queries"));

    m.def(
        "rerank_ivf_pq_candidates_exact_l2_gpu",
        &rerank_ivf_pq_candidates_exact_l2_gpu,
        py::arg("dataset"),
        py::arg("dataset_ids"),
        py::arg("queries"),
        py::arg("candidate_neighbors"),
        py::arg("final_k"),
        py::arg("batch_size") = 512,
        py::arg("device_id") = 0);
}
