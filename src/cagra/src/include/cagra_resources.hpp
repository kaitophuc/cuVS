#pragma once

#include "cagra_common.hpp"
#include "cagra_dlpack.hpp"

#include <cuvs/neighbors/mg_cagra.h>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_id.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/multi_gpu.hpp>
#include <raft/core/resources.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

struct MultiGpuResources {
  struct RankStream {
    int device_id = 0;
    cudaStream_t stream = nullptr;
  };

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

    configure_rank_streams();

    const std::string allocator =
      getenv_or("CUVS_BENCH_CAGRA_DEVICE_ALLOCATOR", "cuda_async");
    if (allocator == "cuda_async") {
      configure_cuda_async_allocator();
      std::cout << "device allocator: cuda_async memory pool\n";
    } else if (allocator == "rmm_pool") {
      const int pool_percent = getenv_int_or("CUVS_BENCH_CAGRA_MEMORY_POOL_PERCENT", 30);
      if (pool_percent > 0) {
        check_cuvs(cuvsMultiGpuResourcesSetMemoryPool(handle, pool_percent),
                   "cuvsMultiGpuResourcesSetMemoryPool");
        bind_workspace_to_current_resources();
        std::cout << "device allocator: RMM pool, " << pool_percent
                  << "% of free memory\n";
      } else {
        std::cout << "device allocator: RMM pool requested but disabled by percent=0\n";
      }
    } else if (allocator == "none") {
      std::cout << "device allocator: cuVS default\n";
    } else {
      throw std::runtime_error(
        "CUVS_BENCH_CAGRA_DEVICE_ALLOCATOR must be 'cuda_async', 'rmm_pool', or 'none'");
    }
  }

  ~MultiGpuResources()
  {
    try {
      sync();
    } catch (...) {
    }

    if (handle != 0) {
      try {
        check_cuvs(cuvsMultiGpuResourcesDestroy(handle), "cuvsMultiGpuResourcesDestroy");
      } catch (...) {
      }
      handle = 0;
    }

    destroy_rank_streams();
    reset_cuda_async_allocator();
  }

  MultiGpuResources(const MultiGpuResources&) = delete;
  MultiGpuResources& operator=(const MultiGpuResources&) = delete;

  void sync() const
  {
    check_cuvs(cuvsStreamSync(handle), "cuvsStreamSync");
    for (const RankStream& rank_stream : rank_streams_) {
      DeviceGuard guard(rank_stream.device_id);
      check_cuda_device(
        cudaStreamSynchronize(rank_stream.stream),
        "sync rank CUDA stream",
        rank_stream.device_id);
    }
  }

private:
  raft::resources& raft_resources() const
  {
    return *reinterpret_cast<raft::resources*>(handle);
  }

  std::vector<raft::resources>& world_resources() const
  {
    return raft::resource::get_multi_gpu_resource(raft_resources());
  }

  void configure_rank_streams()
  {
    if (getenv_or("CUVS_BENCH_CAGRA_NONBLOCKING_STREAMS", "1") == "0") {
      std::cout << "rank CUDA streams: cuVS defaults\n";
      return;
    }

    auto& world = world_resources();
    rank_streams_.reserve(world.size());

    for (int rank = 0; rank < static_cast<int>(world.size()); ++rank) {
      const raft::resources& device_resources =
        raft::resource::set_current_device_to_rank(raft_resources(), rank);
      const int device_id = raft::resource::get_device_id(device_resources);

      cudaStream_t stream = nullptr;
      check_cuda_device(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags rank stream",
        device_id);

      raft::resource::set_cuda_stream(device_resources, rmm::cuda_stream_view(stream));
      rank_streams_.push_back(RankStream{device_id, stream});
    }

    if (!rank_streams_.empty()) {
      check_cuvs(cuvsStreamSet(handle, rank_streams_.front().stream), "cuvsStreamSet");
    }

    std::cout << "rank CUDA streams: nonblocking per GPU\n";
  }

  void configure_cuda_async_allocator()
  {
    auto& world = world_resources();
    async_device_resources_.reserve(world.size());
    async_device_ids_.reserve(world.size());

    for (int rank = 0; rank < static_cast<int>(world.size()); ++rank) {
      const raft::resources& device_resources =
        raft::resource::set_current_device_to_rank(raft_resources(), rank);
      const int device_id = raft::resource::get_device_id(device_resources);

      auto async_resource =
        std::make_shared<rmm::mr::cuda_async_memory_resource>();
      raft::mr::device_resource raft_async_resource{*async_resource};

      rmm::mr::set_per_device_resource(
        rmm::cuda_device_id{device_id},
        raft::mr::device_resource{*async_resource});
      raft::resource::set_workspace_resource(device_resources, raft_async_resource);
      raft::resource::set_large_workspace_resource(device_resources, std::move(raft_async_resource));

      async_device_ids_.push_back(device_id);
      async_device_resources_.push_back(std::move(async_resource));
    }
  }

  void bind_workspace_to_current_resources()
  {
    auto& world = world_resources();
    for (int rank = 0; rank < static_cast<int>(world.size()); ++rank) {
      const raft::resources& device_resources =
        raft::resource::set_current_device_to_rank(raft_resources(), rank);
      raft::resource::set_workspace_to_global_resource(device_resources);
      raft::resource::set_large_workspace_resource(
        device_resources,
        raft::mr::device_resource{rmm::mr::get_current_device_resource_ref()});
    }
  }

  void destroy_rank_streams() noexcept
  {
    for (RankStream& rank_stream : rank_streams_) {
      if (rank_stream.stream == nullptr) continue;
      cudaSetDevice(rank_stream.device_id);
      cudaStreamDestroy(rank_stream.stream);
      rank_stream.stream = nullptr;
    }
    rank_streams_.clear();
  }

  void reset_cuda_async_allocator() noexcept
  {
    for (int device_id : async_device_ids_) {
      cudaSetDevice(device_id);
      rmm::mr::reset_per_device_resource(rmm::cuda_device_id{device_id});
    }
    async_device_ids_.clear();
    async_device_resources_.clear();
  }

  std::vector<RankStream> rank_streams_;
  std::vector<int> async_device_ids_;
  std::vector<std::shared_ptr<rmm::mr::cuda_async_memory_resource>> async_device_resources_;
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

