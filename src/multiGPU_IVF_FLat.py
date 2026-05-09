import time
from IVF_flat import N_LISTS, N_PROBES_SWEEP, K, METRIC, DISTRIBUTION_MODE, describe_ivf_params
from config import DISPLAY_TOP_K, MERGE_MODE, SEARCH_MODE
from load_data import load_default_data
from timing_utils import sync_all_cuda_devices
from cuvs.common import MultiGpuResources
from cuvs.neighbors.mg import ivf_flat

def create_multi_gpu_resources(device_ids=None):
    return MultiGpuResources(device_ids=device_ids)

def create_index_params(
    distribution_mode=DISTRIBUTION_MODE,
    n_lists=N_LISTS,
    metric=METRIC,
):
    return ivf_flat.IndexParams(
        distribution_mode=distribution_mode,
        n_lists=n_lists,
        metric=metric
    )

def build_ivf_flat_index(dataset, index_params, resources=None, sync_fn=None, print_info=False):
    if print_info:
        print("Building IVF-Flat index...")
        print("distribution_mode:", index_params.distribution_mode)
        print("n_lists:", index_params.n_lists)
        print("metric:", index_params.metric)
        print("dataset shape:", dataset.shape)
        print("dataset dtype:", dataset.dtype)

    if sync_fn is None:
        sync_fn = resources.sync if resources is not None else sync_all_cuda_devices

    sync_fn()
    build_start = time.perf_counter()

    index = ivf_flat.build(index_params, dataset, resources=resources)

    sync_fn()
    build_end = time.perf_counter()
    build_time = build_end - build_start

    if print_info:
        print(f"IVF-Flat index built in {build_time:.2f} seconds")

    return index, build_time

def create_search_params(
    n_probes,
    search_mode=SEARCH_MODE,
    merge_mode=MERGE_MODE,
):
    return ivf_flat.SearchParams(
        n_probes=n_probes,
        search_mode=search_mode,
        merge_mode=merge_mode,
    )

def search_ivf_flat(index, queries, search_params, k, resources=None, print_info=False):
    if print_info:
        print("\nPerforming IVF-Flat search...")
        print("n probes:", str(search_params.n_probes) + " (how many IVF lists are searched)")
        print("search mode:", search_params.search_mode + " (distribute queries across GPUs to balance load)")
        print("merge mode:", search_params.merge_mode + " (gather partial results on root rank and merge there)")

    distances, neighbors = ivf_flat.search(search_params, index, queries, k, resources=resources)

    if print_info:
        print("Search completed.")
        print("Distances shape:", distances.shape)
        print("Neighbors shape:", neighbors.shape)
        print("First query top 10 neighbors:", neighbors[0, :DISPLAY_TOP_K])
        print("First query top 10 distances:", distances[0, :DISPLAY_TOP_K])

    return distances, neighbors

def main():
    _, dataset, _, queries = load_default_data(print_info=True)
    describe_ivf_params(dataset, print_info=True)

    resources = create_multi_gpu_resources()
    index_params = create_index_params()
    index, _ = build_ivf_flat_index(dataset, index_params, resources=resources, print_info=True)

    search_params = create_search_params(N_PROBES_SWEEP[-1])
    search_ivf_flat(index, queries, search_params, K, resources=resources, print_info=True)

if __name__ == "__main__":
    main()
