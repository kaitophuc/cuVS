import time
from config import (
    DISPLAY_TOP_K,
    DISTRIBUTION_MODE,
    MERGE_MODE,
    METRIC,
    SEARCH_MODE,
)
from timing_utils import sync_all_cuda_devices
from cuvs.common import MultiGpuResources
from cuvs.neighbors.mg import cagra

def create_multi_gpu_resources(device_ids=None):
    return MultiGpuResources(device_ids=device_ids)

def create_index_params(
    distribution_mode=DISTRIBUTION_MODE,
    metric=METRIC,
    graph_degree=32,
    intermediate_graph_degree=64,
    build_algo="ivf_pq",
):
    return cagra.IndexParams(
        distribution_mode=distribution_mode,
        metric=metric,
        graph_degree=graph_degree,
        intermediate_graph_degree=intermediate_graph_degree,
        build_algo=build_algo,
    )

def build_cagra_index(dataset, index_params, resources=None, sync_fn=None, print_info=False):
    if print_info:
        print("Building CAGRA index...")
        print("distribution_mode:", index_params.distribution_mode)
        print("metric:", index_params.metric)
        print("graph_degree:", index_params.graph_degree)
        print("intermediate_graph_degree:", index_params.intermediate_graph_degree)
        print("build_algo:", index_params.build_algo)
        print("dataset shape:", dataset.shape)
        print("dataset dtype:", dataset.dtype)

    if sync_fn is None:
        sync_fn = resources.sync if resources is not None else sync_all_cuda_devices

    sync_fn()
    build_start = time.perf_counter()

    index = cagra.build(index_params, dataset, resources=resources)

    sync_fn()
    build_end = time.perf_counter()
    build_time = build_end - build_start

    if print_info:
        print(f"CAGRA index built in {build_time:.2f} seconds")

    return index, build_time

def create_search_params(
    itopk_size=64,
    search_width=1,
    max_iterations=0,
    search_mode=SEARCH_MODE,
    merge_mode=MERGE_MODE,
    n_rows_per_batch=1000,
    algo="multi_cta",
):
    return cagra.SearchParams(
        search_mode=search_mode,
        merge_mode=merge_mode,
        n_rows_per_batch=n_rows_per_batch,
        itopk_size=itopk_size,
        search_width=search_width,
        max_iterations=max_iterations,
        algo=algo,
    )

def search_cagra(index, queries, search_params, k, resources=None, print_info=False):
    if print_info:
        print("\nPerforming CAGRA search...")
        print("search_mode:", search_params.search_mode)
        print("merge_mode:", search_params.merge_mode)
        print("n_rows_per_batch:", search_params.n_rows_per_batch)
        print("itopk_size:", search_params.itopk_size)
        print("search_width:", search_params.search_width)
        print("max_iterations:", search_params.max_iterations)

    distances, neighbors = cagra.search(search_params, index, queries, k, resources=resources)
    if print_info:
        print("Search completed.")
        print("Distances shape:", distances.shape)
        print("Neighbors shape:", neighbors.shape)  

    return distances, neighbors