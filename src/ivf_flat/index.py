import time

from cuvs.common import MultiGpuResources
from cuvs.neighbors.mg import ivf_flat

from config import (
    DISTRIBUTION_MODE,
    IVF_FLAT_DEFAULT_N_LISTS,
    MERGE_MODE,
    METRIC,
    SEARCH_MODE,
)
from support.timing import sync_all_cuda_devices


def create_multi_gpu_resources(device_ids=None):
    return MultiGpuResources(device_ids=device_ids)


def create_index_params(n_lists=IVF_FLAT_DEFAULT_N_LISTS):
    return ivf_flat.IndexParams(
        distribution_mode=DISTRIBUTION_MODE,
        n_lists=n_lists,
        metric=METRIC,
    )


def build_ivf_flat_index(dataset, index_params, resources=None, sync_fn=None):
    if sync_fn is None:
        sync_fn = resources.sync if resources is not None else sync_all_cuda_devices

    sync_fn()
    build_start = time.perf_counter()
    index = ivf_flat.build(index_params, dataset, resources=resources)
    sync_fn()

    return index, time.perf_counter() - build_start


def create_search_params(n_probes):
    return ivf_flat.SearchParams(
        n_probes=n_probes,
        search_mode=SEARCH_MODE,
        merge_mode=MERGE_MODE,
    )


def search_ivf_flat(index, queries, search_params, k, resources=None):
    return ivf_flat.search(search_params, index, queries, k, resources=resources)
