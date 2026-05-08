import statistics
import time

from cuda.bindings import runtime


def _check_cuda_error(result):
    error = result[0]
    if int(error) not in (0, 100):
        raise RuntimeError(f"CUDA runtime error: {error}")


def sync_all_cuda_devices():
    device_count_result = runtime.cudaGetDeviceCount()
    if int(device_count_result[0]) == 100:
        return

    _check_cuda_error(device_count_result)

    device_count = device_count_result[1] if len(device_count_result) > 1 else 0
    if not device_count:
        return

    current_device_result = runtime.cudaGetDevice()
    _check_cuda_error(current_device_result)
    original_device = current_device_result[1] if len(current_device_result) > 1 else 0

    for device_id in range(device_count):
        set_device_result = runtime.cudaSetDevice(device_id)
        _check_cuda_error(set_device_result)

        synchronize_result = runtime.cudaDeviceSynchronize()
        _check_cuda_error(synchronize_result)

    set_original_result = runtime.cudaSetDevice(original_device)
    _check_cuda_error(set_original_result)


def summarize_times(times_sec):
    sorted_times = sorted(times_sec)
    p95_index = min(len(sorted_times) - 1, int(0.95 * (len(sorted_times) - 1)))

    return {
        "runs": len(times_sec),
        "min_sec": min(times_sec),
        "mean_sec": statistics.mean(times_sec),
        "median_sec": statistics.median(times_sec),
        "p95_sec": sorted_times[p95_index],
        "all_sec": times_sec,
    }


def measure_synchronized_wall_time(fn, warmup_runs, timed_runs):
    result = None

    for _ in range(warmup_runs):
        result = fn()
        sync_all_cuda_devices()

    times_sec = []

    for _ in range(timed_runs):
        sync_all_cuda_devices()
        start = time.perf_counter()
        result = fn()
        sync_all_cuda_devices()
        end = time.perf_counter()
        times_sec.append(end - start)

    summary = summarize_times(times_sec)
    summary["result"] = result
    return summary
