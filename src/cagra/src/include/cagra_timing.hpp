#pragma once

#include "cagra_common.hpp"
#include "cagra_resources.hpp"

struct TimingSummary {
  double median_sec = 0.0;
  double p95_sec = 0.0;
};

TimingSummary summarize_times(std::vector<double> times);

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
