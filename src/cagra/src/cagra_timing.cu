#include "cagra_timing.hpp"

TimingSummary summarize_times(std::vector<double> times)
{
  std::vector<double> sorted = times;
  std::sort(sorted.begin(), sorted.end());

  TimingSummary s;
  s.median_sec = sorted[sorted.size() / 2];
  const size_t p95_index =
    std::min(sorted.size() - 1, static_cast<size_t>(0.95 * static_cast<double>(sorted.size() - 1)));
  s.p95_sec = sorted[p95_index];
  return s;
}
