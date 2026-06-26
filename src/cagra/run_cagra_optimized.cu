// run_cagra_optimized.cu
// Entry point for the refactored standalone C++/CUDA CAGRA benchmark.

#include "cagra_runner.hpp"

#include <exception>
#include <iostream>

int main()
{
  try {
    run_cagra_optimized();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
