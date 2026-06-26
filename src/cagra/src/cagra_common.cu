#include "cagra_common.hpp"

std::string getenv_or(const char* name, const std::string& default_value)
{
  const char* value = std::getenv(name);
  if (value == nullptr || std::string(value).empty()) return default_value;
  return std::string(value);
}

int getenv_int_or(const char* name, int default_value)
{
  const char* value = std::getenv(name);
  if (value == nullptr || std::string(value).empty()) return default_value;
  return std::stoi(value);
}

std::optional<std::vector<int>> optional_int_list_from_env(const char* name)
{
  const char* raw = std::getenv(name);
  if (raw == nullptr || std::string(raw).empty()) return std::nullopt;

  std::vector<int> values;
  std::stringstream ss(raw);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (!token.empty()) values.push_back(std::stoi(token));
  }
  return values;
}

void check_cuda(cudaError_t status, const std::string& where)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(where + ": " + cudaGetErrorString(status));
  }
}

void check_cuda_device(cudaError_t status, const std::string& where, int device_id)
{
  if (status != cudaSuccess) {
    std::ostringstream message;
    message << where << " on CUDA device " << device_id << ": "
            << cudaGetErrorString(status);

    if (status == cudaErrorMemoryAllocation) {
      message << ". MultiGpuExactReranker could not allocate resident dataset shards "
              << "or rerank buffers; reduce visible devices/index memory pressure, "
              << "reduce rerank batch size or index memory pressure.";
    }

    throw std::runtime_error(message.str());
  }
}

void check_cuvs(cuvsError_t status, const std::string& where)
{
  if (status != CUVS_SUCCESS) {
    const char* text = cuvsGetLastErrorText();
    throw std::runtime_error(where + ": " + (text ? text : "unknown cuVS error"));
  }
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

DeviceGuard::DeviceGuard(int device_id)
{
  check_cuda(cudaGetDevice(&previous_device_), "cudaGetDevice");
  check_cuda(cudaSetDevice(device_id), "cudaSetDevice");
}

DeviceGuard::~DeviceGuard() { cudaSetDevice(previous_device_); }
