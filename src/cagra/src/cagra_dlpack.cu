#include "cagra_dlpack.hpp"

void noop_dl_deleter(DLManagedTensor*) {}

DLDataType dl_float32()
{
  return DLDataType{kDLFloat, 32, 1};
}

DLDataType dl_float16()
{
  return DLDataType{kDLFloat, 16, 1};
}

DLDataType dl_int64()
{
  return DLDataType{kDLInt, 64, 1};
}

DLDataType dl_int32()
{
  return DLDataType{kDLInt, 32, 1};
}

DlpackMatrix::DlpackMatrix(void* data, int64_t rows, int64_t cols, DLDataType dtype)
  : shape{rows, cols}
{
  managed.dl_tensor.data = data;
  managed.dl_tensor.device = DLDevice{kDLCPU, 0};
  managed.dl_tensor.ndim = 2;
  managed.dl_tensor.dtype = dtype;
  managed.dl_tensor.shape = shape.data();
  managed.dl_tensor.strides = nullptr;
  managed.dl_tensor.byte_offset = 0;
  managed.manager_ctx = nullptr;
  managed.deleter = noop_dl_deleter;
}

DLManagedTensor* DlpackMatrix::get() { return &managed; }

DlpackVector::DlpackVector(void* data, int64_t rows, DLDataType dtype)
  : shape{rows}
{
  managed.dl_tensor.data = data;
  managed.dl_tensor.device = DLDevice{kDLCPU, 0};
  managed.dl_tensor.ndim = 1;
  managed.dl_tensor.dtype = dtype;
  managed.dl_tensor.shape = shape.data();
  managed.dl_tensor.strides = nullptr;
  managed.dl_tensor.byte_offset = 0;
  managed.manager_ctx = nullptr;
  managed.deleter = noop_dl_deleter;
}

DLManagedTensor* DlpackVector::get() { return &managed; }
