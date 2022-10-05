#ifndef DGS_UTILS_H_
#define DGS_UTILS_H_

#include <torch/script.h>

namespace dgs {

namespace utils {
inline void* _getTensorVoidDataPtr(torch::Tensor data) {
  return data.storage().data();
}

inline size_t _getTensorTypeSizeOf(torch::Dtype type) {
  if (type == torch::kInt32) {
    return sizeof(int32_t);
  } else if (type == torch::kInt64) {
    return sizeof(int64_t);
  } else if (type == torch::kFloat) {
    return sizeof(float);
  } else if (type == torch::kDouble) {
    return sizeof(double);
  } else {
    printf("Error in _getTensorSizeInByte!\n");
    exit(-1);
  }
}

inline size_t _getTensorSizeInByte(torch::Tensor data) {
  return _getTensorTypeSizeOf(torch::typeMetaToScalarType(data.dtype())) *
         data.numel();
}

}  // namespace utils

}  // namespace dgs

#endif