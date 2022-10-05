#ifndef DGS_UTILS_H_
#define DGS_UTILS_H_

#include <torch/script.h>

namespace dgs {

namespace utils {
size_t _getTensorSizeInByte(torch::Tensor data) {
  if (data.dtype() == torch::kInt32) {
    return sizeof(int32_t) * data.numel();
  } else if (data.dtype() == torch::kInt64) {
    return sizeof(int64_t) * data.numel();
  } else if (data.dtype() == torch::kFloat) {
    return sizeof(float) * data.numel();
  } else if (data.dtype() == torch::kDouble) {
    return sizeof(double) * data.numel();
  } else {
    printf("Error in _getTensorSizeInByte!\n");
  }
}
}  // namespace utils

}  // namespace dgs

#endif