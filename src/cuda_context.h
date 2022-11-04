#ifndef DGS_CUDA_CONTEXT_H_
#define DGS_CUDA_CONTEXT_H_

#include "cuda_common.h"

namespace dgs {
namespace CUDAContext {

class CUDAContext {
 public:
  void Increase(int64_t nbytes) { curr_allocated_ += nbytes; }
  void Decrease(int64_t nbytes) { curr_allocated_ -= nbytes; }
  int64_t GetCurrAllocated() { return curr_allocated_; }

 private:
  int64_t curr_allocated_ = 0;
};

extern CUDAContext cuda_context;

int64_t GetCurrAllocated();

}  // namespace CUDAContext

}  // namespace dgs
#endif