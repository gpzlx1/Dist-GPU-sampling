#ifndef DGS_CUDA_CONTEXT_H_
#define DGS_CUDA_CONTEXT_H_

#include <unordered_map>
#include "cuda_common.h"
#include "dgs_headers.h"

namespace dgs {
namespace CUDAContext {

class CUDAContext {
 public:
  void _increase(int64_t nbytes) { curr_allocated_ += nbytes; }
  void _decrease(int64_t nbytes) { curr_allocated_ -= nbytes; }
  int64_t GetCurrAllocated() { return curr_allocated_; }

  void* raw_alloc(int64_t nbytes) {
    void* p = nullptr;
    if (nbytes > 0) {
      CUDA_CALL(cudaMalloc(&p, nbytes));
      ptr_size_dir_[reinterpret_cast<int64_t>(p)] = nbytes;
      _increase(nbytes);
    }
    return p;
  }

  void raw_delete(void* p) {
    int64_t nbytes = ptr_size_dir_[reinterpret_cast<int64_t>(p)];
    if (nbytes > 0) {
      CUDA_CALL(cudaFree(p));
      _decrease(nbytes);
    }
  }

 private:
  int64_t curr_allocated_ = 0;
  std::unordered_map<int64_t, int64_t> ptr_size_dir_;
};

extern CUDAContext cuda_context;

int64_t GetCurrAllocated();

}  // namespace CUDAContext

}  // namespace dgs
#endif