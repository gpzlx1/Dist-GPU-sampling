#include <cuda.h>
#include <torch/script.h>
#include "./dgs_ops.h"

namespace dgs {
__global__ void hello_world_kernel() {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  printf("hello world from %d\n", tid);
}

void hello_world_from_gpu(int64_t thread_num) {
  hello_world_kernel<<<1, thread_num>>>();
  return;
}
};  // namespace dgs
