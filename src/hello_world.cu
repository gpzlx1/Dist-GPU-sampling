#include <cuda.h>
#include <thrust/device_vector.h>
#include <torch/script.h>
#include "./chunk_tensor.h"
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

template <typename IdType>
__global__ void print_kernel(void** ptrs, int num_item, int num_ptr) {
  for (int i = 0; i < num_ptr; i++) {
    IdType* ptr = reinterpret_cast<IdType*>(ptrs[i]);
    for (int j = 0; j < num_item; j++) {
      printf("[%d %d]: %ld\n", i, j, ptr[j]);
    }
  }
}

template <typename IdType>
__global__ void add_kernel(void** ptrs, int num_item, int num_ptr) {
  for (int i = 0; i < num_ptr; i++) {
    IdType* ptr = reinterpret_cast<IdType*>(ptrs[i]);
    for (int j = 0; j < num_item; j++) {
      ptr[j] = ptr[j] + 1;
    }
  }
}

void test_chunk_tensor(c10::intrusive_ptr<ChunkTensor> c_tensor, int64_t mode) {
  thrust::device_vector<void*> d_vector = c_tensor->device_ptrs_;
  if (mode == 0) {
    print_kernel<int64_t><<<1, 1>>>(thrust::raw_pointer_cast(d_vector.data()),
                                    c_tensor->device_elem_size_,
                                    d_vector.size());
  } else {
    add_kernel<int64_t><<<1, 1>>>(thrust::raw_pointer_cast(d_vector.data()),
                                  c_tensor->device_elem_size_,
                                  d_vector.size());
  }
}

};  // namespace dgs
