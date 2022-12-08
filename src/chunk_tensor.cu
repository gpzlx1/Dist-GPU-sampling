#include "chunk_tensor.h"
#include "cuda_common.h"

#define BLOCK_SIZE 128

namespace dgs {
template <typename ValueType, typename IndexType, int TILE_SIZE>
void __global__ _IndexMemcpyKernel(chunk_tensor_wrapper<ValueType>* data,
                                   const IndexType* const index,
                                   const int num_items, const int dim_size,
                                   ValueType* const output) {
  assert(blockDim.x == BLOCK_SIZE);

  int curr_item = blockIdx.x * TILE_SIZE + threadIdx.y;
  const int last_item =
      MIN(static_cast<int>((blockIdx.x + 1) * TILE_SIZE), num_items);

  while (curr_item < last_item) {
    IndexType pos = index[curr_item];
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
      output[curr_item * dim_size + idx] = data->At(pos * dim_size + idx);
    }
    curr_item += 1;
  }
}

torch::Tensor ChunkTensor::Index(torch::Tensor index) {
  CHECK_CUDA(index);
  DGS_VALUE_TYPE_SWITCH(type_, ValueType, {
    DGS_ID_TYPE_SWITCH(index.dtype(), IndexType, {
      int num_items = index.numel();
      torch::Tensor out_data = torch::empty(
          {num_items, stride_},
          torch::TensorOptions().dtype(type_).device(torch::kCUDA));
      chunk_tensor_wrapper<ValueType>* data_wrapper_ptr =
          reinterpret_cast<chunk_tensor_wrapper<ValueType>*>(wrapper_ptr_);

      constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
      const dim3 block(BLOCK_SIZE);
      const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
      _IndexMemcpyKernel<ValueType, IndexType, TILE_SIZE>
          <<<grid, block>>>(data_wrapper_ptr, index.data_ptr<IndexType>(),
                            num_items, stride_, out_data.data_ptr<ValueType>());
      return out_data;
    });
  });

  return torch::Tensor();
}

template <typename ValueType, typename IndexType>
void __global__ _WholeGraphIndexMemcpyKernel(
    chunk_tensor_wrapper<ValueType>* data, const IndexType* const index,
    const int indice_count, const int group_count, const int entry_count,
    ValueType* const output) {
  int64_t thread_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t idx = thread_idx / group_count;
  if (idx >= indice_count) return;
  int64_t src_idx = index[idx] % entry_count;
  int lane_idx = thread_idx % group_count;
  output[thread_idx] = data->At(src_idx * group_count + lane_idx);
}

torch::Tensor ChunkTensor::WholeGraphIndex(torch::Tensor index) {
  CHECK_CUDA(index);
  DGS_VALUE_TYPE_SWITCH(type_, ValueType, {
    DGS_ID_TYPE_SWITCH(index.dtype(), IndexType, {
      int entry_count = total_tensor_size_;
      int group_count = stride_;
      int indice_count = index.numel();
      torch::Tensor out_data = torch::empty(
          {indice_count, stride_},
          torch::TensorOptions().dtype(type_).device(torch::kCUDA));
      chunk_tensor_wrapper<ValueType>* data_wrapper_ptr =
          reinterpret_cast<chunk_tensor_wrapper<ValueType>*>(wrapper_ptr_);

      int thread_count = group_count;
      if (thread_count < 64) thread_count = 64;
      int group_per_block = thread_count / group_count;
      int block_count = (indice_count + group_per_block - 1) / group_per_block;
      _WholeGraphIndexMemcpyKernel<ValueType, IndexType>
          <<<block_count, thread_count>>>(
              data_wrapper_ptr, index.data_ptr<IndexType>(), indice_count,
              group_count, entry_count, out_data.data_ptr<ValueType>());

      return out_data;
    });
  });

  return torch::Tensor();
}

}  // namespace dgs
