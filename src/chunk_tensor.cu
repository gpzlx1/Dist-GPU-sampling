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

template <typename IndexType, int TILE_SIZE>
__global__ void _GetSplitIndexMaskKernel(
    const int64_t num_items, const int64_t local_rank, const int64_t threshold,
    const int64_t each_partition_size, const IndexType* const in_index,
    bool* const local_index_mask, bool* const remote_index_mask,
    bool* const host_index_mask) {
  int64_t idx = blockIdx.x * TILE_SIZE;
  const int64_t last_idx =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_items);

  while (idx < last_idx) {
    if (in_index[idx] >= threshold) {
      host_index_mask[idx] = true;
    } else if (in_index[idx] >= local_rank * each_partition_size &&
               in_index[idx] < (local_rank + 1) * each_partition_size) {
      local_index_mask[idx] = true;
    } else {
      remote_index_mask[idx] = true;
    }
    idx += 1;
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ChunkTensor::SplitIndex(
    torch::Tensor index) {
  CHECK_CUDA(index);
  DGS_ID_TYPE_SWITCH(index.dtype(), IndexType, {
    int num_items = index.numel();
    torch::Tensor local_index_mask = torch::zeros(
        num_items,
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    torch::Tensor remote_index_mask = torch::zeros(
        num_items,
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    torch::Tensor host_index_mask = torch::zeros(
        num_items,
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
    _GetSplitIndexMaskKernel<IndexType, TILE_SIZE><<<grid, block>>>(
        num_items, nccl::local_rank, threshold_, partion_device_tensor_size_,
        index.data_ptr<IndexType>(), local_index_mask.data_ptr<bool>(),
        remote_index_mask.data_ptr<bool>(), host_index_mask.data_ptr<bool>());
    return std::make_tuple(index.index({local_index_mask}),
                           index.index({remote_index_mask}),
                           index.index({host_index_mask}));
  });
  return std::make_tuple(torch::Tensor(), torch::Tensor(), torch::Tensor());
}

}  // namespace dgs
