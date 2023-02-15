#include "../cuda_common.h"
#include "../dgs_headers.h"
#include "dgs_ops.h"

#define BLOCK_SIZE 128

namespace dgs {
namespace cuda {
template <typename ValueType, typename IndexType, int TILE_SIZE>
void __global__ _IndexMemcpyKernel(
    chunk_tensor_wrapper<ValueType>* __restrict__ data,
    const IndexType* __restrict__ const index, const int num_items,
    const int dim_size, ValueType* __restrict__ const output) {
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

torch::Tensor IndexFromChunkTensorCUDA(ChunkTensor* c_tensor,
                                       torch::Tensor index) {
  CHECK_CUDA(index);
  DGS_VALUE_TYPE_SWITCH(c_tensor->dtype_, ValueType, {
    DGS_ID_TYPE_SWITCH(index.dtype(), IndexType, {
      int num_items = index.numel();
      torch::Tensor out_data = torch::empty(
          {num_items, c_tensor->elem_stride_},
          torch::TensorOptions().dtype(c_tensor->dtype_).device(torch::kCUDA));
      chunk_tensor_wrapper<ValueType>* data_wrapper_ptr =
          reinterpret_cast<chunk_tensor_wrapper<ValueType>*>(
              c_tensor->wrapper_chunktensor_ptr_);

      constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
      const dim3 block(BLOCK_SIZE);
      const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
      _IndexMemcpyKernel<ValueType, IndexType, TILE_SIZE><<<grid, block>>>(
          data_wrapper_ptr, index.data_ptr<IndexType>(), num_items,
          c_tensor->elem_stride_, out_data.data_ptr<ValueType>());
      return out_data;
    });
  });

  return torch::Tensor();
}

template <typename IndexType, int TILE_SIZE>
__global__ void _GetSplitIndexMaskKernel(
    const int64_t num_items, const int64_t local_rank, const int64_t threshold,
    const int64_t each_partition_size, const int64_t stride,
    const IndexType* __restrict__ const in_index,
    bool* __restrict__ const local_index_mask,
    bool* __restrict__ const remote_index_mask,
    bool* __restrict__ const host_index_mask) {
  int64_t idx = blockIdx.x * TILE_SIZE;
  const int64_t last_idx =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_items);

  while (idx < last_idx) {
    if (in_index[idx] >= threshold / stride) {
      host_index_mask[idx] = true;
    } else if (in_index[idx] >= local_rank * each_partition_size / stride &&
               in_index[idx] <
                   (local_rank + 1) * each_partition_size / stride) {
      local_index_mask[idx] = true;
    } else {
      remote_index_mask[idx] = true;
    }
    idx += 1;
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
SplitIndexFromChunkTensorCUDA(ChunkTensor* c_tensor, torch::Tensor index) {
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
        num_items, c_tensor->local_rank_, c_tensor->threshold_,
        c_tensor->device_elem_size_, c_tensor->elem_stride_,
        index.data_ptr<IndexType>(), local_index_mask.data_ptr<bool>(),
        remote_index_mask.data_ptr<bool>(), host_index_mask.data_ptr<bool>());
    return std::make_tuple(index.index({local_index_mask}),
                           index.index({remote_index_mask}),
                           index.index({host_index_mask}));
  });
  return std::make_tuple(torch::Tensor(), torch::Tensor(), torch::Tensor());
}

template <typename ValueType, typename IndexType, int TILE_SIZE>
void __global__ _LocalIndexMemcpyKernel(
    chunk_tensor_wrapper<ValueType>* __restrict__ data,
    const IndexType* __restrict__ const index, const int num_items,
    const int dim_size, ValueType* __restrict__ const output) {
  assert(blockDim.x == BLOCK_SIZE);

  int curr_item = blockIdx.x * TILE_SIZE + threadIdx.y;
  const int last_item =
      MIN(static_cast<int>((blockIdx.x + 1) * TILE_SIZE), num_items);

  while (curr_item < last_item) {
    IndexType pos = index[curr_item];
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
      output[curr_item * dim_size + idx] = data->LocalAt(pos * dim_size + idx);
    }
    curr_item += 1;
  }
}

template <typename ValueType, typename IndexType, int TILE_SIZE>
void __global__ _RemoteIndexMemcpyKernel(
    chunk_tensor_wrapper<ValueType>* __restrict__ data,
    const IndexType* __restrict__ const index, const int num_items,
    const int dim_size, ValueType* __restrict__ const output) {
  assert(blockDim.x == BLOCK_SIZE);

  int curr_item = blockIdx.x * TILE_SIZE + threadIdx.y;
  const int last_item =
      MIN(static_cast<int>((blockIdx.x + 1) * TILE_SIZE), num_items);

  while (curr_item < last_item) {
    IndexType pos = index[curr_item];
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
      output[curr_item * dim_size + idx] = data->RemoteAt(pos * dim_size + idx);
    }
    curr_item += 1;
  }
}

template <typename ValueType, typename IndexType, int TILE_SIZE>
void __global__ _HostIndexMemcpyKernel(
    chunk_tensor_wrapper<ValueType>* __restrict__ data,
    const IndexType* __restrict__ const index, const int num_items,
    const int dim_size, ValueType* __restrict__ const output) {
  assert(blockDim.x == BLOCK_SIZE);

  int curr_item = blockIdx.x * TILE_SIZE + threadIdx.y;
  const int last_item =
      MIN(static_cast<int>((blockIdx.x + 1) * TILE_SIZE), num_items);

  while (curr_item < last_item) {
    IndexType pos = index[curr_item];
    for (int idx = threadIdx.x; idx < dim_size; idx += BLOCK_SIZE) {
      output[curr_item * dim_size + idx] = data->HostAt(pos * dim_size + idx);
    }
    curr_item += 1;
  }
}

torch::Tensor LocalIndexFromChunkTensorCUDA(ChunkTensor* c_tensor,
                                            torch::Tensor index) {
  CHECK_CUDA(index);
  DGS_VALUE_TYPE_SWITCH(c_tensor->dtype_, ValueType, {
    DGS_ID_TYPE_SWITCH(index.dtype(), IndexType, {
      int num_items = index.numel();
      torch::Tensor out_data = torch::empty(
          {num_items, c_tensor->elem_stride_},
          torch::TensorOptions().dtype(c_tensor->dtype_).device(torch::kCUDA));
      chunk_tensor_wrapper<ValueType>* data_wrapper_ptr =
          reinterpret_cast<chunk_tensor_wrapper<ValueType>*>(
              c_tensor->wrapper_chunktensor_ptr_);

      constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
      const dim3 block(BLOCK_SIZE);
      const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
      _LocalIndexMemcpyKernel<ValueType, IndexType, TILE_SIZE><<<grid, block>>>(
          data_wrapper_ptr, index.data_ptr<IndexType>(), num_items,
          c_tensor->elem_stride_, out_data.data_ptr<ValueType>());
      return out_data;
    });
  });

  return torch::Tensor();
}

torch::Tensor RemoteIndexFromChunkTensorCUDA(ChunkTensor* c_tensor,
                                             torch::Tensor index) {
  CHECK_CUDA(index);
  DGS_VALUE_TYPE_SWITCH(c_tensor->dtype_, ValueType, {
    DGS_ID_TYPE_SWITCH(index.dtype(), IndexType, {
      int num_items = index.numel();
      torch::Tensor out_data = torch::empty(
          {num_items, c_tensor->elem_stride_},
          torch::TensorOptions().dtype(c_tensor->dtype_).device(torch::kCUDA));
      chunk_tensor_wrapper<ValueType>* data_wrapper_ptr =
          reinterpret_cast<chunk_tensor_wrapper<ValueType>*>(
              c_tensor->wrapper_chunktensor_ptr_);

      constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
      const dim3 block(BLOCK_SIZE);
      const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
      _RemoteIndexMemcpyKernel<ValueType, IndexType, TILE_SIZE>
          <<<grid, block>>>(data_wrapper_ptr, index.data_ptr<IndexType>(),
                            num_items, c_tensor->elem_stride_,
                            out_data.data_ptr<ValueType>());
      return out_data;
    });
  });

  return torch::Tensor();
}

torch::Tensor HostIndexFromChunkTensorCUDA(ChunkTensor* c_tensor,
                                           torch::Tensor index) {
  CHECK_CUDA(index);
  DGS_VALUE_TYPE_SWITCH(c_tensor->dtype_, ValueType, {
    DGS_ID_TYPE_SWITCH(index.dtype(), IndexType, {
      int num_items = index.numel();
      torch::Tensor out_data = torch::empty(
          {num_items, c_tensor->elem_stride_},
          torch::TensorOptions().dtype(c_tensor->dtype_).device(torch::kCUDA));
      chunk_tensor_wrapper<ValueType>* data_wrapper_ptr =
          reinterpret_cast<chunk_tensor_wrapper<ValueType>*>(
              c_tensor->wrapper_chunktensor_ptr_);

      constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
      const dim3 block(BLOCK_SIZE);
      const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
      _HostIndexMemcpyKernel<ValueType, IndexType, TILE_SIZE><<<grid, block>>>(
          data_wrapper_ptr, index.data_ptr<IndexType>(), num_items,
          c_tensor->elem_stride_, out_data.data_ptr<ValueType>());
      return out_data;
    });
  });

  return torch::Tensor();
}

}  // namespace cuda
}  // namespace dgs