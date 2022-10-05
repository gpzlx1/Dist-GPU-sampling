#include <curand_kernel.h>
#include <torch/script.h>

#include "atomic.h"
#include "cub_function.h"
#include "cuda_common.h"
#include "dgs_ops.h"

#define BLOCK_SIZE 128

namespace dgs {

template <typename IdType>

struct chunk_tensor_wrapper {
  int64_t threshold;
  int64_t num_partitions;
  int64_t each_partion_size;
  thrust::device_vector<IdType *> on_device_data_vector;
  IdType *on_host_data_ptr;
  IdType **on_device_data_ptrs;

  chunk_tensor_wrapper(c10::intrusive_ptr<ChunkTensor> c_tensor) {
    threshold = c_tensor->threshold_;
    num_partitions = c_tensor->num_partitions_;
    each_partion_size = c_tensor->partion_device_tensor_size_;

    on_host_data_ptr = reinterpret_cast<IdType *>(c_tensor->uva_host_ptr_);
    on_device_data_vector.resize(c_tensor->uva_device_ptrs_.size());
    for (int i = 0; i < c_tensor->uva_device_ptrs_.size(); i++) {
      on_device_data_vector[i] =
          reinterpret_cast<IdType *>(c_tensor->uva_device_ptrs_[i]);
    }
    on_device_data_ptrs =
        thrust::raw_pointer_cast(on_device_data_vector.data());
  }

  __host__ __device__ inline IdType At(int64_t index) {
    int partition_idx = index % num_partitions;
    return index >= threshold
               ? on_host_data_ptr[index]
               : on_device_data_ptrs[partition_idx]
                                    [index - each_partion_size * partition_idx];
  }
};

template <typename IdType>
inline void _GetSubIndptr(const IdType *const seeds_ptr,
                          chunk_tensor_wrapper<IdType> *indptr_ptr,
                          IdType *const out_ptr, int64_t num_pick,
                          int64_t num_items, bool replace) {
  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(thrust::device, it(0), it(num_items),
                   [in = seeds_ptr, in_indptr = indptr_ptr, out = out_ptr,
                    replace, num_pick] __device__(int i) mutable {
                     IdType row = in[i];
                     IdType begin = in_indptr->At(row);
                     IdType end = in_indptr->At(row + 1);
                     if (replace) {
                       out[i] = (end - begin) == 0 ? 0 : num_pick;
                     } else {
                       out[i] = MIN(end - begin, num_pick);
                     }
                   });

  cub_exclusiveSum<IdType>(out_ptr, num_items + 1);
}

template <typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformKernel(
    const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
    const IdType *const in_rows, chunk_tensor_wrapper<IdType> *in_ptr,
    chunk_tensor_wrapper<IdType> *in_index, const IdType *const out_ptr,
    IdType *const out_rows, IdType *const out_cols) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = in_ptr->At(row);
    const int64_t deg = in_ptr->At(row + 1) - in_row_start;
    const int64_t out_row_start = out_ptr[out_row];

    if (deg <= num_picks) {
      // just copy row when there is not enough nodes to sample.
      for (int idx = threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        const IdType in_idx = in_row_start + idx;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index->At(in_idx);
      }
    } else {
      // generate permutation list via reservoir algorithm
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        out_cols[out_row_start + idx] = idx;
      }
      __syncthreads();

      for (int idx = num_picks + threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        const int num = curand(&rng) % (idx + 1);
        if (num < num_picks) {
          // use max so as to achieve the replacement order the serial
          // algorithm would have
          atomic::AtomicMax(out_cols + out_row_start + num, IdType(idx));
        }
      }
      __syncthreads();

      // copy permutation over
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        const IdType perm_idx = out_cols[out_row_start + idx] + in_row_start;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index->At(perm_idx);
      }
    }
    out_row += 1;
  }
}

template <typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformReplaceKernel(
    const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
    const IdType *const in_rows, chunk_tensor_wrapper<IdType> *in_ptr,
    chunk_tensor_wrapper<IdType> *in_index, const IdType *const out_ptr,
    IdType *const out_rows, IdType *const out_cols) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = in_ptr->At(row);
    const int64_t out_row_start = out_ptr[out_row];
    const int64_t deg = in_ptr->At(row + 1) - in_row_start;

    if (deg > 0) {
      // each thread then blindly copies in rows only if deg > 0.
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        const int64_t edge = curand(&rng) % deg;
        const int64_t out_idx = out_row_start + idx;
        out_rows[out_idx] = row;
        out_cols[out_idx] = in_index->At(in_row_start + edge);
      }
    }
    out_row += 1;
  }
}

template <typename IdType>
std::tuple<torch::Tensor, torch::Tensor>
RowWiseSamplingUniformCUDAWithChunkTensorCUDA(
    torch::Tensor seeds, c10::intrusive_ptr<ChunkTensor> indptr,
    c10::intrusive_ptr<ChunkTensor> indices, int64_t num_picks, bool replace) {
  chunk_tensor_wrapper<IdType> h_indptr_wrapper(indptr);
  chunk_tensor_wrapper<IdType> h_indices_wrapper(indices);

  chunk_tensor_wrapper<IdType> *d_indptr_wrapper_ptr;
  chunk_tensor_wrapper<IdType> *d_indices_wrapper_ptr;
  CUDA_CALL(
      cudaMalloc(&d_indptr_wrapper_ptr, sizeof(chunk_tensor_wrapper<IdType>)));
  CUDA_CALL(
      cudaMalloc(&d_indices_wrapper_ptr, sizeof(chunk_tensor_wrapper<IdType>)));

  CUDA_CALL(cudaMemcpy(d_indptr_wrapper_ptr, &h_indptr_wrapper,
                       sizeof(chunk_tensor_wrapper<IdType>),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_indices_wrapper_ptr, &h_indices_wrapper,
                       sizeof(chunk_tensor_wrapper<IdType>),
                       cudaMemcpyHostToDevice));

  int num_items = seeds.numel();
  torch::Tensor sub_indptr = torch::empty(
      (num_items + 1),
      torch::TensorOptions().dtype(indptr->type_).device(torch::kCUDA));
  _GetSubIndptr<IdType>(seeds.data_ptr<IdType>(), d_indptr_wrapper_ptr,
                        sub_indptr.data_ptr<IdType>(), num_picks, num_items,
                        replace);
  thrust::device_ptr<IdType> item_prefix(
      static_cast<IdType *>(sub_indptr.data_ptr<IdType>()));
  int nnz = item_prefix[num_items];

  torch::Tensor coo_row = torch::empty(nnz, seeds.options());
  torch::Tensor coo_col = torch::empty(
      nnz, torch::TensorOptions().dtype(indices->type_).device(torch::kCUDA));

  const uint64_t random_seed = 7777;
  constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
  if (replace) {
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
    _CSRRowWiseSampleUniformReplaceKernel<IdType, TILE_SIZE><<<grid, block>>>(
        random_seed, num_picks, num_items, seeds.data_ptr<IdType>(),
        d_indptr_wrapper_ptr, d_indices_wrapper_ptr,
        sub_indptr.data_ptr<IdType>(), coo_row.data_ptr<IdType>(),
        coo_col.data_ptr<IdType>());
  } else {
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
    _CSRRowWiseSampleUniformKernel<IdType, TILE_SIZE><<<grid, block>>>(
        random_seed, num_picks, num_items, seeds.data_ptr<IdType>(),
        d_indptr_wrapper_ptr, d_indices_wrapper_ptr,
        sub_indptr.data_ptr<IdType>(), coo_row.data_ptr<IdType>(),
        coo_col.data_ptr<IdType>());
  }

  return std::make_tuple(coo_row, coo_col);
}

std::tuple<torch::Tensor, torch::Tensor> RowWiseSamplingUniformWithChunkTensor(
    torch::Tensor seeds, c10::intrusive_ptr<ChunkTensor> indptr,
    c10::intrusive_ptr<ChunkTensor> indices, int64_t num_picks, bool replace) {
  DGS_ID_TYPE_SWITCH(indptr->type_, IdType, {
    return RowWiseSamplingUniformCUDAWithChunkTensorCUDA<IdType>(
        seeds, indptr, indices, num_picks, replace);
  });
  return std::make_tuple(torch::Tensor(), torch::Tensor());
}
}  // namespace dgs