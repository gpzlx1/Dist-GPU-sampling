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
  thrust::device_vector<IdType*> on_device_data_vector;
  IdType* on_host_data_ptr;
  IdType** on_device_data_ptrs;

  chunk_tensor_wrapper(c10::intrusive_ptr<ChunkTensor> c_tensor) {
    threshold = c_tensor->threshold_;
    num_partitions = c_tensor->num_partitions_;
    each_partion_size = c_tensor->partion_device_tensor_size_;

    on_host_data_ptr = reinterpret_cast<IdType*>(c_tensor->uva_host_ptr_);
    on_device_data_vector.resize(c_tensor->uva_device_ptrs_.size());
    for (int i = 0; i < c_tensor->uva_device_ptrs_.size(); i++) {
      on_device_data_vector[i] =
          reinterpret_cast<IdType*>(c_tensor->uva_device_ptrs_[i]);
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
inline void _GetSubIndptr(const IdType* const seeds_ptr,
                          chunk_tensor_wrapper<IdType> indptr_ptr,
                          IdType* const out_ptr, int64_t num_pick,
                          int64_t num_items, bool replace) {
  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(thrust::device, it(0), it(num_items),
                   [in = seeds_ptr, in_indptr = indptr_ptr, out = out_ptr,
                    replace, num_pick] __device__(int i) mutable {
                     IdType row = in[i];
                     IdType begin = in_indptr.At(row);
                     IdType end = in_indptr.At(row + 1);
                     if (replace) {
                       out[i] = (end - begin) == 0 ? 0 : num_pick;
                     } else {
                       out[i] = MIN(end - begin, num_pick);
                     }
                   });

  cub_exclusiveSum<IdType>(out_ptr, num_items + 1);
}

template <typename IdType>
std::tuple<torch::Tensor, torch::Tensor>
RowWiseSamplingUniformCUDAWithChunkTensorCUDA(
    torch::Tensor seeds, c10::intrusive_ptr<ChunkTensor> indptr,
    c10::intrusive_ptr<ChunkTensor> indices, int64_t num_picks, bool replace) {
  /*
  chunk_tensor_wrapper<IdType> h_indptr_wrapper(indptr);
  chunk_tensor_wrapper<IdType> h_indices_wrapper(indices);

  chunk_tensor_wrapper<IdType>* d_indptr_wrapper_ptr;
  chunk_tensor_wrapper<IdType>* d_indices_wrapper_ptr;
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
  */
  // int num_items = seeds.numel();
  // torch::Tensor sub_indptr = torch::empty((num_items + 1), indptr.options());
  //_GetSubIndptr<IdType>(seeds.data_ptr<int64_t>, )
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