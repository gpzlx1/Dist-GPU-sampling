#ifndef DGS_CHUNK_TENSOR_H_
#define DGS_CHUNK_TENSOR_H_

#include <torch/script.h>

#include "cuda_common.h"

namespace dgs {

template <typename ValueType>
struct chunk_tensor_wrapper {
  int64_t threshold_;
  int64_t num_partitions_;
  int64_t each_partion_size_;
  int64_t local_rank_;
  void *host_ptr_;
  void **device_ptrs_;

  __host__ chunk_tensor_wrapper(int64_t threshold, int64_t num_partitions,
                                int64_t each_partion_size, int64_t local_rank,
                                void *host_ptr, void **device_ptrs) {
    threshold_ = threshold;
    num_partitions_ = num_partitions;
    each_partion_size_ = each_partion_size;
    local_rank_ = local_rank;
    host_ptr_ = host_ptr;
    device_ptrs_ = device_ptrs;
    CHECK(threshold_ == each_partion_size_ * num_partitions_);
  }

  ~chunk_tensor_wrapper(){};

  __device__ inline ValueType At(int64_t index) {
    if (index >= threshold_) {
      return reinterpret_cast<ValueType *>(host_ptr_)[index - threshold_];
    } else {
      int partition_idx = index / each_partion_size_;
      return reinterpret_cast<ValueType *>(
          device_ptrs_[partition_idx])[index -
                                       each_partion_size_ * partition_idx];
    }
  }

  __device__ inline ValueType LocalAt(int64_t index) {
    return reinterpret_cast<ValueType *>(
        device_ptrs_[local_rank_])[index - each_partion_size_ * local_rank_];
  }

  __device__ inline ValueType RemoteAt(int64_t index) {
    int partition_idx = index / each_partion_size_;
    return reinterpret_cast<ValueType *>(
        device_ptrs_[partition_idx])[index -
                                     each_partion_size_ * partition_idx];
  }

  __device__ inline ValueType HostAt(int64_t index) {
    return reinterpret_cast<ValueType *>(host_ptr_)[index - threshold_];
  }
};

class ChunkTensor : public torch::CustomClassHolder {
 public:
  ChunkTensor(std::vector<int64_t> shapes, torch::ScalarType dtype,
              int64_t capacity_per_gpu);
  ~ChunkTensor() { _Free(); }

  torch::Tensor GetHostTensor();

  torch::Tensor GetSubDeviceTensor();

  torch::Tensor Index(torch::Tensor index);
  torch::Tensor LocalIndex(torch::Tensor index);
  torch::Tensor RemoteIndex(torch::Tensor index);
  torch::Tensor HostIndex(torch::Tensor index);
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SplitIndex(
      torch::Tensor index);

  void LoadFromTensor(torch::Tensor data);

  torch::ScalarType dtype_;
  int64_t dtype_size_t_;

  std::vector<int64_t> strides_;
  std::vector<int64_t> shapes_;
  int64_t elem_stride_;

  int64_t host_elem_size_;
  int64_t device_elem_size_;
  int64_t total_elem_size_;
  int64_t threshold_;

  int64_t num_partitions_;
  int64_t local_rank_;

  int shmid_ = -1;

  void *host_ptr_ = nullptr;
  void **wrapper_device_ptrs_ = nullptr;
  thrust::host_vector<void *> device_ptrs_;

  void *wrapper_chunktensor_ptr_ = nullptr;

 private:
  void _CreateWrapperPtr();
  void _Free();
};
}  // namespace dgs

#endif