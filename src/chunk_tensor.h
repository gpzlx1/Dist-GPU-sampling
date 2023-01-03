#ifndef DGS_CHUNK_TENSOR_H_
#define DGS_CHUNK_TENSOR_H_

#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/script.h>

#include "./cuda_common.h"
#include "./utils.h"
#include "cuda_context.h"
#include "nccl_context.h"

namespace dgs {

template <typename ValueType>
struct chunk_tensor_wrapper {
  int64_t threshold_;
  int64_t num_partitions_;
  int64_t each_partion_size_;
  int64_t local_rank_;
  void *on_host_data_ptr_;
  void **on_device_data_ptrs_;

  __host__ chunk_tensor_wrapper(int64_t threshold, int64_t num_partitions,
                                int64_t each_partion_size, int64_t local_rank,
                                void *on_host_data_ptr,
                                void **uva_device_ptrs_data) {
    threshold_ = threshold;
    num_partitions_ = num_partitions;
    each_partion_size_ = each_partion_size;
    local_rank_ = local_rank;
    on_host_data_ptr_ = on_host_data_ptr;
    on_device_data_ptrs_ = uva_device_ptrs_data;
  }

  ~chunk_tensor_wrapper(){};

  __device__ inline ValueType At(int64_t index) {
    if (index >= threshold_) {
      return reinterpret_cast<ValueType *>(on_host_data_ptr_)[index];
    } else {
      int partition_idx = index / each_partion_size_;
      return reinterpret_cast<ValueType *>(
          on_device_data_ptrs_[partition_idx])[index - each_partion_size_ *
                                                           partition_idx];
    }
  }

  __device__ inline ValueType LocalAt(int64_t index) {
    return reinterpret_cast<ValueType *>(
        on_device_data_ptrs_[local_rank_])[index -
                                           each_partion_size_ * local_rank_];
  }

  __device__ inline ValueType RemoteAt(int64_t index) {
    int partition_idx = index / each_partion_size_;
    return reinterpret_cast<ValueType *>(
        on_device_data_ptrs_[partition_idx])[index - each_partion_size_ *
                                                         partition_idx];
  }

  __device__ inline ValueType HostAt(int64_t index) {
    return reinterpret_cast<ValueType *>(on_host_data_ptr_)[index];
  }
};

class ChunkTensor : public torch::CustomClassHolder {
 public:
  ChunkTensor(torch::Tensor data, int64_t capacity_per_gpu) {
    CHECK(data.dim() == 1 or data.dim() == 2);
    CHECK_CPU(data);
    local_rank_ = nccl::local_rank;
    num_partitions_ = nccl::world_size;

    total_tensor_size_ = data.numel();
    stride_ = data.stride(0);
    type_ = torch::typeMetaToScalarType(data.dtype());
    type_size_t_ = utils::_getTensorTypeSizeOf(type_);
    partion_device_tensor_size_ = capacity_per_gpu / type_size_t_;
    threshold_ = partion_device_tensor_size_ * num_partitions_;
    uva_device_ptrs_.resize(num_partitions_);
    for (int i = 0; i < num_partitions_; i++) {
      uva_device_ptrs_[i] = nullptr;
    }

    if (threshold_ > total_tensor_size_) {
      threshold_ = total_tensor_size_;
      partion_device_tensor_size_ =
          (total_tensor_size_ + num_partitions_ - 1) / num_partitions_;
    }

    // cudaHostRegister for uva_host_ptr_
    host_tensor_ = data;  // avoid data are freed.
    uva_host_ptr_ = utils::_getTensorVoidDataPtr(data);
    CUDA_CALL(cudaHostRegister(uva_host_ptr_, total_tensor_size_ * type_size_t_,
                               cudaHostRegisterDefault));

    if (partion_device_tensor_size_ > 0) {
      // Malloc for uva_device_ptr/uva_device_ptrs_
      // use CUDACachingAllocator, so torch.cuda.max_memory_allocated can read
      // how much memory is allocated for chunk tensor
      size_t each_partion_size_t = partion_device_tensor_size_ * type_size_t_;
      void *uva_device_ptr =
          CUDAContext::cuda_context.raw_alloc(each_partion_size_t);

      CUDA_CALL(cudaMemset(uva_device_ptr, -1, each_partion_size_t));
      CUDA_CALL(cudaMemcpy(
          uva_device_ptr,
          reinterpret_cast<char *>(utils::_getTensorVoidDataPtr(data)) +
              local_rank_ * each_partion_size_t,
          MIN(each_partion_size_t, total_tensor_size_ * type_size_t_ -
                                       local_rank_ * each_partion_size_t),
          cudaMemcpyHostToDevice));
      uva_device_ptrs_[local_rank_] = uva_device_ptr;

      // Context IPC for uva_device_ptrs_
      if (num_partitions_ > 1) {
        cudaIpcMemHandle_t ipc_device_mem_handle;
        cudaIpcMemHandle_t ipc_device_mem_handle_recvbuff[num_partitions_];

        CUDA_CALL(cudaIpcGetMemHandle(&ipc_device_mem_handle, uva_device_ptr));
        CUDA_CALL(cudaHostRegister(&ipc_device_mem_handle,
                                   sizeof(cudaIpcMemHandle_t),
                                   cudaHostRegisterDefault));
        CUDA_CALL(cudaHostRegister(ipc_device_mem_handle_recvbuff,
                                   sizeof(cudaIpcMemHandle_t) * num_partitions_,
                                   cudaHostRegisterDefault));
        NCCL_CALL(ncclAllGather(&ipc_device_mem_handle,
                                ipc_device_mem_handle_recvbuff,
                                sizeof(cudaIpcMemHandle_t), ncclChar,
                                nccl::global_comm, nccl::nccl_stream));
        nccl::_Barrier();
        CUDA_CALL(cudaHostUnregister(&ipc_device_mem_handle));
        CUDA_CALL(cudaHostUnregister(ipc_device_mem_handle_recvbuff));

        // after communication, setup uva_device_ptrs_;
        for (int i = 0; i < int(uva_device_ptrs_.size()); i++) {
          if (i != local_rank_) {
            CUDA_CALL(cudaIpcOpenMemHandle(&uva_device_ptrs_[i],
                                           ipc_device_mem_handle_recvbuff[i],
                                           cudaIpcMemLazyEnablePeerAccess));
          }
        }
      }
    }

    // create wrapper
    uva_device_ptrs_data_ = reinterpret_cast<void **>(
        CUDAContext::cuda_context.raw_alloc(sizeof(void *) * num_partitions_));
    CUDA_CALL(cudaMemcpy(uva_device_ptrs_data_,
                         thrust::raw_pointer_cast(uva_device_ptrs_.data()),
                         sizeof(void *) * num_partitions_,
                         cudaMemcpyHostToDevice));
    _CreateWrapperPtr();
  };

  ~ChunkTensor() { _Free(); }

  torch::Tensor GetHostTensor() {
    if (uva_host_ptr_ != nullptr) {
      return torch::from_blob(
          uva_host_ptr_, total_tensor_size_,
          torch::TensorOptions().device(torch::kCPU).dtype(type_));
    } else {
      printf("No uva_host_ptr is needed for this data!\n");
      return torch::Tensor();
    }
  }

  torch::Tensor GetSubDeviceTensor() {
    return torch::from_blob(
        uva_device_ptrs_[local_rank_], partion_device_tensor_size_,
        torch::TensorOptions().dtype(type_).device(torch::kCUDA));
  }

  torch::Tensor Index(torch::Tensor index);
  torch::Tensor LocalIndex(torch::Tensor index);
  torch::Tensor RemoteIndex(torch::Tensor index);
  torch::Tensor HostIndex(torch::Tensor index);
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SplitIndex(
      torch::Tensor index);
  double MeasureIndexTime(torch::Tensor nids, std::string option);

  void _CreateWrapperPtr() {
    DGS_VALUE_TYPE_SWITCH(type_, ValueType, {
      chunk_tensor_wrapper<ValueType> wrapper(
          threshold_, num_partitions_, partion_device_tensor_size_, local_rank_,
          uva_host_ptr_, uva_device_ptrs_data_);
      wrapper_ptr_ = CUDAContext::cuda_context.raw_alloc(
          sizeof(chunk_tensor_wrapper<ValueType>));
      CUDA_CALL(cudaMemcpy(wrapper_ptr_, &wrapper, sizeof(wrapper),
                           cudaMemcpyHostToDevice));
    });
  }

  void _Free() {
    CUDAContext::cuda_context.raw_delete(uva_device_ptrs_data_);
    CUDAContext::cuda_context.raw_delete(wrapper_ptr_);

    CUDA_CALL(cudaHostUnregister(uva_host_ptr_));
    host_tensor_ = torch::Tensor();

    if (num_partitions_ > 1 && partion_device_tensor_size_ > 0) {
      for (int i = 0; i < num_partitions_; i++) {
        if (local_rank_ != i)
          CUDA_CALL(cudaIpcCloseMemHandle(uva_device_ptrs_[i]);)
      }
      nccl::_Barrier();
    }

    // free uva_device_ptrs_.
    CUDAContext::cuda_context.raw_delete(uva_device_ptrs_[local_rank_]);
  }

  torch::Dtype type_;
  // sizoef(type_)
  int64_t type_size_t_;

  int64_t partion_device_tensor_size_;
  int64_t total_tensor_size_;
  int64_t stride_;

  int64_t threshold_;
  int64_t num_partitions_;

  int64_t local_rank_;

  void *uva_host_ptr_ = nullptr;
  void **uva_device_ptrs_data_ = nullptr;
  thrust::host_vector<void *> uva_device_ptrs_;

  void *wrapper_ptr_ = nullptr;

  torch::Tensor host_tensor_;
};
}  // namespace dgs

#endif