#ifndef DGS_CHUNK_TENSOR_H_
#define DGS_CHUNK_TENSOR_H_

#include <torch/script.h>

#include "./cuda_common.h"
#include "./mpi_context.h"
#include "./utils.h"

namespace dgs {

template <typename IdType>
struct chunk_tensor_wrapper {
  int64_t threshold_;
  int64_t num_partitions_;
  int64_t each_partion_size_;
  void *on_host_data_ptr_;
  void **on_device_data_ptrs_;

  __host__ chunk_tensor_wrapper(int64_t threshold, int64_t num_partitions,
                                int64_t each_partion_size,
                                void *on_host_data_ptr,
                                void **uva_device_ptrs_data) {
    threshold_ = threshold;
    num_partitions_ = num_partitions;
    each_partion_size_ = each_partion_size;
    on_host_data_ptr_ = on_host_data_ptr;
    on_device_data_ptrs_ = uva_device_ptrs_data;
  }

  ~chunk_tensor_wrapper(){};

  __device__ inline IdType At(int64_t index) {
    if (index >= threshold_) {
      return reinterpret_cast<IdType *>(on_host_data_ptr_)[index];
    } else {
      int partition_idx = index / each_partion_size_;
      return reinterpret_cast<IdType *>(
          on_device_data_ptrs_[partition_idx])[index - each_partion_size_ *
                                                           partition_idx];
    }
  }
};

class ChunkTensor : public torch::CustomClassHolder {
 public:
  ChunkTensor(torch::Tensor data, int64_t capacity_per_gpu) {
    CHECK_CPU(data);
    int64_t local_rank = mpi::local_rank;
    num_partitions_ = mpi::global_comm_size;

    total_tensor_size_ = data.numel();
    type_ = torch::typeMetaToScalarType(data.dtype());
    type_size_t_ = utils::_getTensorTypeSizeOf(type_);
    partion_device_tensor_size_ = capacity_per_gpu / type_size_t_;
    threshold_ = partion_device_tensor_size_ * num_partitions_;
    uva_device_ptrs_.resize(num_partitions_);

    if (threshold_ > total_tensor_size_) {
      threshold_ = total_tensor_size_;
      partion_device_tensor_size_ =
          (total_tensor_size_ + num_partitions_ - 1) / num_partitions_;
    }

    // cudaMallocHost for uva_host_ptr_
    if (threshold_ < total_tensor_size_) {
      CUDA_CALL(
          cudaMallocHost(&uva_host_ptr_, total_tensor_size_ * type_size_t_));
      CUDA_CALL(cudaMemcpy(uva_host_ptr_, utils::_getTensorVoidDataPtr(data),
                           total_tensor_size_ * type_size_t_,
                           cudaMemcpyHostToHost));
    } else {
      // Dist-Graph has been fully stored on GPUs. So, we no need to
      // cudaMallocHost for uva_host_ptr_
      uva_host_ptr_ = nullptr;
    }

    void *uva_device_ptr = nullptr;
    size_t each_partion_size_t = partion_device_tensor_size_ * type_size_t_;
    CUDA_CALL(cudaMalloc(&uva_device_ptr, each_partion_size_t));
    CUDA_CALL(cudaMemset(uva_device_ptr, -1, each_partion_size_t));
    CUDA_CALL(cudaMemcpy(
        uva_device_ptr,
        reinterpret_cast<char *>(utils::_getTensorVoidDataPtr(data)) +
            local_rank * each_partion_size_t,
        MIN(each_partion_size_t, total_tensor_size_ * type_size_t_ -
                                     local_rank * each_partion_size_t),
        cudaMemcpyHostToDevice));

    // Context IPC for uva_device_ptrs_
    cudaIpcMemHandle_t ipc_device_mem_handle;
    cudaIpcMemHandle_t ipc_device_mem_handle_recvbuff[num_partitions_];
    CUDA_CALL(cudaIpcGetMemHandle(&ipc_device_mem_handle, uva_device_ptr));
    MPI_Allgather(&ipc_device_mem_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE,
                  ipc_device_mem_handle_recvbuff, sizeof(cudaIpcMemHandle_t),
                  MPI_BYTE, mpi::global_comm);

    // after communication, setup uva_device_ptrs_;
    for (int i = 0; i < int(uva_device_ptrs_.size()); i++) {
      if (i != local_rank) {
        CUDA_CALL(cudaIpcOpenMemHandle(&uva_device_ptrs_[i],
                                       ipc_device_mem_handle_recvbuff[i],
                                       cudaIpcMemLazyEnablePeerAccess));
      } else {
        uva_device_ptrs_[local_rank] = uva_device_ptr;
      }
    }

    CUDA_CALL(
        cudaMalloc(&uva_device_ptrs_data_, sizeof(void *) * num_partitions_));
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
        uva_device_ptrs_[mpi::local_rank], partion_device_tensor_size_,
        torch::TensorOptions().dtype(type_).device(torch::kCUDA));
  }

  void _CreateWrapperPtr() {
    DGS_ID_TYPE_SWITCH(type_, IdType, {
      chunk_tensor_wrapper<IdType> wrapper(
          threshold_, num_partitions_, partion_device_tensor_size_,
          uva_host_ptr_, uva_device_ptrs_data_);
      CUDA_CALL(
          cudaMalloc(&wrapper_ptr_, sizeof(chunk_tensor_wrapper<IdType>)));
      CUDA_CALL(cudaMemcpy(wrapper_ptr_, &wrapper, sizeof(wrapper),
                           cudaMemcpyHostToDevice));
    });
  }

  void _Free() {
    CUDA_CALL(cudaFree(uva_device_ptrs_data_));
    CUDA_CALL(cudaFree(wrapper_ptr_));

    int local_rank = mpi::local_rank;
    if (uva_host_ptr_ != nullptr) {
      CUDA_CALL(cudaFreeHost(uva_host_ptr_));
    }

    for (int i = 0; i < num_partitions_; i++) {
      if (local_rank != i)
        CUDA_CALL(cudaIpcCloseMemHandle(uva_device_ptrs_[i]);)
    }
    MPI_Barrier(mpi::global_comm);
    CUDA_CALL(cudaFree(uva_device_ptrs_[local_rank]));
  }

  torch::Dtype type_;
  // sizoef(type_)
  int64_t type_size_t_;

  int64_t partion_device_tensor_size_;
  int64_t total_tensor_size_;

  int64_t threshold_;
  int64_t num_partitions_;

  void *uva_host_ptr_ = nullptr;
  void **uva_device_ptrs_data_ = nullptr;
  thrust::host_vector<void *> uva_device_ptrs_;

  void *wrapper_ptr_ = nullptr;
};
}  // namespace dgs

#endif