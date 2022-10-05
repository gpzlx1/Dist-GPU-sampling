#ifndef DGS_CHUNK_TENSOR_H_
#define DGS_CHUNK_TENSOR_H_

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <torch/custom_class.h>
#include <torch/script.h>
#include <vector>

#include "./cuda_common.h"
#include "./mpi_context.h"
#include "./utils.h"

namespace dgs {

class ChunkTensor : public torch::CustomClassHolder {
 public:
  ChunkTensor(torch::Tensor data, int64_t capacity_per_gpu) {
    int64_t local_rank = mpi::local_rank;

    total_tensor_size_ = data.numel();
    type_ = torch::typeMetaToScalarType(data.dtype());
    type_size_t_ = utils::_getTensorTypeSizeOf(type_);
    partion_device_tensor_size_ = capacity_per_gpu / type_size_t_;
    threshold_ = partion_device_tensor_size_ * mpi::global_comm_size;
    uva_device_ptrs_.resize(mpi::global_comm_size);

    if (threshold_ > total_tensor_size_) {
      threshold_ = total_tensor_size_;
      partion_device_tensor_size_ =
          (total_tensor_size_ + mpi::global_comm_size - 1) /
          mpi::global_comm_size;
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
    cudaIpcMemHandle_t ipc_device_mem_handle_recvbuff[mpi::global_comm_size];
    CUDA_CALL(cudaIpcGetMemHandle(&ipc_device_mem_handle, uva_device_ptr));
    MPI_Allgather(&ipc_device_mem_handle, sizeof(cudaIpcMemHandle_t), MPI_CHAR,
                  ipc_device_mem_handle_recvbuff,
                  sizeof(cudaIpcMemHandle_t) * mpi::global_comm_size, MPI_CHAR,
                  mpi::global_comm);

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
  };

  ~ChunkTensor() { Free(); }

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

  void Free() {
    int local_rank = mpi::local_rank;
    if (uva_host_ptr_ != nullptr) {
      CUDA_CALL(cudaFree(uva_host_ptr_));
    }
    CUDA_CALL(cudaFree(uva_device_ptrs_[local_rank]));
  }

  torch::Dtype type_;
  // sizoef(type_)
  int64_t type_size_t_;

  int64_t partion_device_tensor_size_;
  int64_t total_tensor_size_;

  int64_t threshold_;

  void *uva_host_ptr_ = nullptr;
  thrust::host_vector<void *> uva_device_ptrs_;
};
}  // namespace dgs

#endif