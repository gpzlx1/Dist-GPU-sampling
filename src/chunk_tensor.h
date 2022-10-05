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
    tensor_size_in_btye_ = utils::_getTensorSizeInByte(data);
    int64_t local_rank = mpi::local_rank;

    type_ = torch::typeMetaToScalarType(data.dtype());
    capacity_per_gpu_ = capacity_per_gpu;
    threshold_ = capacity_per_gpu_ * mpi::global_comm_size;
    uva_device_ptrs_.resize(mpi::global_comm_size);

    // cudaIpcGetMemHandle not support host_ptr. Therefore, just alias here.
    uva_host_ptr_ = utils::_getTensorVoidDataPtr(data);

    /*
    if (local_rank == 0) {
      CUDA_CALL(cudaMallocHost(&uva_host_ptr_, tensor_size_in_btye_));
      CUDA_CALL(cudaMemcpy(uva_host_ptr_,
    utils::_getTensorVoidDataPtr(data), tensor_size_in_btye_,
    cudaMemcpyHostToHost));
    }

    // Context IPC for uva_host_ptr_
    cudaIpcMemHandle_t ipc_host_mem_handle;
    if (local_rank == 0) {
      CUDA_CALL(cudaIpcGetMemHandle(&ipc_host_mem_handle, uva_host_ptr_));
    }
    MPI_Bcast(&ipc_host_mem_handle, sizeof(cudaIpcMemHandle_t), MPI_CHAR, 0,
              mpi::global_comm);
    if (local_rank != 0) {
      cudaIpcOpenMemHandle(&uva_host_ptr_, ipc_host_mem_handle,
                           cudaIpcMemLazyEnablePeerAccess);
    }
    */

    void *uva_device_ptr = nullptr;
    CUDA_CALL(cudaMalloc(&uva_device_ptr, capacity_per_gpu_));
    CUDA_CALL(cudaMemcpy(uva_device_ptr,
                         reinterpret_cast<char *>(uva_host_ptr_) +
                             local_rank * capacity_per_gpu_,
                         capacity_per_gpu_, cudaMemcpyHostToDevice));

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
    return torch::from_blob(
        uva_host_ptr_,
        tensor_size_in_btye_ / utils::_getTensorTypeSizeOf(type_),
        torch::TensorOptions().device(torch::kCPU).dtype(type_));
  }

  torch::Tensor GetSubDeviceTensor() {
    return torch::from_blob(
        uva_device_ptrs_[mpi::local_rank],
        capacity_per_gpu_ / utils::_getTensorTypeSizeOf(type_),
        torch::TensorOptions().dtype(type_).device(torch::kCUDA));
  }

  void Free() {
    int local_rank = mpi::local_rank;
    cudaFree(uva_device_ptrs_[local_rank]);
  }

  torch::Dtype type_;
  int64_t tensor_size_in_btye_;
  int64_t capacity_per_gpu_;
  int64_t threshold_;

  void *uva_host_ptr_;
  thrust::host_vector<void *> uva_device_ptrs_;
};
}  // namespace dgs

#endif