#ifndef DGS_CHUNK_TENSOR_H_
#define DGS_CHUNK_TENSOR_H_

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <torch/custom_class.h>
#include <torch/script.h>
#include <vector>

#include "./mpi_context.h"
#include "./utils.h"

namespace dgs {

class ChunkTensor : public torch::CustomClassHolder {
 public:
  ChunkTensor(){};
  ChunkTensor(torch::Tensor data, int64_t capacity_per_gpu) {
    int64_t tensor_size_in_byte = utils::_getTensorSizeInByte(data);
    int64_t local_rank = mpi::local_rank;
    threshold_ = capacity_per_gpu * mpi::global_comm_size;
    uva_device_ptrs_.resize(mpi::global_comm_size);

    if (local_rank == 0) {
      cudaMallocHost(&uva_host_ptr_, tensor_size_in_byte);
      cudaMemcpy(uva_host_ptr_, data.data_ptr<void>(), tensor_size_in_byte,
                 cudaMemcpyHostToHost);
    }

    // Context IPC for uva_host_ptr_
    cudaIpcMemHandle_t ipc_host_mem_handle;
    if (local_rank == 0) {
      cudaIpcGetMemHandle(&ipc_host_mem_handle, uva_host_ptr_);
    }
    MPI_Bcast(&ipc_host_mem_handle, sizeof(cudaIpcMemHandle_t), MPI_CHAR, 0,
              mpi::global_comm);
    if (local_rank != 0) {
      cudaIpcOpenMemHandle(&uva_host_ptr_, ipc_host_mem_handle,
                           cudaIpcMemLazyEnablePeerAccess);
    }

    void *uva_device_ptr = nullptr;
    cudaMalloc(&uva_device_ptr, capacity_per_gpu);
    cudaMemcpy(uva_device_ptr, uva_host_ptr_ + local_rank * capacity_per_gpu,
               capacity_per_gpu, cudaMemcpyDefault);
    uva_device_ptrs_[local_rank] = uva_device_ptr;

    // Context IPC for uva_device_ptrs_
    cudaIpcMemHandle_t ipc_device_mem_handle;
    cudaIpcMemHandle_t *ipc_device_mem_handle_recvbuff;
    cudaIpcGetMemHandle(&ipc_device_mem_handle, uva_device_ptr);
    cudaMallocHost(&ipc_device_mem_handle_recvbuff,
                   mpi::global_comm_size * sizeof(cudaIpcMemHandle_t));
    MPI_Allgather(&ipc_device_mem_handle, sizeof(cudaIpcMemHandle_t), MPI_CHAR,
                  ipc_device_mem_handle_recvbuff,
                  sizeof(cudaIpcMemHandle_t) * mpi::global_comm_size, MPI_CHAR,
                  mpi::global_comm);

    // after communication, setup uva_device_ptrs_;
    for (int i = 0; i < uva_device_ptrs_.size(); i++) {
      if (i != local_rank) {
        cudaIpcOpenMemHandle(&uva_device_ptrs_[i],
                             ipc_device_mem_handle_recvbuff[i],
                             cudaIpcMemLazyEnablePeerAccess);
      }
    }
    cudaFree(ipc_device_mem_handle_recvbuff);
  };

  torch::Tensor operator[](int64_t index) const {}

  // for test
  ChunkTensor(torch::Tensor data) : data_(data){};

  // for test
  torch::Tensor GetData() const { return data_; };

  // for test
  int64_t GetGlobalData() const { return mpi::global_data; };

 private:
  torch::Tensor data_;
  int64_t threshold_;

  void *uva_host_ptr_;
  thrust::host_vector<void *> uva_device_ptrs_;

  std::vector<cudaIpcMemHandle_t> all_ipc_handle;
};
}  // namespace dgs
#endif