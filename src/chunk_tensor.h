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
    // mpi one-to-all
    // .... todo

    void *uva_device_ptr = nullptr;
    cudaMalloc(&uva_device_ptr, capacity_per_gpu);
    cudaMemcpy(uva_device_ptr, uva_host_ptr_ + local_rank * capacity_per_gpu,
               capacity_per_gpu, cudaMemcpyDefault);
    uva_device_ptrs_[local_rank] = uva_device_ptr;

    // Context IPC for uva_uva_ptrs_
    // mpi allgather
    // ...... todo

    // after communication, setup uva_device_ptrs_;
    for (int i = 0; i < uva_device_ptrs_.size(); i++) {
      // uva_device_ptrs_[i] = ...;
    }

    /*
    MPI_Bcast(&all_uva_cpu_ptr, sizeof(int64_t *), MPI_CHAR, 0,
              mpi::global_comm);
    MPI_Bcast(&threshold, sizeof(int64_t), MPI_CHAR, 0, mpi::global_comm);

    cudaMalloc(&all_uva_ptr, capacity_per_gpu);
    cudaMemcpy(all_uva_ptr, all_uva_cpu_ptr + local_rank * capacity_per_gpu,
               capacity_per_gpu, cudaMemcpyDefault);

    cudaIpcMemHandle_t local_ipc_handle;
    cudaIpcGetMemHandle(&local_ipc_handle, all_uva_ptr);
    cudaIpcMemHandle_t *ipc_handle_recv_buff;
    cudaMallocHost(&ipc_handle_recv_buff,
                   mpi::global_comm_size * sizeof(cudaIpcMemHandle_t));
    MPI_Allgather(&local_ipc_handle, sizeof(cudaIpcMemHandle_t), MPI_CHAR,
                  ipc_handle_recv_buff,
                  sizeof(cudaIpcMemHandle_t) * mpi::global_comm_size, MPI_CHAR,
                  mpi::global_comm);
    for (int i = 0; i < mpi::global_comm_size; i++) {
      all_ipc_handle.push_back(ipc_handle_recv_buff[i]);
    }
    cudaFree(ipc_handle_recv_buff);
    */
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