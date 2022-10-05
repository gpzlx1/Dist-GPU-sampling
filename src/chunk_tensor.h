#ifndef DGS_CHUNK_TENSOR_H_
#define DGS_CHUNK_TENSOR_H_

#include <torch/custom_class.h>
#include <torch/script.h>
#include <vector>
#include "cuda_runtime.h"

#include "./mpi_context.h"

namespace dgs {

class ChunkTensor : public torch::CustomClassHolder {
 public:
  ChunkTensor(){};
  ChunkTensor(torch::Tensor data, int64_t total_size, int64_t local_rank,
              int64_t capacity_per_gpu) {
    if (local_rank == 0) {
      cudaMallocHost(&all_uva_cpu_ptr, total_size);
      for (int i = 0; i < data.size(0); i++) {
        all_uva_cpu_ptr[i] = data[i].item().toLong();
      }
      threshold = capacity_per_gpu * mpi::global_comm_size;
      if (threshold > total_size) {
        threshold = total_size;
      }
    }
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

  int64_t *all_uva_cpu_ptr;
  int64_t *all_uva_ptr;
  int64_t threshold;
  std::vector<cudaIpcMemHandle_t> all_ipc_handle;
};
}  // namespace dgs
#endif