#ifndef DGS_CHUNK_TENSOR_H_
#define DGS_CHUNK_TENSOR_H_

#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/script.h>

#include <sys/ipc.h>
#include <sys/shm.h>

#include "./common.h"
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
  }

  ~chunk_tensor_wrapper(){};

  __device__ inline ValueType At(int64_t index) {
    if (index >= threshold_) {
      return reinterpret_cast<ValueType *>(host_ptr_)[index];
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
    return reinterpret_cast<ValueType *>(host_ptr_)[index];
  }
};

class ChunkTensor : public torch::CustomClassHolder {
 public:
  ChunkTensor(std::vector<int64_t> shapes, torch::ScalarType dtype,
              int64_t capacity_per_gpu) {
    CHECK(shapes.size() == 1 || shapes.size() == 2);
    CHECK(capacity_per_gpu > 0);

    local_rank_ = nccl::local_rank;
    num_partitions_ = nccl::world_size;
    int64_t stride = 1;

    shapes_.assign(shapes.begin(), shapes.end());
    strides_.resize(shapes.size());
    for (int i = shapes.size() - 1; i >= 0; i--) {
      strides_[i] = stride;
      stride *= shapes[i];
    }

    elem_stride_ = strides_[0];
    total_elem_size_ = stride;
    dtype_ = dtype;
    dtype_size_t_ = utils::_getTensorTypeSizeOf(dtype_);
    device_elem_size_ = capacity_per_gpu / dtype_size_t_;
    threshold_ = device_elem_size_ * num_partitions_;
    device_ptrs_.resize(num_partitions_);
    for (int i = 0; i < num_partitions_; i++) {
      device_ptrs_[i] = nullptr;
    }

    if (threshold_ > total_elem_size_) {
      threshold_ = total_elem_size_;
      device_elem_size_ =
          (total_elem_size_ + num_partitions_ - 1) / num_partitions_;
    }

    // malloc shared memory for host_ptr_;
    // todo
    host_ptr_ = nullptr;
    host_elem_size_ = total_elem_size_ - threshold_;
    int shmid;
    if (host_elem_size_ > 0) {
      if (local_rank_ == 0) {
        // malloc shared memory for read and write
        shmid = shmget((key_t)0x12345, host_elem_size_ * dtype_size_t_,
                       IPC_CREAT | IPC_EXCL | 0666);
        SHM_CHECK(shmid);
      }

      // HostRegister for direct communication via nccl;
      CUDA_CALL(cudaHostRegister(&shmid, sizeof(int), cudaHostRegisterDefault));
      NCCL_CALL(ncclBroadcast(&shmid, &shmid, 1, ncclInt, 0, nccl::global_comm,
                              nccl::nccl_stream));
      nccl::_Barrier();
      CUDA_CALL(cudaHostUnregister(&shmid));

      // get share memory address
      host_ptr_ = (void *)shmat(shmid, nullptr, 0);
      shmid_ = shmid;
      SHM_CHECK(reinterpret_cast<int64_t>(host_ptr_));
      CUDA_CALL(cudaHostRegister(host_ptr_, host_elem_size_ * dtype_size_t_,
                                 cudaHostRegisterDefault));
    }

    // Malloc GPU shared memory for device_ptrs_;
    if (device_elem_size_ > 0) {
      // Malloc for uva_device_ptr/device_ptrs_ by CUDACachingAllocator, so
      // torch.cuda.max_memory_allocated can read how much memory is allocated
      // for chunk tensor
      size_t each_partion_size_t = device_elem_size_ * dtype_size_t_;
      void *uva_device_ptr =
          CUDAContext::cuda_context.raw_alloc(each_partion_size_t);

      CUDA_CALL(cudaMemset(uva_device_ptr, -1, each_partion_size_t));
      device_ptrs_[local_rank_] = uva_device_ptr;

      // Context IPC for device_ptrs_
      if (num_partitions_ > 1) {
        cudaIpcMemHandle_t ipc_device_mem_handle;
        cudaIpcMemHandle_t ipc_device_mem_handle_recvbuff[num_partitions_];

        CUDA_CALL(cudaIpcGetMemHandle(&ipc_device_mem_handle, uva_device_ptr));
        // HostRegister for direct communication via nccl;
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

        // after communication, setup device_ptrs_;
        for (int i = 0; i < static_cast<int>(device_ptrs_.size()); i++) {
          if (i != local_rank_)
            CUDA_CALL(cudaIpcOpenMemHandle(&device_ptrs_[i],
                                           ipc_device_mem_handle_recvbuff[i],
                                           cudaIpcMemLazyEnablePeerAccess));
        }
      }
    }

    // create wrapper
    wrapper_device_ptrs_ = reinterpret_cast<void **>(
        CUDAContext::cuda_context.raw_alloc(sizeof(void *) * num_partitions_));
    CUDA_CALL(cudaMemcpy(
        wrapper_device_ptrs_, thrust::raw_pointer_cast(device_ptrs_.data()),
        sizeof(void *) * num_partitions_, cudaMemcpyHostToDevice));
    _CreateWrapperPtr();
  }

  ~ChunkTensor() { _Free(); }

  torch::Tensor GetHostTensor() {
    if (host_ptr_ != nullptr) {
      return torch::from_blob(
          host_ptr_, host_elem_size_,
          torch::TensorOptions().device(torch::kCPU).dtype(dtype_));
    } else {
      printf("No uva_host_ptr is needed for this data!\n");
      return torch::Tensor();
    }
  }

  torch::Tensor GetSubDeviceTensor() {
    return torch::from_blob(
        device_ptrs_[local_rank_], device_elem_size_,
        torch::TensorOptions().dtype(dtype_).device(torch::kCUDA));
  }

  torch::Tensor Index(torch::Tensor index);
  torch::Tensor LocalIndex(torch::Tensor index);
  torch::Tensor RemoteIndex(torch::Tensor index);
  torch::Tensor HostIndex(torch::Tensor index);
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SplitIndex(
      torch::Tensor index);

  void LoadFromTensor(torch::Tensor data) {
    CHECK(static_cast<size_t>(data.dim()) == shapes_.size());
    CHECK(data.dtype() == dtype_);
    CHECK(data.device() == torch::kCPU);

    for (uint64_t i = 0; i < shapes_.size(); i++) {
      CHECK(data.size(i) == shapes_[i]);
      CHECK(data.stride(i) == strides_[i]);
    }
    // memcpy
    int64_t each_partion_size_t = device_elem_size_ * dtype_size_t_;
    for (int i = 0; i < num_partitions_; i++) {
      CUDA_CALL(cudaMemcpy(
          device_ptrs_[i],
          reinterpret_cast<char *>(utils::_getTensorVoidDataPtr(data)) +
              i * each_partion_size_t,
          MIN(each_partion_size_t,
              total_elem_size_ * dtype_size_t_ - i * each_partion_size_t),
          cudaMemcpyHostToDevice));
    }
    if (host_elem_size_ > 0) {
      CUDA_CALL(cudaMemcpy(
          host_ptr_,
          reinterpret_cast<char *>(utils::_getTensorVoidDataPtr(data)) +
              each_partion_size_t * num_partitions_,
          host_elem_size_ * dtype_size_t_, cudaMemcpyHostToHost));
    }
  }

  void _CreateWrapperPtr() {
    DGS_VALUE_TYPE_SWITCH(dtype_, ValueType, {
      chunk_tensor_wrapper<ValueType> wrapper(threshold_, num_partitions_,
                                              device_elem_size_, local_rank_,
                                              host_ptr_, wrapper_device_ptrs_);
      wrapper_chunktensor_ptr_ = CUDAContext::cuda_context.raw_alloc(
          sizeof(chunk_tensor_wrapper<ValueType>));
      CUDA_CALL(cudaMemcpy(wrapper_chunktensor_ptr_, &wrapper, sizeof(wrapper),
                           cudaMemcpyHostToDevice));
    });
  }

  void _Free() {
    CUDAContext::cuda_context.raw_delete(wrapper_device_ptrs_);
    CUDAContext::cuda_context.raw_delete(wrapper_chunktensor_ptr_);
    int err = 0;

    // detach Host shared memory;
    if (host_elem_size_ > 0) {
      CUDA_CALL(cudaHostUnregister(host_ptr_));
      err = shmdt(host_ptr_);
      SHM_CHECK(err);

      nccl::_Barrier();
      if (local_rank_ == 0) {
        int err = shmctl(shmid_, IPC_RMID, nullptr);
        SHM_CHECK(err);
      }
    }

    // detach GPU shared memory;
    if (num_partitions_ > 1 && device_elem_size_ > 0) {
      for (int i = 0; i < num_partitions_; i++) {
        if (local_rank_ != i) CUDA_CALL(cudaIpcCloseMemHandle(device_ptrs_[i]);)
      }
    }
    nccl::_Barrier();
    CUDAContext::cuda_context.raw_delete(device_ptrs_[local_rank_]);
  }

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
};
}  // namespace dgs

#endif