#ifndef DGS_CHUNK_TENSOR_H_
#define DGS_CHUNK_TENSOR_H_

#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/script.h>

#include <sys/ipc.h>
#include <sys/shm.h>

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
  ChunkTensor(std::vector<int64_t> shapes, torch::ScalarType dtype,
              int64_t capacity_per_gpu) {
    CHECK(shapes.size() == 1 || shapes.size() == 2);

    printf("Begin\n");

    local_rank_ = nccl::local_rank;
    num_partitions_ = nccl::world_size;
    int64_t stride = 1;

    shapes_.assign(shapes.begin(), shapes.end());
    strides_.resize(shapes.size());
    for (int i = shapes.size() - 1; i >= 0; i--) {
      strides_[i] = stride;
      stride *= shapes[i];
    }
    printf("0\n");

    // todo: remove stride_ in the future
    stride_ = strides_[0];
    total_tensor_size_ = stride;
    type_ = dtype;
    type_size_t_ = utils::_getTensorTypeSizeOf(type_);
    partion_device_tensor_size_ = capacity_per_gpu / type_size_t_;
    threshold_ = partion_device_tensor_size_ * num_partitions_;
    uva_device_ptrs_.resize(num_partitions_);
    for (int i = 0; i < num_partitions_; i++) {
      uva_device_ptrs_[i] = nullptr;
    }
    printf("1\n");

    if (threshold_ > total_tensor_size_) {
      threshold_ = total_tensor_size_;
      partion_device_tensor_size_ =
          (total_tensor_size_ + num_partitions_ - 1) / num_partitions_;
    }
    printf("2\n");

    // malloc shared memory for uva_host_ptr_;
    // todo
    uva_host_ptr_ = nullptr;
    host_tensor_size_ = total_tensor_size_ - threshold_;
    int shmid;
    if (host_tensor_size_ > 0) {
      if (local_rank_ == 0) {
        // malloc shared memory for read and write
        shmid = shmget((key_t)0x12345, host_tensor_size_ * type_size_t_,
                       IPC_CREAT | IPC_EXCL | 0666);
        if (shmid == -1) {
          std::cout << "Create share memory for chunktensor failed!"
                    << std::endl;
          std::cout << std::strerror(errno) << std::endl;
        }
      }

      // HostRegister for direct communication via nccl;
      CUDA_CALL(cudaHostRegister(&shmid, sizeof(int), cudaHostRegisterDefault));
      NCCL_CALL(ncclBroadcast(&shmid, &shmid, 1, ncclInt, 0, nccl::global_comm,
                              nccl::nccl_stream));
      nccl::_Barrier();
      CUDA_CALL(cudaHostUnregister(&shmid));

      // get share memory address
      uva_host_ptr_ = (void *)shmat(shmid, nullptr, 0);
      shmid_ = shmid;
      if (reinterpret_cast<int64_t>(uva_host_ptr_) == -1) {
        std::cout << "Attach share memory for chunktensor failed!" << std::endl;
        std::cout << std::strerror(errno) << std::endl;
      }
      CUDA_CALL(cudaHostRegister(uva_host_ptr_,
                                 host_tensor_size_ * type_size_t_,
                                 cudaHostRegisterDefault));
    }
    printf("3\n");

    // Malloc GPU shared memory for uva_device_ptrs_;
    if (partion_device_tensor_size_ > 0) {
      // Malloc for uva_device_ptr/uva_device_ptrs_ by CUDACachingAllocator, so
      // torch.cuda.max_memory_allocated can read how much memory is allocated
      // for chunk tensor
      size_t each_partion_size_t = partion_device_tensor_size_ * type_size_t_;
      void *uva_device_ptr =
          CUDAContext::cuda_context.raw_alloc(each_partion_size_t);

      CUDA_CALL(cudaMemset(uva_device_ptr, -1, each_partion_size_t));
      uva_device_ptrs_[local_rank_] = uva_device_ptr;

      // Context IPC for uva_device_ptrs_
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

        // after communication, setup uva_device_ptrs_;
        for (int i = 0; i < static_cast<int>(uva_device_ptrs_.size()); i++) {
          if (i != local_rank_)
            CUDA_CALL(cudaIpcOpenMemHandle(&uva_device_ptrs_[i],
                                           ipc_device_mem_handle_recvbuff[i],
                                           cudaIpcMemLazyEnablePeerAccess));
        }
      }
    }
    printf("4\n");

    // create wrapper
    uva_device_ptrs_data_ = reinterpret_cast<void **>(
        CUDAContext::cuda_context.raw_alloc(sizeof(void *) * num_partitions_));
    CUDA_CALL(cudaMemcpy(uva_device_ptrs_data_,
                         thrust::raw_pointer_cast(uva_device_ptrs_.data()),
                         sizeof(void *) * num_partitions_,
                         cudaMemcpyHostToDevice));
    _CreateWrapperPtr();
    printf("5\n");
  }

  ~ChunkTensor() { _Free(); }

  torch::Tensor GetHostTensor() {
    if (uva_host_ptr_ != nullptr) {
      return torch::from_blob(
          uva_host_ptr_, host_tensor_size_,
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

  void LoadFromTensor(torch::Tensor data) {
    CHECK(static_cast<size_t>(data.dim()) == shapes_.size());
    for (uint64_t i = 0; i < shapes_.size(); i++) {
      CHECK(data.size(i) == shapes_[i]);
      CHECK(data.stride(i) == strides_[i]);
    }
    // memcpy
    int64_t each_partion_size_t = partion_device_tensor_size_ * type_size_t_;
    for (int i = 0; i < num_partitions_; i++) {
      CUDA_CALL(cudaMemcpy(
          uva_device_ptrs_[i],
          reinterpret_cast<char *>(utils::_getTensorVoidDataPtr(data)) +
              i * each_partion_size_t,
          MIN(each_partion_size_t,
              total_tensor_size_ * type_size_t_ - i * each_partion_size_t),
          cudaMemcpyHostToDevice));
    }
    if (host_tensor_size_ > 0) {
      CUDA_CALL(cudaMemcpy(
          uva_host_ptr_,
          reinterpret_cast<char *>(utils::_getTensorVoidDataPtr(data)) +
              each_partion_size_t * num_partitions_,
          host_tensor_size_ * type_size_t_, cudaMemcpyHostToHost));
    }
  }

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
    int err = 0;

    // detach Host shared memory;
    CUDA_CALL(cudaHostUnregister(uva_host_ptr_));
    err = shmdt(uva_host_ptr_);
    if (err == -1) {
      std::cout << "Detach share memory for chunktensor failed!" << std::endl;
      std::cout << std::strerror(errno) << std::endl;
    }

    if (num_partitions_ > 1 && partion_device_tensor_size_ > 0) {
      // detach GPU shared memory;
      for (int i = 0; i < num_partitions_; i++) {
        if (local_rank_ != i)
          CUDA_CALL(cudaIpcCloseMemHandle(uva_device_ptrs_[i]);)
      }
    }
    nccl::_Barrier();

    // free
    CUDAContext::cuda_context.raw_delete(uva_device_ptrs_[local_rank_]);
    if (host_tensor_size_ > 0) {
      if (local_rank_ == 0) {
        int err = shmctl(shmid_, IPC_RMID, nullptr);
        if (err == -1) {
          std::cout << "Delete share memory for chunktensor failed!"
                    << std::endl;
          std::cout << std::strerror(errno) << std::endl;
        }
      }
    }
  }

  torch::Dtype type_;
  // sizoef(type_)
  int64_t type_size_t_;

  std::vector<int64_t> strides_;
  std::vector<int64_t> shapes_;

  int64_t partion_device_tensor_size_;
  int64_t total_tensor_size_;
  int64_t host_tensor_size_;
  int64_t stride_;

  int64_t threshold_;
  int64_t num_partitions_;

  int64_t local_rank_;

  int shmid_ = -1;

  void *uva_host_ptr_ = nullptr;
  void **uva_device_ptrs_data_ = nullptr;
  thrust::host_vector<void *> uva_device_ptrs_;

  void *wrapper_ptr_ = nullptr;
};
}  // namespace dgs

#endif