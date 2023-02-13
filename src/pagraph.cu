#include <torch/script.h>
#include "atomic.h"
#include "cuda_common.h"
#include "dgs_ops.h"

#define BLOCK_SIZE 128
namespace dgs {
template <typename IdType>
struct Hashmap {
  __device__ inline Hashmap(IdType *__restrict__ Kptr,
                            IdType *__restrict__ Vptr, size_t numel)
      : kptr(Kptr), vptr(Vptr), capacity(numel){};

  __device__ inline void Update(IdType key, IdType value) {
    uint32_t delta = 1;
    uint32_t pos = hash(key);
    IdType prev = atomic::AtomicCAS(&kptr[pos], kEmptyKey, key);

    while (prev != key and prev != kEmptyKey) {
      pos = hash(pos + delta);
      delta += 1;
      prev = atomic::AtomicCAS(&kptr[pos], kEmptyKey, key);
    }

    vptr[pos] = value;
  }

  __device__ inline IdType SearchForPos(IdType key) {
    uint32_t delta = 1;
    uint32_t pos = hash(key);

    while (true) {
      if (kptr[pos] == key) {
        return pos;
      }
      if (kptr[pos] == kEmptyKey) {
        return -1;
      }
      pos = hash(pos + delta);
      delta += 1;
    }
  }

  __device__ inline uint32_t Hash32Shift(uint32_t key) {
    key = ~key + (key << 15);  // key = (key << 15) - key - 1;
    key = key ^ (key >> 12);
    key = key + (key << 2);
    key = key ^ (key >> 4);
    key = key * 2057;  // key = (key + (key << 3)) + (key << 11);
    key = key ^ (key >> 16);
    return key;
  }

  __device__ inline uint64_t Hash64Shift(uint64_t key) {
    key = (~key) + (key << 21);  // key = (key << 21) - key - 1;
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);  // key * 265
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);  // key * 21
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return key;
  }

  __device__ inline uint32_t hash(int32_t key) {
    return Hash32Shift(key) & (capacity - 1);
  }

  __device__ inline uint32_t hash(uint32_t key) {
    return Hash32Shift(key) & (capacity - 1);
  }

  __device__ inline uint32_t hash(int64_t key) {
    return static_cast<uint32_t>(Hash64Shift(key)) & (capacity - 1);
  }

  __device__ inline uint32_t hash(uint64_t key) {
    return static_cast<uint32_t>(Hash64Shift(key)) & (capacity - 1);
  }

  IdType kEmptyKey{-1};
  IdType *kptr;
  IdType *vptr;
  uint32_t capacity{0};
};

inline int _UpPower(int key) {
  int ret = 1 << static_cast<uint32_t>(std::log2(key) + 1);
  return ret;
}

std::tuple<torch::Tensor, torch::Tensor> CreateHashMapTensor(
    torch::Tensor input_key, torch::Tensor input_value) {
  CHECK_CUDA(input_key);
  CHECK_CUDA(input_value);
  DGS_ID_TYPE_SWITCH(input_key.dtype(), IdType, {
    int num_items = input_key.numel();
    int dir_size = _UpPower(num_items) * 2;

    IdType MAX = std::numeric_limits<IdType>::max();
    torch::Tensor key_buff_tensor =
        torch::full({dir_size}, -1, input_key.options());
    torch::Tensor value_buff_tensor =
        torch::full({dir_size}, MAX, input_value.options());

    // insert
    using it = thrust::counting_iterator<IdType>;
    thrust::for_each(it(0), it(num_items),
                     [in_key = input_key.data_ptr<IdType>(),
                      in_value = input_value.data_ptr<IdType>(),
                      key_buff = key_buff_tensor.data_ptr<IdType>(),
                      value_buff = value_buff_tensor.data_ptr<IdType>(),
                      dir_size] __device__(int i) mutable {
                       Hashmap<IdType> table(key_buff, value_buff, dir_size);
                       table.Update(in_key[i], in_value[i]);
                     });

    return std::make_tuple(key_buff_tensor, value_buff_tensor);
  });

  return std::make_tuple(torch::Tensor(), torch::Tensor());
}

template <typename IdType, typename FloatType, int TILE_SIZE>
__global__ void _FetchDataKernel(const int64_t num_nids, const int64_t dir_size,
                                 const int64_t data_dim,
                                 const IdType *__restrict__ const in_nids,
                                 const FloatType *__restrict__ const cpu_data,
                                 const FloatType *__restrict__ const gpu_data,
                                 IdType *__restrict__ const hashed_key,
                                 IdType *__restrict__ const hashed_value,
                                 FloatType *__restrict__ const out_data) {
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_node = blockIdx.x * TILE_SIZE;
  const int64_t last_node =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_nids);
  Hashmap<IdType> table(hashed_key, hashed_value, dir_size);

  while (out_node < last_node) {
    const int64_t pos = table.SearchForPos(in_nids[out_node]);

    if (pos != -1) {
      for (int idx = threadIdx.x; idx < data_dim; idx += BLOCK_SIZE) {
        out_data[out_node * data_dim + idx] =
            gpu_data[hashed_value[pos] * data_dim + idx];
      }
    } else {
      for (int idx = threadIdx.x; idx < data_dim; idx += BLOCK_SIZE) {
        out_data[out_node * data_dim + idx] =
            cpu_data[in_nids[out_node] * data_dim + idx];
      }
    }

    out_node += 1;
  }
}

template <typename IdType, typename FloatType, int TILE_SIZE>
__global__ void _FetchDataWithChunkTensorKernel(
    const int64_t num_nids, const int64_t dir_size, const int64_t data_dim,
    const IdType *__restrict__ const in_nids,
    const FloatType *__restrict__ const cpu_data,
    chunk_tensor_wrapper<FloatType> *__restrict__ gpu_data,
    IdType *__restrict__ const hashed_key,
    IdType *__restrict__ const hashed_value,
    FloatType *__restrict__ const out_data) {
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_node = blockIdx.x * TILE_SIZE;
  const int64_t last_node =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_nids);
  Hashmap<IdType> table(hashed_key, hashed_value, dir_size);

  while (out_node < last_node) {
    const int64_t pos = table.SearchForPos(in_nids[out_node]);

    if (pos != -1) {
      for (int idx = threadIdx.x; idx < data_dim; idx += BLOCK_SIZE) {
        out_data[out_node * data_dim + idx] =
            gpu_data->At(hashed_value[pos] * data_dim + idx);
      }
    } else {
      for (int idx = threadIdx.x; idx < data_dim; idx += BLOCK_SIZE) {
        out_data[out_node * data_dim + idx] =
            cpu_data[in_nids[out_node] * data_dim + idx];
      }
    }

    out_node += 1;
  }
}

torch::Tensor FetchData(torch::Tensor cpu_data, torch::Tensor gpu_data,
                        torch::Tensor nid, torch::Tensor hashed_key_tensor,
                        torch::Tensor hashed_value_tensor) {
  CHECK_CUDA(gpu_data);
  CHECK_CUDA(nid);
  CHECK_CUDA(hashed_key_tensor);
  CHECK_CUDA(hashed_value_tensor);
  DGS_ID_TYPE_SWITCH(nid.dtype(), IdType, {
    DGS_VALUE_TYPE_SWITCH(gpu_data.dtype(), FloatType, {
      int num_items = nid.numel();
      int dim = gpu_data.size(1);
      int dir_size = hashed_key_tensor.numel();
      torch::Tensor data_buff =
          torch::empty({num_items, dim}, gpu_data.options());

      constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
      const dim3 block(BLOCK_SIZE);
      const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
      _FetchDataKernel<IdType, FloatType, TILE_SIZE><<<grid, block>>>(
          num_items, dir_size, dim, nid.data_ptr<IdType>(),
          cpu_data.data_ptr<FloatType>(), gpu_data.data_ptr<FloatType>(),
          hashed_key_tensor.data_ptr<IdType>(),
          hashed_value_tensor.data_ptr<IdType>(),
          data_buff.data_ptr<FloatType>());

      return data_buff;
    });
  });

  return torch::Tensor();
}

torch::Tensor FetchDataWithChunkTensor(torch::Tensor cpu_data,
                                       c10::intrusive_ptr<ChunkTensor> gpu_data,
                                       torch::Tensor nid,
                                       torch::Tensor hashed_key_tensor,
                                       torch::Tensor hashed_value_tensor) {
  CHECK_CUDA(nid);
  CHECK_CUDA(hashed_key_tensor);
  CHECK_CUDA(hashed_value_tensor);
  DGS_ID_TYPE_SWITCH(nid.dtype(), IdType, {
    DGS_VALUE_TYPE_SWITCH(cpu_data.dtype(), FloatType, {
      int num_items = nid.numel();
      int dim = cpu_data.size(1);
      int dir_size = hashed_key_tensor.numel();
      torch::Tensor data_buff = torch::empty(
          {num_items, dim},
          torch::TensorOptions().dtype(cpu_data.dtype()).device(torch::kCUDA));
      chunk_tensor_wrapper<FloatType> *gpu_data_wrapper_ptr =
          reinterpret_cast<chunk_tensor_wrapper<FloatType> *>(
              gpu_data->wrapper_chunktensor_ptr_);
      constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
      const dim3 block(BLOCK_SIZE);
      const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
      _FetchDataWithChunkTensorKernel<IdType, FloatType, TILE_SIZE>
          <<<grid, block>>>(num_items, dir_size, dim, nid.data_ptr<IdType>(),
                            cpu_data.data_ptr<FloatType>(),
                            gpu_data_wrapper_ptr,
                            hashed_key_tensor.data_ptr<IdType>(),
                            hashed_value_tensor.data_ptr<IdType>(),
                            data_buff.data_ptr<FloatType>());

      return data_buff;
    });
  });

  return torch::Tensor();
}

}  // namespace dgs
