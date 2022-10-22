#include <torch/script.h>
#include "atomic.h"
#include "cuda_common.h"
#include "dgs_ops.h"

namespace dgs {
template <typename IdType>
struct Hashmap {
  __device__ inline Hashmap(IdType *Kptr, IdType *Vptr, size_t numel)
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

    atomic::AtomicMin(vptr + pos, value);
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

  __device__ inline uint32_t hash(int32_t key) { return key & (capacity - 1); }

  __device__ inline uint32_t hash(uint32_t key) { return key & (capacity - 1); }

  __device__ inline uint32_t hash(int64_t key) { return key & (capacity - 1); }

  __device__ inline uint32_t hash(uint64_t key) { return key & (capacity - 1); }

  IdType kEmptyKey{-1};
  IdType *kptr;
  IdType *vptr;
  uint32_t capacity{0};
};

inline int _UpPower(int key) {
  int ret = 1 << static_cast<uint32_t>(std::log2(key) + 1);
  return ret;
}

template <typename IdType>
inline std::tuple<torch::Tensor, torch::Tensor> CreateHashMap(
    torch::Tensor input_key, torch::Tensor input_value) {
  int num_items = input_key.numel();
  int dir_size = _UpPower(num_items);

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
}

std::tuple<torch::Tensor, torch::Tensor> CreateHashMapTensor(
    torch::Tensor input_key, torch::Tensor input_value) {
  CHECK_CUDA(input_key);
  CHECK_CUDA(input_value);
  DGS_ID_TYPE_SWITCH(input_key.dtype(), IdType,
                     { return CreateHashMap<IdType>(input_key, input_value); });

  return std::make_tuple(torch::Tensor(), torch::Tensor());
}

template <typename IdType, typename FloatType>
torch::Tensor FetchDataCUDA(torch::Tensor cpu_data, torch::Tensor gpu_data,
                            torch::Tensor nid, torch::Tensor hashed_key_tensor,
                            torch::Tensor hashed_value_tensor) {
  int num_items = nid.numel();
  int dim = gpu_data.size(1);
  int dir_size = hashed_key_tensor.numel();
  torch::Tensor data_buff = torch::empty({num_items, dim}, gpu_data.options());

  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(
      it(0), it(num_items),
      [cpu_data_buff = cpu_data.data_ptr<FloatType>(),
       gpu_data_buff = gpu_data.data_ptr<FloatType>(),
       key = hashed_key_tensor.data_ptr<IdType>(),
       value = hashed_value_tensor.data_ptr<IdType>(),
       in = nid.data_ptr<IdType>(), out = data_buff.data_ptr<FloatType>(),
       dir_size, dim] __device__(int i) mutable {
        Hashmap<IdType> table(key, value, dir_size);
        int pos = table.SearchForPos(in[i]);
        if (pos != -1) {
          for (int idx = 0; idx < dim; idx++) {
            out[i * dim + idx] = gpu_data_buff[value[pos] * dim + idx];
          }
        } else {
          for (int idx = 0; idx < dim; idx++) {
            out[i * dim + idx] = cpu_data_buff[in[i] * dim + idx];
          }
        }
      });

  return data_buff;
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
      return FetchDataCUDA<IdType, FloatType>(
          cpu_data, gpu_data, nid, hashed_key_tensor, hashed_value_tensor);
    });
  });

  return torch::Tensor();
}

}  // namespace dgs
