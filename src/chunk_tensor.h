#ifndef DGS_CHUNK_TENSOR_H_
#define DGS_CHUNK_TENSOR_H_

#include <torch/custom_class.h>
#include <torch/script.h>

#include "./mpi_context.h"

namespace dgs {

class ChunkTensor : public torch::CustomClassHolder {
 public:
  ChunkTensor(){};
  ChunkTensor(torch::Tensor data, int64_t total_size, int64_t local_rank,
              int64_t capacity_per_gpu){};

  // for test
  ChunkTensor(torch::Tensor data) : data_(data){};

  // for test
  torch::Tensor GetData() const { return data_; };

  // for test
  int64_t GetGlobalData() const { return mpi::global_data; };

 private:
  torch::Tensor data_;
};
}  // namespace dgs
#endif