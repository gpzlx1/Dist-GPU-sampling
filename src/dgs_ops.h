#ifndef DGS_DGS_OPS_H_
#define DGS_DGS_OPS_H_

#include <torch/script.h>
#include "chunk_tensor.h"

namespace dgs {
void hello_world_from_gpu(int64_t thread_num);
void test_chunk_tensor(c10::intrusive_ptr<ChunkTensor> c_tensor, int64_t mode);
std::tuple<torch::Tensor, std::vector<torch::Tensor>> TensorRelabel(
    std::vector<torch::Tensor> mapping_tensors,
    std::vector<torch::Tensor> requiring_relabel_tensors);
std::tuple<torch::Tensor, torch::Tensor> RowWiseSamplingUniform(
    torch::Tensor seeds, torch::Tensor indptr, torch::Tensor indices,
    int64_t num_picks, bool replace);
std::tuple<torch::Tensor, torch::Tensor> RowWiseSamplingUniformWithChunkTensor(
    torch::Tensor seeds, c10::intrusive_ptr<ChunkTensor> indptr,
    c10::intrusive_ptr<ChunkTensor> indices, int64_t num_picks, bool replace);
}  // namespace dgs

#endif