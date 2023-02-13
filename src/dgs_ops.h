#ifndef DGS_DGS_OPS_H_
#define DGS_DGS_OPS_H_

#include <torch/script.h>
#include "chunk_tensor.h"

namespace dgs {
// Relabel
std::tuple<torch::Tensor, torch::Tensor> CreateHashMapTensor(
    torch::Tensor input_key, torch::Tensor input_value);
std::tuple<torch::Tensor, std::vector<torch::Tensor>> TensorRelabel(
    std::vector<torch::Tensor> mapping_tensors,
    std::vector<torch::Tensor> requiring_relabel_tensors);

// sampling
std::tuple<torch::Tensor, torch::Tensor> RowWiseSamplingUniform(
    torch::Tensor seeds, torch::Tensor indptr, torch::Tensor indices,
    int64_t num_picks, bool replace);
std::tuple<torch::Tensor, torch::Tensor> RowWiseSamplingUniformWithChunkTensor(
    torch::Tensor seeds, c10::intrusive_ptr<ChunkTensor> indptr,
    c10::intrusive_ptr<ChunkTensor> indices, int64_t num_picks, bool replace);
std::tuple<torch::Tensor, torch::Tensor> RowWiseSamplingProb(
    torch::Tensor seeds, torch::Tensor indptr, torch::Tensor indices,
    torch::Tensor probs, int64_t num_picks, bool replace);
std::tuple<torch::Tensor, torch::Tensor> RowWiseSamplingProbWithChunkTensor(
    torch::Tensor seeds, c10::intrusive_ptr<ChunkTensor> indptr,
    c10::intrusive_ptr<ChunkTensor> indices,
    c10::intrusive_ptr<ChunkTensor> probs, int64_t num_picks, bool replace);

torch::Tensor FetchData(torch::Tensor cpu_data, torch::Tensor gpu_data,
                        torch::Tensor nid, torch::Tensor hashed_key_tensor,
                        torch::Tensor hashed_value_tensor);
torch::Tensor FetchDataWithChunkTensor(torch::Tensor cpu_data,
                                       c10::intrusive_ptr<ChunkTensor> gpu_data,
                                       torch::Tensor nid,
                                       torch::Tensor hashed_key_tensor,
                                       torch::Tensor hashed_value_tensor);

// Index from ChunkTensor
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
SplitIndexFromChunkTensor(ChunkTensor* c_tensor, torch::Tensor index);
torch::Tensor IndexFromChunkTensor(ChunkTensor* c_tensor, torch::Tensor index);
torch::Tensor LocalIndexFromChunkTensor(ChunkTensor* c_tensor,
                                        torch::Tensor index);
torch::Tensor RemoteIndexFromChunkTensor(ChunkTensor* c_tensor,
                                         torch::Tensor index);
torch::Tensor HostIndexFromChunkTensor(ChunkTensor* c_tensor,
                                       torch::Tensor index);

}  // namespace dgs

#endif