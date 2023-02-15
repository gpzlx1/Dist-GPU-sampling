#ifndef DGS_CUDA_DGS_OPS_H_
#define DGS_CUDA_DGS_OPS_H_

#include <torch/script.h>
#include "../chunk_tensor.h"

namespace dgs {
namespace cuda {
// Relabel
std::tuple<torch::Tensor, torch::Tensor> CreateHashMapTensorCUDA(
    torch::Tensor input_key, torch::Tensor input_value);
std::tuple<torch::Tensor, std::vector<torch::Tensor>> TensorRelabelCUDA(
    std::vector<torch::Tensor> mapping_tensors,
    std::vector<torch::Tensor> requiring_relabel_tensors);

// sampling
std::tuple<torch::Tensor, torch::Tensor> RowWiseSamplingUniformCUDA(
    torch::Tensor seeds, torch::Tensor indptr, torch::Tensor indices,
    int64_t num_picks, bool replace);
std::tuple<torch::Tensor, torch::Tensor>
RowWiseSamplingUniformWithChunkTensorCUDA(
    torch::Tensor seeds, c10::intrusive_ptr<ChunkTensor> indptr,
    c10::intrusive_ptr<ChunkTensor> indices, int64_t num_picks, bool replace);
std::tuple<torch::Tensor, torch::Tensor> RowWiseSamplingProbCUDA(
    torch::Tensor seeds, torch::Tensor indptr, torch::Tensor indices,
    torch::Tensor probs, int64_t num_picks, bool replace);
std::tuple<torch::Tensor, torch::Tensor> RowWiseSamplingProbWithChunkTensorCUDA(
    torch::Tensor seeds, c10::intrusive_ptr<ChunkTensor> indptr,
    c10::intrusive_ptr<ChunkTensor> indices,
    c10::intrusive_ptr<ChunkTensor> probs, int64_t num_picks, bool replace);

torch::Tensor FetchDataCUDA(torch::Tensor cpu_data, torch::Tensor gpu_data,
                            torch::Tensor nid, torch::Tensor hashed_key_tensor,
                            torch::Tensor hashed_value_tensor);
torch::Tensor FetchDataWithChunkTensorCUDA(
    torch::Tensor cpu_data, c10::intrusive_ptr<ChunkTensor> gpu_data,
    torch::Tensor nid, torch::Tensor hashed_key_tensor,
    torch::Tensor hashed_value_tensor);

// Index from ChunkTensor
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
SplitIndexFromChunkTensorCUDA(ChunkTensor* c_tensor, torch::Tensor index);
torch::Tensor IndexFromChunkTensorCUDA(ChunkTensor* c_tensor,
                                       torch::Tensor index);
torch::Tensor LocalIndexFromChunkTensorCUDA(ChunkTensor* c_tensor,
                                            torch::Tensor index);
torch::Tensor RemoteIndexFromChunkTensorCUDA(ChunkTensor* c_tensor,
                                             torch::Tensor index);
torch::Tensor HostIndexFromChunkTensorCUDA(ChunkTensor* c_tensor,
                                           torch::Tensor index);

}  // namespace cuda
}  // namespace dgs
#endif