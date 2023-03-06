#ifndef DGS_GRAPH_RELABEL_H_
#define DGS_GRAPH_RELABEL_H_

#include <torch/script.h>

namespace dgs {

std::tuple<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>,
           torch::Tensor>
GraphRelabelCSC(
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> graph_tensors,
    torch::Tensor inversed_relabel_map);

}  // namespace dgs

#endif