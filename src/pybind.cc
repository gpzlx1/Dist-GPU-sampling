#include <torch/custom_class.h>
#include <torch/script.h>

#include "./chunk_tensor.h"
#include "./dgs_ops.h"

using namespace dgs;

TORCH_LIBRARY(dgs_classes, m) {
  m.class_<ChunkTensor>("ChunkTensor").def(torch::init<torch::Tensor>());
}

TORCH_LIBRARY(dgs_ops, m) { m.def("_CAPI_hello_world", &hello_world_from_gpu); }