#include <torch/custom_class.h>
#include <torch/script.h>

#include "./chunk_tensor.h"
#include "./dgs_ops.h"

using namespace dgs;

TORCH_LIBRARY(dgs_classes, m) {
  m.class_<ChunkTensor>("ChunkTensor")
      .def(torch::init<torch::Tensor, int64_t>())
      .def("_CAPI_get_host_tensor", &ChunkTensor::GetHostTensor)
      .def("_CAPI_get_sub_device_tensor", &ChunkTensor::GetSubDeviceTensor);
}

TORCH_LIBRARY(dgs_ops, m) {
  m.def("_CAPI_hello_world", &hello_world_from_gpu)
      .def("_CAPI_initialize", &mpi::Initialize)
      .def("_CAPI_finalize", &mpi::Finalize)
      .def("_CAPI_test_chunk_tensor", &test_chunk_tensor)
      .def("_CAPI_get_rank", &mpi::GetRank)
      .def("_CAPI_get_size", &mpi::GetSize)
      .def("_CAPI_tensor_relabel", &TensorRelabel)
      .def("_CAPI_sample_neighbors", &RowWiseSamplingUniform);
}