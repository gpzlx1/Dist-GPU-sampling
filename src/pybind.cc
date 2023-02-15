#include <torch/custom_class.h>
#include <torch/script.h>

#include "chunk_tensor.h"
#include "cuda/dgs_ops.h"
#include "cuda_context.h"
#include "nccl_context.h"

using namespace dgs;

TORCH_LIBRARY(dgs_classes, m) {
  m.class_<ChunkTensor>("ChunkTensor")
      .def(torch::init<std::vector<int64_t>, torch::ScalarType, int64_t>())
      .def("_CAPI_get_host_tensor", &ChunkTensor::GetHostTensor)
      .def("_CAPI_get_sub_device_tensor", &ChunkTensor::GetSubDeviceTensor)
      .def("_CAPI_index", &ChunkTensor::Index)
      .def("_CAPI_split_index", &ChunkTensor::SplitIndex)
      .def("_CAPI_local_index", &ChunkTensor::LocalIndex)
      .def("_CAPI_remote_index", &ChunkTensor::RemoteIndex)
      .def("_CAPI_host_index", &ChunkTensor::HostIndex)
      .def("_CAPI_load_from_tensor", &ChunkTensor::LoadFromTensor);
}

TORCH_LIBRARY(dgs_ops, m) {
  m.def("_CAPI_tensor_relabel", &cuda::TensorRelabelCUDA)
      .def("_CAPI_sample_neighbors", &cuda::RowWiseSamplingUniformCUDA)
      .def("_CAPI_sample_neighbors_with_chunk_tensor",
           &cuda::RowWiseSamplingUniformWithChunkTensorCUDA)
      .def("_CAPI_sample_neighbors_with_probs", &cuda::RowWiseSamplingProbCUDA)
      .def("_CAPI_sample_neighbors_with_probs_with_chunk_tensor",
           &cuda::RowWiseSamplingProbWithChunkTensorCUDA)
      .def("_CAPI_create_hash_map", &cuda::CreateHashMapTensorCUDA)
      .def("_CAPI_fetch_data", &cuda::FetchDataCUDA)
      .def("_CAPI_fetch_data_chunk_tensor", &cuda::FetchDataWithChunkTensorCUDA)
      .def("_CAPI_get_unique_id", &nccl::GetUniqueId)
      .def("_CAPI_set_nccl", &nccl::SetNCCL)
      .def("_CAPI_get_current_allocated", &CUDAContext::GetCurrAllocated);
}