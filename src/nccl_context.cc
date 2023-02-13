#include <torch/script.h>
#include <vector>

#include "cuda_common.h"
#include "dgs_headers.h"
#include "nccl_context.h"

namespace dgs {
namespace nccl {

ncclComm_t global_comm;
int local_rank;
int world_size;
cudaStream_t nccl_stream;
float* device_buffer = nullptr;

std::vector<int64_t> GetUniqueId() {
  std::vector<int64_t> unique_id(AlignUp(sizeof(DGSUniqueId), sizeof(int64_t)));
  DGSUniqueId* ptr = (DGSUniqueId*)unique_id.data();
  ncclGetUniqueId(&ptr->nccl_unique_id_);
  return unique_id;
};

void SetNCCL(int64_t nranks, std::vector<int64_t> unique_id_array,
             int64_t rank) {
  DGSUniqueId unique_id;
  memcpy(&unique_id, unique_id_array.data(), sizeof(unique_id));
  NCCL_CALL(
      ncclCommInitRank(&global_comm, nranks, unique_id.nccl_unique_id_, rank));
  NCCL_CALL(ncclCommUserRank(global_comm, &local_rank));
  NCCL_CALL(ncclCommCount(global_comm, &world_size));
  nccl_stream = 0;
  CUDA_CALL(cudaMalloc(&device_buffer, sizeof(float)));
};

void _Barrier() {
  NCCL_CALL(ncclAllReduce(device_buffer, device_buffer, 1, ncclFloat, ncclSum,
                          global_comm, nccl_stream));
  CUDA_CALL(cudaStreamSynchronize(nccl_stream));
}

}  // namespace nccl

}  // namespace dgs