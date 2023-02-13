#ifndef DGS_NCCL_CONTEXT_H_
#define DGS_NCCL_CONTEXT_H_

#include <nccl.h>

namespace dgs {
namespace nccl {

typedef struct {
  ncclUniqueId nccl_unique_id_;
} DGSUniqueId;

extern ncclComm_t global_comm;
extern int local_rank;
extern int world_size;
extern cudaStream_t nccl_stream;
extern float* device_buffer;

std::vector<int64_t> GetUniqueId();
void SetNCCL(int64_t nrank, std::vector<int64_t> unique_id_array, int64_t rank);
void _Barrier();
}  // namespace nccl

}  // namespace dgs

#endif