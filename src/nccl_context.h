#ifndef DGS_NCCL_CONTEXT_H_
#define DGS_NCCL_CONTEXT_H_

#include <nccl.h>
#include <torch/script.h>
#include "cuda_common.h"

#define AlignUp(X, ALIGN_SIZE) \
  (((X) + (ALIGN_SIZE)-1) / (ALIGN_SIZE) * (ALIGN_SIZE))

#define NCCL_CALL(X)                                                           \
  do {                                                                         \
    auto result = X;                                                           \
    if (result != ncclSuccess) {                                               \
      const char* p_err_str = ncclGetErrorString(result);                      \
      fprintf(stderr, "File %s Line %d %s returned %s.\n", __FILE__, __LINE__, \
              #X, p_err_str);                                                  \
      abort();                                                                 \
    }                                                                          \
  } while (0)

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