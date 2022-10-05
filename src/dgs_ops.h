#ifndef DGS_DGS_OPS_H_
#define DGS_DGS_OPS_H_

#include <torch/script.h>

namespace dgs {
void hello_world_from_gpu(int64_t thread_num);
void test_chunk_tensor(c10::intrusive_ptr<ChunkTensor> c_tensor, int64_t mode);
}  // namespace dgs

#endif