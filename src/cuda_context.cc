#include "cuda_context.h"

namespace dgs {
namespace CUDAContext {
CUDAContext cuda_context;
int64_t GetCurrAllocated() { return cuda_context.GetCurrAllocated(); };
}  // namespace CUDAContext

}  // namespace dgs