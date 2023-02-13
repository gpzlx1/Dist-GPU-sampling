#ifndef DGS_DGS_HEADERS_H_
#define DGS_DGS_HEADERS_H_

#define CHECK_CPU(x) \
  TORCH_CHECK(!x.device().is_cuda(), #x " must be a CPU tensor")
#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define AlignUp(X, ALIGN_SIZE) (((X) + (ALIGN_SIZE)-1) / (ALIGN_SIZE))
#define MIN(x, y) ((x < y) ? x : y)

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

#define SHM_CHECK(err)                                          \
  {                                                             \
    if (err == -1) {                                            \
      fprintf(stderr,                                           \
              "%s:%d ERROR: SHM call failed "                   \
              "with "                                           \
              "%s (%d).\n",                                     \
              __FILE__, __LINE__, std::strerror(errno), errno); \
    }                                                           \
  }

#define CUDA_CALL(call)                                                  \
  {                                                                      \
    cudaError_t cudaStatus = call;                                       \
    if (cudaSuccess != cudaStatus) {                                     \
      fprintf(stderr,                                                    \
              "%s:%d ERROR: CUDA RT call \"%s\" failed "                 \
              "with "                                                    \
              "%s (%d).\n",                                              \
              __FILE__, __LINE__, #call, cudaGetErrorString(cudaStatus), \
              cudaStatus);                                               \
      exit(cudaStatus);                                                  \
    }                                                                    \
  }

#define DGS_ID_TYPE_SWITCH(val, IdType, ...)         \
  do {                                               \
    if ((val) == torch::kInt32) {                    \
      typedef int32_t IdType;                        \
      { __VA_ARGS__ }                                \
    } else if ((val) == torch::kInt64) {             \
      typedef int64_t IdType;                        \
      { __VA_ARGS__ }                                \
    } else {                                         \
      LOG(FATAL) << "ID can only be int32 or int64"; \
    }                                                \
  } while (0);

#define DGS_VALUE_TYPE_SWITCH(val, VType, ...)                     \
  do {                                                             \
    if ((val) == torch::kInt32) {                                  \
      typedef int32_t VType;                                       \
      { __VA_ARGS__ }                                              \
    } else if ((val) == torch::kInt64) {                           \
      typedef int64_t VType;                                       \
      { __VA_ARGS__ }                                              \
    } else if ((val) == torch::kFloat32) {                         \
      typedef float VType;                                         \
      { __VA_ARGS__ }                                              \
    } else {                                                       \
      LOG(FATAL) << "Value can only be int32 or int64 or float32"; \
    }                                                              \
  } while (0);

#endif