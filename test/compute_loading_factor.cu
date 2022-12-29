#include <torch/script.h>
#include <chrono>
#include <cstdio>

#include "chunk_tensor.h"
#include "cuda_common.h"
#include "nccl_context.h"

void createDGSCommunicator(int world_size, int local_rank) {
  std::vector<int64_t> unique_id;
  if (local_rank == 0) {
    unique_id = dgs::nccl::GetUniqueId();
  }
  if (world_size > 1) {
    // broadcast unique id
  }
  dgs::nccl::SetNCCL(world_size, unique_id, local_rank);
}

void computeLocalFactor(double valid_time_threshold, float bandwidth_local) {
  int feature_size = 100000;
  int feature_dim = 128;
  int nids_size = 50000;

  int valid_count = 0;
  std::vector<float> valid_factors;

  createDGSCommunicator(1, 0);

  while (valid_count < 15) {
    auto features = torch::ones(
        {feature_size, feature_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto chunk_features =
        dgs::ChunkTensor(features, features.numel() * features.element_size());
    auto nids = torch::randint(
        0, feature_size, {nids_size},
        torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));

    auto start = std::chrono::system_clock::now();
    auto fetched = chunk_features.LocalIndex(nids);
    CUDA_CALL(cudaDeviceSynchronize());
    auto end = std::chrono::system_clock::now();
    double time = double(std::chrono::duration_cast<std::chrono::microseconds>(
                             end - start)
                             .count()) /
                  1000;

    if (time > valid_time_threshold) {
      double infer_time =
          double(nids.numel() * feature_dim * features.element_size()) / 1024 /
          1024 / 1024 / bandwidth_local * 1000;
      valid_factors.push_back(time / infer_time);
      valid_count += 1;
    } else {
      feature_size *= 10;
      nids_size *= 10;
    }
  }

  float sum = 0;
  for (int i = 3; i < valid_count; i++) {
    sum += valid_factors[i];
  }
  float avg_factor = sum / (valid_count - 3);

  printf("The penalty factor of local loading time = %.3f\n", avg_factor);
}

int main(void) {
  computeLocalFactor(1, 1300);
  return 0;
}