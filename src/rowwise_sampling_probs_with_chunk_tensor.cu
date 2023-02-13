#include <curand_kernel.h>
#include <torch/script.h>

#include "chunk_tensor.h"
#include "cub_function.h"
#include "cuda_common.h"
#include "dgs_ops.h"
#include "warpselect/WarpSelect.cuh"

#define BLOCK_SIZE 128
namespace dgs {

template <typename IdType, typename FloatType, int TILE_SIZE, int BLOCK_WARPS,
          int WARP_SIZE, int NumWarpQ, int NumThreadQ>
__global__ void _CSRRowWiseSampleKernel(
    const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
    const IdType *__restrict__ const in_rows,
    chunk_tensor_wrapper<IdType> *__restrict__ in_index,
    chunk_tensor_wrapper<FloatType> *__restrict__ prob,
    const IdType *__restrict__ const out_ptr,
    const IdType *__restrict__ const row_begin,
    const IdType *__restrict__ const row_end,
    IdType *__restrict__ const out_rows, IdType *__restrict__ const out_cols) {
  // we assign one warp per row
  assert(num_picks <= 32);
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == BLOCK_WARPS);

  __shared__ IdType warpselect_out_index[WARP_SIZE * BLOCK_WARPS];

  // init warpselect
  warpselect::WarpSelect<FloatType, IdType,
                         true,  // produce largest values
                         warpselect::Comparator<FloatType>, NumWarpQ,
                         NumThreadQ, WARP_SIZE * BLOCK_WARPS>
      heap(warpselect::_Limits<FloatType>::getMin(), -1, num_picks);

  int64_t out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  const int64_t last_row =
      MIN(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x,
              threadIdx.y * WARP_SIZE + threadIdx.x, 0, &rng);

  int laneid = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.y;
  IdType *warpselect_out_index_per_warp =
      warpselect_out_index + warp_id * WARP_SIZE;

  while (out_row < last_row) {
    const IdType row = in_rows[out_row];
    const IdType in_row_start = row_begin[out_row];
    const IdType deg = row_end[out_row] - in_row_start;
    const IdType out_row_start = out_ptr[out_row];
    // A-Res value needs to be calculated only if deg is greater than num_picks
    // in weighted rowwise sampling without replacement
    if (deg > num_picks) {
      heap.reset();
      int limit = warpselect::roundDown(deg, WARP_SIZE);
      IdType i = laneid;

      for (; i < limit; i += WARP_SIZE) {
        FloatType item_prob = prob->At(in_row_start + i);
        FloatType ares_prob = __powf(curand_uniform(&rng), 1.0f / item_prob);
        heap.add(ares_prob, i);
      }

      if (i < deg) {
        FloatType item_prob = prob->At(in_row_start + i);
        FloatType ares_prob = __powf(curand_uniform(&rng), 1.0f / item_prob);
        heap.addThreadQ(ares_prob, i);
        i += WARP_SIZE;
      }

      heap.reduce();
      heap.writeOutV(warpselect_out_index_per_warp, num_picks);

      for (int idx = laneid; idx < num_picks; idx += WARP_SIZE) {
        const IdType out_idx = out_row_start + idx;
        const IdType in_idx = warpselect_out_index_per_warp[idx] + in_row_start;
        out_rows[out_idx] = static_cast<IdType>(row);
        out_cols[out_idx] = in_index->At(in_idx);
      }
    } else {
      for (int idx = threadIdx.x; idx < deg; idx += WARP_SIZE) {
        // get in and out index
        const IdType out_idx = out_row_start + idx;
        const IdType in_idx = in_row_start + idx;
        // copy permutation over
        out_rows[out_idx] = static_cast<IdType>(row);
        out_cols[out_idx] = in_index->At(in_idx);
      }
    }

    out_row += BLOCK_WARPS;
  }
}

template <typename IdType, typename FloatType, int TILE_SIZE, int BLOCK_WARPS,
          int WARP_SIZE>
__global__ void _CSRRowWiseSampleReplaceKernel(
    const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
    const IdType *__restrict__ const in_rows,
    chunk_tensor_wrapper<IdType> *__restrict__ in_index,
    chunk_tensor_wrapper<FloatType> *__restrict__ prob,
    const IdType *__restrict__ const out_ptr,
    const IdType *__restrict__ const row_begin,
    const IdType *__restrict__ const row_end,
    const IdType *__restrict__ const cdf_ptr, FloatType *__restrict__ const cdf,
    IdType *__restrict__ const out_rows, IdType *__restrict__ const out_cols) {
  // we assign one warp per row
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == BLOCK_WARPS);

  int64_t out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  const int64_t last_row =
      MIN(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x,
              threadIdx.y * BLOCK_WARPS + threadIdx.x, 0, &rng);

  typedef cub::WarpScan<FloatType> WarpScan;
  __shared__ typename WarpScan::TempStorage temp_storage[BLOCK_WARPS];
  int warp_id = threadIdx.y;
  int laneid = threadIdx.x;

  while (out_row < last_row) {
    const IdType row = in_rows[out_row];
    const IdType in_row_start = row_begin[out_row];
    const IdType out_row_start = out_ptr[out_row];
    const IdType cdf_row_start = cdf_ptr[out_row];
    const IdType deg = row_end[out_row] - in_row_start;
    const FloatType MIN_THREAD_DATA = static_cast<FloatType>(0.0f);

    if (deg > 0) {
      IdType max_iter = (1 + (deg - 1) / WARP_SIZE) * WARP_SIZE;
      // Have the block iterate over segments of items

      FloatType warp_aggregate = static_cast<FloatType>(0.0f);
      for (int idx = laneid; idx < max_iter; idx += WARP_SIZE) {
        FloatType thread_data =
            idx < deg ? prob->At(in_row_start + idx) : MIN_THREAD_DATA;
        if (laneid == 0) thread_data += warp_aggregate;
        thread_data = max(thread_data, MIN_THREAD_DATA);

        WarpScan(temp_storage[warp_id])
            .InclusiveSum(thread_data, thread_data, warp_aggregate);
        __syncwarp();
        // Store scanned items to cdf array
        if (idx < deg) {
          cdf[cdf_row_start + idx] = thread_data;
        }
      }
      __syncwarp();

      for (int idx = laneid; idx < num_picks; idx += WARP_SIZE) {
        // get random value
        FloatType sum = cdf[cdf_row_start + deg - 1];
        FloatType rand = static_cast<FloatType>(curand_uniform(&rng) * sum);
        // get the offset of the first value within cdf array which is greater
        // than random value.
        IdType item = cub::UpperBound<FloatType *, IdType, FloatType>(
            &cdf[cdf_row_start], deg, rand);
        item = MIN(item, deg - 1);
        // get in and out index
        const IdType in_idx = in_row_start + item;
        const IdType out_idx = out_row_start + idx;
        // copy permutation over
        out_rows[out_idx] = static_cast<IdType>(row);
        out_cols[out_idx] = in_index->At(in_idx);
      }
    }
    out_row += BLOCK_WARPS;
  }
}

template <typename IdType, typename FloatType>
std::tuple<torch::Tensor, torch::Tensor> RowWiseSamplingProbWithChunkTensorCUDA(
    torch::Tensor seeds, c10::intrusive_ptr<ChunkTensor> indptr,
    c10::intrusive_ptr<ChunkTensor> indices,
    c10::intrusive_ptr<ChunkTensor> probs, int64_t num_picks, bool replace) {
  chunk_tensor_wrapper<IdType> *d_indptr_wrapper_ptr =
      reinterpret_cast<chunk_tensor_wrapper<IdType> *>(
          indptr->wrapper_chunktensor_ptr_);
  chunk_tensor_wrapper<IdType> *d_indices_wrapper_ptr =
      reinterpret_cast<chunk_tensor_wrapper<IdType> *>(
          indices->wrapper_chunktensor_ptr_);
  chunk_tensor_wrapper<FloatType> *f_probs_wrapper_ptr =
      reinterpret_cast<chunk_tensor_wrapper<FloatType> *>(
          probs->wrapper_chunktensor_ptr_);

  int num_items = seeds.numel();
  torch::Tensor row_begin_tensor = torch::empty(
      num_items,
      torch::TensorOptions().dtype(indptr->dtype_).device(torch::kCUDA));
  torch::Tensor row_end_tensor = torch::empty(
      num_items,
      torch::TensorOptions().dtype(indptr->dtype_).device(torch::kCUDA));
  torch::Tensor sub_indptr = torch::empty(
      (num_items + 1),
      torch::TensorOptions().dtype(indptr->dtype_).device(torch::kCUDA));
  torch::Tensor temp_indptr = torch::empty(
      (num_items + 1),
      torch::TensorOptions().dtype(indptr->dtype_).device(torch::kCUDA));

  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(
      thrust::device, it(0), it(num_items),
      [in = seeds.data_ptr<IdType>(), in_indptr = d_indptr_wrapper_ptr,
       sub_ptr = sub_indptr.data_ptr<IdType>(),
       tmp_ptr = temp_indptr.data_ptr<IdType>(),
       row_begin = row_begin_tensor.data_ptr<IdType>(),
       row_end = row_end_tensor.data_ptr<IdType>(), replace, num_picks,
       num_items] __device__(int i) mutable {
        IdType row = in[i];
        row_begin[i] = in_indptr->At(row);
        row_end[i] = in_indptr->At(row + 1);
        IdType deg = row_end[i] - row_begin[i];
        if (replace) {
          sub_ptr[i] = deg == 0 ? 0 : num_picks;
          tmp_ptr[i] = deg;
        } else {
          sub_ptr[i] = MIN(deg, num_picks);
          tmp_ptr[i] = deg > num_picks ? deg : 0;
        }
        if (i == num_items - 1) {
          sub_ptr[num_items] = 0;
          tmp_ptr[num_items] = 0;
        }
      });

  cub_exclusiveSum<IdType>(sub_indptr.data_ptr<IdType>(), num_items + 1);
  cub_exclusiveSum<IdType>(temp_indptr.data_ptr<IdType>(), num_items + 1);

  thrust::device_ptr<IdType> sub_prefix(
      static_cast<IdType *>(sub_indptr.data_ptr<IdType>()));
  thrust::device_ptr<IdType> temp_prefix(
      static_cast<IdType *>(temp_indptr.data_ptr<IdType>()));
  int nnz = sub_prefix[num_items];
  int temp_size = temp_prefix[num_items];

  torch::Tensor coo_row = torch::empty(nnz, seeds.options());
  torch::Tensor coo_col = torch::empty(
      nnz, torch::TensorOptions().dtype(indices->dtype_).device(torch::kCUDA));
  torch::Tensor temp = torch::empty(
      temp_size,
      torch::TensorOptions().dtype(probs->dtype_).device(torch::kCUDA));

  const uint64_t random_seed = 7777;
  constexpr int WARP_SIZE = 32;
  constexpr int BLOCK_WARPS = BLOCK_SIZE / WARP_SIZE;
  constexpr int TILE_SIZE = 16;

  if (replace) {
    const dim3 block(WARP_SIZE, BLOCK_WARPS);
    const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
    _CSRRowWiseSampleReplaceKernel<IdType, FloatType, TILE_SIZE, BLOCK_WARPS,
                                   WARP_SIZE><<<grid, block>>>(
        random_seed, num_picks, num_items, seeds.data_ptr<IdType>(),
        d_indices_wrapper_ptr, f_probs_wrapper_ptr,
        sub_indptr.data_ptr<IdType>(), row_begin_tensor.data_ptr<IdType>(),
        row_end_tensor.data_ptr<IdType>(), temp_indptr.data_ptr<IdType>(),
        temp.data_ptr<FloatType>(), coo_row.data_ptr<IdType>(),
        coo_col.data_ptr<IdType>());
  } else {
    const dim3 block(WARP_SIZE, BLOCK_WARPS);
    const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
    _CSRRowWiseSampleKernel<IdType, FloatType, TILE_SIZE, BLOCK_WARPS,
                            WARP_SIZE, 32, 2><<<grid, block>>>(
        random_seed, num_picks, num_items, seeds.data_ptr<IdType>(),
        d_indices_wrapper_ptr, f_probs_wrapper_ptr,
        sub_indptr.data_ptr<IdType>(), row_begin_tensor.data_ptr<IdType>(),
        row_end_tensor.data_ptr<IdType>(), coo_row.data_ptr<IdType>(),
        coo_col.data_ptr<IdType>());
  }

  return std::make_tuple(coo_row, coo_col);
}

std::tuple<torch::Tensor, torch::Tensor> RowWiseSamplingProbWithChunkTensor(
    torch::Tensor seeds, c10::intrusive_ptr<ChunkTensor> indptr,
    c10::intrusive_ptr<ChunkTensor> indices,
    c10::intrusive_ptr<ChunkTensor> probs, int64_t num_picks, bool replace) {
  CHECK_CUDA(seeds);
  DGS_ID_TYPE_SWITCH(indptr->dtype_, IdType, {
    DGS_VALUE_TYPE_SWITCH(probs->dtype_, FloatType, {
      return RowWiseSamplingProbWithChunkTensorCUDA<IdType, FloatType>(
          seeds, indptr, indices, probs, num_picks, replace);
    });
  });

  return std::make_tuple(torch::Tensor(), torch::Tensor());
}
}  // namespace dgs