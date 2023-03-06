#include "graph_relabel.h"
#include "dgs_headers.h"

namespace dgs {

std::tuple<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>,
           torch::Tensor>
GraphRelabelCSC(
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> graph_tensors,
    torch::Tensor inversed_relabel_map) {
  DGS_ID_TYPE_SWITCH(inversed_relabel_map.dtype(), IdType, {
    torch::Tensor indptr = std::get<0>(graph_tensors);
    torch::Tensor indices = std::get<1>(graph_tensors);
    torch::Tensor eid = std::get<2>(graph_tensors);
    torch::Tensor relabeled_indptr =
        torch::empty(indptr.numel(), indptr.options());
    torch::Tensor relabeled_indices =
        torch::empty(indices.numel(), indices.options());
    torch::Tensor relabeled_eid = torch::empty(eid.numel(), eid.options());
    torch::Tensor relabel_map = torch::empty(inversed_relabel_map.numel(),
                                             inversed_relabel_map.options());

    auto indptr_data = indptr.data_ptr<IdType>();
    auto indices_data = indices.data_ptr<IdType>();
    auto eid_data = eid.data_ptr<IdType>();

    auto relabeled_indptr_data = relabeled_indptr.data_ptr<IdType>();
    auto relabeled_indices_data = relabeled_indices.data_ptr<IdType>();
    auto relabeled_eid_data = relabeled_eid.data_ptr<IdType>();

    // inversed relabel map: relabeled nid to original nid
    auto inversed_relabel_map_data = inversed_relabel_map.data_ptr<IdType>();
    auto relabel_map_data = relabel_map.data_ptr<IdType>();

    int64_t num_nodes = inversed_relabel_map.numel();

    // construct relabel map: original nid to relabeled nid
    for (int64_t i = 0; i < num_nodes; i += 1) {
      relabel_map_data[inversed_relabel_map_data[i]] = i;
    }

    // construct relabeled indptr, copy indices and eid, relabel indices
    relabeled_indptr_data[0] = 0;
    for (int64_t i = 0; i < num_nodes; i += 1) {
      IdType original_nid = inversed_relabel_map_data[i];
      IdType original_col_start = indptr_data[original_nid];
      IdType col_start = relabeled_indptr_data[i];
      int degree = indptr_data[original_nid + 1] - original_col_start;
      relabeled_indptr_data[i + 1] = degree + col_start;
      for (int j = 0; j < degree; j += 1) {
        relabeled_indices_data[col_start + j] =
            relabel_map_data[indices_data[original_col_start + j]];
        relabeled_eid_data[col_start + j] = eid_data[original_col_start + j];
      }
    }

    // sort relabeled indices of each node
    for (int64_t i = 0; i < num_nodes; i += 1) {
      IdType col_start = relabeled_indptr_data[i];
      int degree = relabeled_indptr_data[i + 1] - col_start;
      for (int j = 0; j < degree - 1; j += 1) {
        for (int k = j; k < degree - 1; k += 1) {
          if (relabeled_indices_data[col_start + k] >=
              relabeled_indices_data[col_start + k + 1]) {
            IdType indices_swap_temp = relabeled_indices_data[col_start + k];
            IdType eid_swap_temp = relabeled_eid_data[col_start + k];
            relabeled_indices_data[col_start + k] =
                relabeled_indices_data[col_start + k + 1];
            relabeled_eid_data[col_start + k] =
                relabeled_eid_data[col_start + k + 1];
            relabeled_indices_data[col_start + k + 1] = indices_swap_temp;
            relabeled_eid_data[col_start + k + 1] = eid_swap_temp;
          }
        }
      }
    }

    return std::make_tuple(
        std::make_tuple(relabeled_indptr, relabeled_indices, relabeled_eid),
        relabel_map);
  });

  return std::make_tuple(
      std::make_tuple(torch::Tensor(), torch::Tensor(), torch::Tensor()),
      torch::Tensor());
}

}  // namespace dgs