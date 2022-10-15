from dgl.dataloading.base import BlockSampler
import torch
import dgl


class ChunkTensorSampler(BlockSampler):

    def __init__(self,
                 fanouts,
                 g,
                 prob=None,
                 replace=False,
                 cache_percent_indices=0.0,
                 cache_percent_indptr=0.0,
                 cache_percent_probs=0.0,
                 comm_size=1,
                 prefetch_node_feats=None,
                 prefetch_labels=None,
                 prefetch_edge_feats=None,
                 output_device=None):
        super().__init__(prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        self.fanouts = fanouts
        self.prob = prob
        self.replace = replace

        indptr = g.adj_sparse("csc")[0]
        self.chunk_indptr = torch.classes.dgs_classes.ChunkTensor(
            indptr,
            int((indptr.numel() * indptr.element_size() * cache_percent_indptr)
                / comm_size) + indptr.element_size())

        indices = g.adj_sparse("csc")[1]
        self.chunk_indices = torch.classes.dgs_classes.ChunkTensor(
            indices,
            int((indices.numel() * indices.element_size() *
                 cache_percent_indices) / comm_size) + indices.element_size())

        if self.prob:
            probs = g.edata[self.prob]
            self.chunk_probs = torch.classes.dgs_classes.ChunkTensor(
                probs,
                int((probs.numel() * probs.element_size() *
                     cache_percent_probs) / comm_size) + probs.element_size())

    def __del__(self):
        del self.chunk_indices
        del self.chunk_indptr
        if (self.prob):
            del self.chunk_probs

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        seeds = seed_nodes
        blocks = []
        for fan_out in reversed(self.fanouts):
            if (self.prob):
                coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_with_probs_with_chunk_tensor(
                    seeds, self.chunk_indptr, self.chunk_indices,
                    self.chunk_probs, fan_out, self.replace)
            else:
                coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_with_chunk_tensor(
                    seeds, self.chunk_indptr, self.chunk_indices, fan_out,
                    self.replace)

            frontier, (coo_row,
                       coo_col) = torch.ops.dgs_ops._CAPI_tensor_relabel(
                           [seeds, coo_col], [coo_row, coo_col])
            block = dgl.create_block((coo_col, coo_row),
                                     num_src_nodes=frontier.numel(),
                                     num_dst_nodes=seeds.numel())
            blocks.insert(0, block)

            seeds = frontier

        return frontier, seed_nodes, blocks
