from dgl.dataloading.base import BlockSampler
import torch
import dgl


class gpu_sampler(BlockSampler):

    def __init__(self,
                 fanouts,
                 g,
                 edge_dir='in',
                 prob=None,
                 replace=False,
                 prefetch_node_feats=None,
                 prefetch_labels=None,
                 prefetch_edge_feats=None,
                 output_device=None,
                 cache_percent_indices=0,
                 cache_percent_indptr=0,
                 rank=0,
                 world_size=1,
                 type="long"):
        super().__init__(prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.prob = prob
        self.replace = replace

        if type == "long":
            indptr = g.adj_sparse("csc")[0].long()
            indices = g.adj_sparse("csc")[1].long()
            type_size = 8
        elif type == "int":
            indptr = g.adj_sparse("csc")[0].int()
            indices = g.adj_sparse("csc")[1].int()
            type_size = 4

        self.chunk_indptr = torch.classes.dgs_classes.ChunkTensor(
            indptr,
            int((indptr.numel() * type_size * cache_percent_indices) /
                world_size) + type_size)
        self.chunk_indices = torch.classes.dgs_classes.ChunkTensor(
            indices,
            int((indices.numel() * type_size * cache_percent_indptr) /
                world_size) + type_size)

    def __del__(self):
        del self.chunk_indices
        del self.chunk_indptr

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        seeds = seed_nodes
        blocks = []
        for fan_out in reversed(self.fanouts):
            coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_with_chunk_tensor(
                seeds, self.chunk_indptr, self.chunk_indices, fan_out, False)

            frontier, (coo_row,
                       coo_col) = torch.ops.dgs_ops._CAPI_tensor_relabel(
                           [seeds, coo_col], [coo_row, coo_col])

            blocks.insert(
                0,
                dgl.create_block((coo_col, coo_row),
                                 num_src_nodes=frontier.numel(),
                                 num_dst_nodes=seeds.numel()))
            blocks[0].srcdata[dgl.NID] = frontier
            blocks[0].dstdata[dgl.NID] = seeds

            seeds = frontier

        return frontier, seed_nodes, blocks
