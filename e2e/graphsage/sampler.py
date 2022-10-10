from dgl.dataloading import DataLoader, NeighborSampler
from dgl.dataloading.base import BlockSampler
import torch
import os
import torch.distributed as dist
import dgl

class gpu_sampler(BlockSampler):
    def __init__(self, fanouts, g, edge_dir='in', prob=None, replace=False,
                 prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None,
                 output_device=None, cache_percent_indices=0, cache_percent_indptr=0, rank=0, world_size=1):
        super().__init__(prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.prob = prob
        self.replace = replace

        # torch.ops.load_library("./build/libdgs.so")
        # torch.ops.dgs_ops._CAPI_initialize()    

        # torch.set_num_threads(1)
        # torch.cuda.set_device(torch.ops.dgs_ops._CAPI_get_rank())
        # os.environ["RANK"] = str(torch.ops.dgs_ops._CAPI_get_rank())
        # os.environ["WORLD_SIZE"] = str(torch.ops.dgs_ops._CAPI_get_size())        
        # dist.init_process_group(backend='nccl', init_method="env://")
        torch.cuda.set_device(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)        
        indptr = g.adj_sparse("csc")[0].long()
        indices = g.adj_sparse("csc")[1].long()
        print(len(indptr), len(indices))
        print(dist.get_rank(), torch.ops.dgs_ops._CAPI_get_rank())
        print(dist.get_world_size(), torch.ops.dgs_ops._CAPI_get_size())
        self.chunk_indptr = torch.classes.dgs_classes.ChunkTensor(indptr, int(cache_percent_indptr*len(indptr))*8)
        self.chunk_indices = torch.classes.dgs_classes.ChunkTensor(indices, int(cache_percent_indices*len(indices))*8)
        print(self.chunk_indices._CAPI_get_host_tensor())
        print(rank, self.chunk_indices._CAPI_get_sub_device_tensor())        
        # self.chunk_indices = torch.classes.dgs_classes.ChunkTensor(indices, int(indptr[int(cache_percent_indptr*len(indptr))-1])*8)


    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        # output_nodes = seed_nodes
        # blocks = []
        # for fanout in reversed(self.fanouts):
        #     frontier = g.sample_neighbors(
        #         seed_nodes, fanout, edge_dir=self.edge_dir, prob=self.prob,
        #         replace=self.replace, output_device=self.output_device,
        #         exclude_edges=exclude_eids)
        #     eid = frontier.edata[EID]
        #     block = to_block(frontier, seed_nodes)
        #     block.edata[EID] = eid
        #     seed_nodes = block.srcdata[NID]
        #     blocks.insert(0, block)
        seeds = seed_nodes
        # print("begin")
        blocks = []
        for fan_out in reversed(self.fanouts):
            coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_with_chunk_tensor(
                seeds, self.chunk_indptr, self.chunk_indices, fan_out, False)

            frontier, (coo_row, coo_col) = torch.ops.dgs_ops._CAPI_tensor_relabel(
                [seeds, coo_col], [coo_row, coo_col])

            # print(coo_row, coo_col, frontier.numel(), seeds.numel())
            blocks.insert(0, dgl.create_block((coo_col, coo_row),
                                    num_src_nodes=frontier.numel(),
                                    num_dst_nodes=seeds.numel()))
            blocks[0].srcdata[dgl.NID] = frontier
            blocks[0].dstdata[dgl.NID] = seeds
            # frontier = seeds
            seeds = frontier
        # print("end")

        return frontier, seed_nodes, blocks    
    