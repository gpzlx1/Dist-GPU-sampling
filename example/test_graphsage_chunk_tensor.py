import os
from random import seed
import torch
import torch.distributed as dist

torch.ops.load_library("./build/libdgs.so")
torch.ops.dgs_ops._CAPI_initialize()
torch.set_num_threads(1)
torch.cuda.set_device(torch.ops.dgs_ops._CAPI_get_rank())
os.environ["RANK"] = str(torch.ops.dgs_ops._CAPI_get_rank())
os.environ["WORLD_SIZE"] = str(torch.ops.dgs_ops._CAPI_get_size())

if "MASTER_ADDR" not in os.environ:
    os.environ["MASTER_ADDR"] = "localhost"
if "MASTER_PORT" not in os.environ:
    os.environ["MASTER_PORT"] = "12335"

dist.init_process_group(backend='nccl', init_method="env://")

indptr = torch.arange(0, 1001).int().cuda() * 5
indices = torch.arange(0, 5000).int().cuda()
seeds = torch.randint(0, 1000, (5, )).int().cuda().unique()

print(indptr)
print(indices)
print(seeds)

# cache 320 / 4 = 80 int
chunk_indptr = torch.classes.dgs_classes.ChunkTensor(indptr, 500)
print(chunk_indptr._CAPI_get_host_tensor())
print(chunk_indptr._CAPI_get_sub_device_tensor())

chunk_indices = torch.classes.dgs_classes.ChunkTensor(indices, 500)
print(chunk_indices._CAPI_get_host_tensor())
print(chunk_indices._CAPI_get_sub_device_tensor())

print("begin")

for fan_out in [5]:
    coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors(
        seeds, indptr, indices, fan_out, False)
    frontier, (coo_row, coo_col) = torch.ops.dgs_ops._CAPI_tensor_relabel(
        [seeds, coo_col], [coo_row, coo_col])
    #block = dgl.create_block((coo_row, coo_col),
    #                         num_src_nodes=frontier.numel(),
    #                         num_dst_nodes=seeds.numel())
    frontier = seeds

torch.ops.dgs_ops._CAPI_finalize()
