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

indptr = torch.tensor([0, 10, 11, 20, 22, 22]).int()
indices = torch.arange(22).int()
probs = torch.ones_like(indices).float()
probs[0] = 100
train_nid = torch.tensor([2, 3, 1, 0]).int().cuda()

print(indptr)
print(indices)
print(probs)
print(train_nid)

chunk_indptr = torch.classes.dgs_classes.ChunkTensor(indptr, 12)
print(chunk_indptr._CAPI_get_host_tensor())
print(chunk_indptr._CAPI_get_sub_device_tensor())

chunk_indices = torch.classes.dgs_classes.ChunkTensor(indices, 44)
print(chunk_indices._CAPI_get_host_tensor())
print(chunk_indices._CAPI_get_sub_device_tensor())

chunk_probs = torch.classes.dgs_classes.ChunkTensor(probs, 44)
print(chunk_probs._CAPI_get_host_tensor())
print(chunk_probs._CAPI_get_sub_device_tensor())

print("begin")

seeds = train_nid
for fan_out in [5]:
    coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_with_probs_with_chunk_tensor(
        seeds, chunk_indptr, chunk_indices, chunk_probs, fan_out, False)
    print(coo_row)
    print(coo_col)
    frontier, (coo_row, coo_col) = torch.ops.dgs_ops._CAPI_tensor_relabel(
        [seeds, coo_col], [coo_row, coo_col])
    #block = dgl.create_block((coo_row, coo_col),
    #                         num_src_nodes=frontier.numel(),
    #                         num_dst_nodes=seeds.numel())
    seeds = frontier

seeds = train_nid
for fan_out in [5]:
    coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_with_probs_with_chunk_tensor(
        seeds, chunk_indptr, chunk_indices, chunk_probs, fan_out, True)
    print(coo_row)
    print(coo_col)
    frontier, (coo_row, coo_col) = torch.ops.dgs_ops._CAPI_tensor_relabel(
        [seeds, coo_col], [coo_row, coo_col])
    #block = dgl.create_block((coo_row, coo_col),
    #                         num_src_nodes=frontier.numel(),
    #                         num_dst_nodes=seeds.numel())
    seeds = frontier

del chunk_indptr
del chunk_indices
del chunk_probs

torch.ops.dgs_ops._CAPI_finalize()
