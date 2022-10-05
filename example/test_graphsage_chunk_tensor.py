import os
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
chunk_indptr = torch.classes.dgs_classes.ChunkTensor(indptr, 320)
chunk_indices = torch.classes.dgs_classes.ChunkTensor(indices, 320)

print("begin")

for fan_out in [5]:
    coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_with_chunk_tensor(
        seeds, chunk_indptr, chunk_indices, fan_out, False)

torch.ops.dgs_ops._CAPI_finalize()
