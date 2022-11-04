import torch
import torch.distributed as dist
from dgs_create_communicator import create_dgs_communicator

torch.ops.load_library("./build/libdgs.so")

dist.init_process_group(backend='nccl', init_method="env://")
torch.set_num_threads(1)
torch.cuda.set_device(dist.get_rank())

create_dgs_communicator(dist.get_world_size(), dist.get_rank())

print(torch.ops.dgs_ops._CAPI_get_current_allocated())
data = torch.arange(300).long()

print(torch.ops.dgs_ops._CAPI_get_current_allocated())
c_tensor = torch.classes.dgs_classes.ChunkTensor(data, 30000)

print(torch.ops.dgs_ops._CAPI_get_current_allocated())

del c_tensor
print(torch.ops.dgs_ops._CAPI_get_current_allocated())