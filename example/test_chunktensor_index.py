import time
import torch
import torch.distributed as dist
from dgs_create_communicator import create_dgs_communicator

torch.ops.load_library("./build/libdgs.so")

dist.init_process_group(backend='nccl', init_method="env://")
torch.set_num_threads(1)
torch.cuda.set_device(dist.get_rank())

create_dgs_communicator(dist.get_world_size(), dist.get_rank())

data = torch.arange(10).int()
c_tensor = torch.classes.dgs_classes.ChunkTensor(data, 32)
index = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).int().cuda()
print(c_tensor._CAPI_index(index))