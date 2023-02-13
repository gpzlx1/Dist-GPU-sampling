import time
import torch
import torch.distributed as dist
from dgs_create_communicator import create_dgs_communicator
import os

dist.init_process_group(backend='nccl', init_method="env://")
torch.set_num_threads(1)
torch.cuda.set_device(dist.get_rank())

create_dgs_communicator(dist.get_world_size(), dist.get_rank())

c_a = torch.classes.dgs_classes.ChunkTensor([100], torch.int64, 400)
if dist.get_rank() == 0:
    a = torch.arange(100).long()
    c_a._CAPI_load_from_tensor(a)

print(c_a._CAPI_get_host_tensor())
