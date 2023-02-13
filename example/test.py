import time
import torch
import torch.distributed as dist
from dgs_create_communicator import create_dgs_communicator
import os

torch.ops.load_library("./build/libdgs.so")
os.environ["RANK"] = str(0)
os.environ["WORLD_SIZE"] = str(1)
if "MASTER_ADDR" not in os.environ:
    os.environ["MASTER_ADDR"] = "localhost"
if "MASTER_PORT" not in os.environ:
    os.environ["MASTER_PORT"] = "12335"

dist.init_process_group(backend='nccl', init_method="env://")
torch.set_num_threads(1)
torch.cuda.set_device(dist.get_rank())

create_dgs_communicator(dist.get_world_size(), dist.get_rank())

c_a = torch.classes.dgs_classes.ChunkTensor([100], torch.int64, 400)

a = torch.arange(100).long()

c_a._CAPI_load_from_tensor(a)

print(c_a._CAPI_get_host_tensor())
