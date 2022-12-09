import time
import torch
import torch.distributed as dist
from dgs_create_communicator import create_dgs_communicator

torch.ops.load_library("./build/libdgs.so")

dist.init_process_group(backend='nccl', init_method="env://")
torch.set_num_threads(1)
torch.cuda.set_device(dist.get_rank())

create_dgs_communicator(dist.get_world_size(), dist.get_rank())

data = torch.arange(10).long()
c_tensor = torch.classes.dgs_classes.ChunkTensor(data, 32)

print(c_tensor._CAPI_get_host_tensor())
print(c_tensor._CAPI_get_sub_device_tensor())

index = torch.arange(0, 10).cuda()
local, remote, host = c_tensor._CAPI_split_index(index)

if dist.get_rank() == 0:
    print("from rank == 0")
    print("index to be split:", index)
    print("local:", local)
    print("remote:", remote)
    print("host:", host)

time.sleep(3)

if dist.get_rank() == 1:
    print("from rank == 1")
    print("index to be split:", index)
    print("local:", local)
    print("remote:", remote)
    print("host:", host)