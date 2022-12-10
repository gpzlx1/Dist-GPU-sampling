import time
import torch
import torch.distributed as dist
from dgs_create_communicator import create_dgs_communicator

torch.ops.load_library("./build/libdgs.so")

dist.init_process_group(backend='nccl', init_method="env://")
torch.set_num_threads(1)
torch.cuda.set_device(dist.get_rank())

create_dgs_communicator(dist.get_world_size(), dist.get_rank())

data = torch.arange(10).repeat_interleave(5).float().reshape(-1, 5) * 2

c_tensor = torch.classes.dgs_classes.ChunkTensor(data, 80)

print(c_tensor._CAPI_get_host_tensor().reshape(-1, 5))
print(c_tensor._CAPI_get_sub_device_tensor().reshape(-1, 5))

time.sleep(1)

index = torch.tensor([2, 4, 7, 0, 9]).cuda()
local, remote, host = c_tensor._CAPI_split_index(index)

local_data = c_tensor._CAPI_local_index(local)
remote_data = c_tensor._CAPI_remote_index(remote)
host_data = c_tensor._CAPI_host_index(host)

if dist.get_rank() == 0:
    print("from rank == 0")
    print("index to be split:", index)
    print("local:", local)
    print("remote:", remote)
    print("host:", host)
    print("local data:", local_data)
    print("remote data:", remote_data)
    print("host data:", host_data)

time.sleep(1)

if dist.get_rank() == 1:
    print("from rank == 1")
    print("index to be split:", index)
    print("local:", local)
    print("remote:", remote)
    print("host:", host)
    print("local data:", local_data)
    print("remote data:", remote_data)
    print("host data:", host_data)