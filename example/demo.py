import time
import torch
import torch.distributed as dist
from dgs_create_communicator import create_dgs_communicator
import os

num_nodes = 1

# create global group
dist.init_process_group(backend='nccl', init_method="env://")
# create local group
if num_nodes == 1:
    local_subgroup = None
else:
    local_subgroup, subgroups = dist.new_subgroups()

torch.set_num_threads(1)
torch.cuda.set_device(dist.get_rank(local_subgroup))

create_dgs_communicator(dist.get_world_size(local_subgroup),
                        dist.get_rank(local_subgroup), local_subgroup)

print("[rank={}] Create ChunkTensor".format(dist.get_rank(local_subgroup)))
c_a = torch.classes.dgs_classes.ChunkTensor([100], torch.int64, 200)
if dist.get_rank(local_subgroup) == 0:
    print("[rank={}] Load data".format(dist.get_rank(local_subgroup)))
    a = torch.arange(100).long()
    c_a._CAPI_load_from_tensor(a)

print("[rank={}] Print HostTensor in ChunkTensor:".format(
    dist.get_rank(local_subgroup)))
print(c_a._CAPI_get_host_tensor())

print("[rank={}] Print DeviceTensor in ChunkTensor:".format(
    dist.get_rank(local_subgroup)))
print(c_a._CAPI_get_sub_device_tensor())