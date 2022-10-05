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


data = torch.arange(10).long()
c_tensor = torch.classes.dgs_classes.ChunkTensor(data, 32)

print(c_tensor._CAPI_get_host_tensor())
print(c_tensor._CAPI_get_sub_device_tensor())

torch.ops.dgs_ops._CAPI_finalize()
