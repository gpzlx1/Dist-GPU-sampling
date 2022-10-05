import torch
import torch.distributed as dist

dist.init_process_group(backend='nccl')
torch.cuda.set_device(dist.get_rank())

torch.ops.load_library("./build/libdgs.so")

torch.ops.dgs_ops._CAPI_initialize()
data = torch.arange(10).long()
c_tensor = torch.classes.dgs_classes.ChunkTensor(data, 16)

print(c_tensor._CAPI_get_host_tensor())
print(c_tensor._CAPI_get_sub_device_tensor())

torch.ops.dgs_ops._CAPI_finalize()