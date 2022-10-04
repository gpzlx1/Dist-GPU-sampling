import torch

torch.ops.load_library("./build/libdgs.so")

data = torch.ones(10)
c_tensor = torch.classes.dgs_classes.ChunkTensor(data)
print(c_tensor._CAPI_get_data())
print(c_tensor._CAPI_get_global_data())