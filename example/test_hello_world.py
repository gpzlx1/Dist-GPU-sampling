import torch

torch.ops.load_library("./build/libdgs.so")

torch.ops.dgs_ops._CAPI_hello_world(10)