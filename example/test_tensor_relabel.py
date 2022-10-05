import torch

torch.ops.load_library("./build/libdgs.so")

a = torch.randint(0, 100, (1, )).cuda()
b = torch.randint(0, 100, (2, )).cuda()
c = torch.randint(0, 100, (3, )).cuda()

print(a)
print(b)
print(c)

for i in torch.ops.dgs_ops._CAPI_tensor_relabel([a, b, c], [b, c]):
    print(i)