import torch

torch.ops.load_library("./build/libdgs.so")

indptr = torch.arange(0, 1001).int().cuda() * 5
indices = torch.arange(0, 5000).int().cuda()

seeds = torch.randint(0, 1000, (5, )).int().cuda().unique()

print(indptr)
print(indices)
print(seeds)

for fan_out in [5]:
    coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors(
        seeds, indptr, indices, fan_out, False)
    frontier, (coo_row, coo_col) = torch.ops.dgs_ops._CAPI_tensor_relabel(
        [seeds, coo_col], [coo_row, coo_col])
    #block = dgl.create_block((coo_row, coo_col),
    #                         num_src_nodes=frontier.numel(),
    #                         num_dst_nodes=seeds.numel())
    seeds = frontier
