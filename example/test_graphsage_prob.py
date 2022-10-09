import torch

torch.ops.load_library("./build/libdgs.so")

indptr = torch.tensor([0, 10, 11, 20, 22, 22]).cuda()
indices = torch.arange(22).cuda()
probs = torch.ones_like(indices).cuda().float()
probs[0] = 100
train_nid = torch.tensor([2, 3, 1, 0]).cuda()

seeds = train_nid
for fan_out in [5]:
    coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_with_probs(
        seeds, indptr, indices, probs, fan_out, True)
    print(coo_row)
    print(coo_col)
    frontier, (coo_row, coo_col) = torch.ops.dgs_ops._CAPI_tensor_relabel(
        [seeds, coo_col], [coo_row, coo_col])
    #block = dgl.create_block((coo_row, coo_col),
    #                         num_src_nodes=frontier.numel(),
    #                         num_dst_nodes=seeds.numel())
    seeds = frontier

seeds = train_nid
for fan_out in [5]:
    coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_with_probs(
        seeds, indptr, indices, probs, fan_out, False)
    print(coo_row)
    print(coo_col)
    frontier, (coo_row, coo_col) = torch.ops.dgs_ops._CAPI_tensor_relabel(
        [seeds, coo_col], [coo_row, coo_col])
    #block = dgl.create_block((coo_row, coo_col),
    #                         num_src_nodes=frontier.numel(),
    #                         num_dst_nodes=seeds.numel())
    seeds = frontier

