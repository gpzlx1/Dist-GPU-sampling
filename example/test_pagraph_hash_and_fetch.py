import torch

torch.ops.load_library("./build/libdgs.so")

nid = torch.arange(10).cuda()
cpu_data = torch.arange(10 * 5).float().reshape([10, 5])

gpu_cached_mask = torch.tensor([True, False]).bool().repeat(5)

gpu_data = cpu_data[gpu_cached_mask].cuda()
gpu_cached_nid = nid[gpu_cached_mask]
nid_in_gpu = torch.arange(gpu_cached_nid.numel()).cuda()

print("nid", nid)
print("gpu_cached_nid", gpu_cached_nid)
print("nid_in_gpu", nid_in_gpu)
print("cpu_data", cpu_data)
print("gpu_data", gpu_data)

cpu_data = cpu_data.pin_memory()

hashed_key, hashed_val = torch.ops.dgs_ops._CAPI_create_hash_map(
    gpu_cached_nid, nid_in_gpu)

print("hashed_key", hashed_key)
print("hashed_val", hashed_val)

fetched_data = torch.ops.dgs_ops._CAPI_fetch_data(cpu_data, gpu_data, nid,
                                                  hashed_key, hashed_val)

print("fetched_data", fetched_data)
