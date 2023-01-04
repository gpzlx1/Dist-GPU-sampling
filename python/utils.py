import torch
import torch.distributed as dist


def create_dgs_communicator(world_size, local_rank):
    if local_rank == 0:
        unique_id_array = torch.ops.dgs_ops._CAPI_get_unique_id()
        broadcast_list = [unique_id_array]
    else:
        broadcast_list = [None]

    dist.broadcast_object_list(broadcast_list, 0)
    unique_ids = broadcast_list[0]
    torch.ops.dgs_ops._CAPI_set_nccl(world_size, unique_ids, local_rank)


def get_available_memory(device, num_node):
    available_mem = torch.cuda.mem_get_info(
        device)[1] - torch.ops.dgs_ops._CAPI_get_current_allocated(
        ) - torch.cuda.max_memory_allocated(
        ) - 0.3 * torch.cuda.max_memory_reserved(
        ) - 4 * 1024 * 1024 * 1024 - num_node
    available_mem = torch.tensor([available_mem]).long().cuda()
    dist.all_reduce(available_mem, dist.ReduceOp.MIN)
    return available_mem.cpu().numpy()[0]
