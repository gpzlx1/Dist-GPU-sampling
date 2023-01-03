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
