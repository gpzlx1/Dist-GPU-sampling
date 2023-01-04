import numpy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from utils import create_dgs_communicator, get_available_memory

torch.ops.load_library("./build/libdgs.so")


def compute_loading_factor(rank, world_size, valid_time_threshold, bandwidth,
                           option):
    if option not in ['local', 'remote', 'host']:
        return -1

    torch.cuda.set_device(rank)
    torch.manual_seed(rank)
    dist.init_process_group('nccl',
                            'tcp://127.0.0.1:12347',
                            world_size=world_size,
                            rank=rank)
    create_dgs_communicator(world_size, rank)

    feature_size = 100000
    feature_dim = 128
    nids_size = 2000

    valid_count = 0

    valid_factor_log = []

    while valid_count < 10:
        features = torch.ones((world_size * feature_size, feature_dim)).float()
        if option == 'local':
            cache_size = min(get_available_memory(rank, features.shape[0]),
                             features.numel() * features.element_size())
            cache_nodes_num = int(cache_size / feature_dim /
                                  features.element_size())
            chunk_features = torch.classes.dgs_classes.ChunkTensor(
                features, cache_size)
            total_nids = torch.randint(
                0, cache_nodes_num,
                (world_size * nids_size, )).unique().long().cuda()
            nids, _, _ = chunk_features._CAPI_split_index(total_nids)
            del total_nids
        elif option == 'remote':
            cache_size = min(get_available_memory(rank, features.shape[0]),
                             features.numel() * features.element_size())
            cache_nodes_num = int(cache_size / feature_dim /
                                  features.element_size())
            chunk_features = torch.classes.dgs_classes.ChunkTensor(
                features, cache_size)
            total_nids = torch.randint(
                0, cache_nodes_num,
                (world_size * nids_size, )).unique().long().cuda()
            _, nids, _ = chunk_features._CAPI_split_index(total_nids)
            del total_nids
        elif option == 'host':
            chunk_features = torch.classes.dgs_classes.ChunkTensor(features, 0)
            nids = torch.randint(0, world_size * feature_size,
                                 (nids_size, )).unique().long().cuda()
        fact_time = chunk_features._CAPI_measure_index_time(nids, option)

        if fact_time > valid_time_threshold:
            infer_time = nids.numel() * feature_dim * features.element_size(
            ) / 1024 / 1024 / 1024 / bandwidth * 1000
            valid_factor_log.append(fact_time / infer_time)
            valid_count += 1
            feature_size = int(feature_size * 1.1)
            nids_size = int(nids_size * 1.1)
        else:
            feature_size *= 10
            nids_size *= 10

    factor = numpy.mean(valid_factor_log[1:])
    all_gather_list = [None for _ in range(world_size)]
    dist.all_gather_object(all_gather_list, factor)

    if rank == 0:
        print("The penalty factor of {} loading time = {:.3f}".format(
            option, numpy.mean(all_gather_list)))


if __name__ == '__main__':
    mp.spawn(compute_loading_factor, args=(1, 1, 1300, 'local'), nprocs=1)
    mp.spawn(compute_loading_factor, args=(2, 1, 260, 'remote'), nprocs=2)
    mp.spawn(compute_loading_factor, args=(1, 5, 31, 'host'), nprocs=1)
