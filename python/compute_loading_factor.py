import numpy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dgs_create_communicator import create_dgs_communicator

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
    nids_size = 50000

    valid_count = 0

    valid_factor_log = []

    while valid_count < 15:
        features = torch.ones((world_size * feature_size, feature_dim)).float()
        nids = torch.randint(rank * feature_size, (rank + 1) * feature_size,
                             (nids_size, )).unique().long().cuda()
        if option == 'local' or option == 'remote':
            chunk_features = torch.classes.dgs_classes.ChunkTensor(
                features,
                features.numel() * features.element_size())
        elif option == 'host':
            chunk_features = torch.classes.dgs_classes.ChunkTensor(features, 0)

        fact_time = chunk_features._CAPI_measure_index_time(nids, option)

        if fact_time > valid_time_threshold:
            infer_time = nids.numel() * feature_dim * features.element_size(
            ) / 1024 / 1024 / 1024 / bandwidth * 1000
            valid_factor_log.append(fact_time / infer_time)

            valid_count += 1
        else:
            feature_size *= 10
            nids_size *= 10

    factor = numpy.mean(valid_factor_log[3:])
    all_gather_list = [None for _ in range(world_size)]
    dist.all_gather_object(all_gather_list, factor)

    if rank == 0:
        print("The penalty factor of {} loading time = {:.3f}".format(
            option, numpy.mean(all_gather_list)))


if __name__ == '__main__':
    mp.spawn(compute_loading_factor, args=(1, 1, 1300, 'local'), nprocs=1)
    mp.spawn(compute_loading_factor, args=(2, 1, 260, 'remote'), nprocs=2)
    mp.spawn(compute_loading_factor, args=(1, 5, 31, 'host'), nprocs=1)
