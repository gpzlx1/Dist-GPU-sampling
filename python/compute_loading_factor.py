import numpy
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from utils import create_dgs_communicator, get_available_memory

torch.ops.load_library("./build/libdgs.so")


def compute_loading_factor(rank, world_size, features, valid_time_threshold,
                           bandwidth, option):
    if option not in ['local', 'remote', 'host']:
        return -1

    torch.cuda.set_device(rank)
    torch.manual_seed(rank)
    dist.init_process_group('nccl',
                            'tcp://127.0.0.1:12347',
                            world_size=world_size,
                            rank=rank)
    create_dgs_communicator(world_size, rank)

    feature_size = features.shape[0]
    feature_dim = features.shape[1]

    if option == 'local' or option == 'remote':
        cache_size = min(get_available_memory(rank, feature_size),
                         features.numel() * features.element_size())
        cache_nodes_num = int(cache_size / feature_dim /
                              features.element_size())
        chunk_features = torch.classes.dgs_classes.ChunkTensor(
            features.shape, features.dtype, cache_size)
        chunk_features._CAPI_load_from_tensor(features)
    elif option == 'host':
        chunk_features = torch.classes.dgs_classes.ChunkTensor(
            features.shape, features.dtype, 0)
        chunk_features._CAPI_load_from_tensor(features)

    nids_size = 2000

    valid_count = 0
    valid_factor_log = []

    while valid_count < 10:
        fact_time_log = []
        for _ in range(10):
            if option == 'local':
                total_nids = torch.randint(
                    0, cache_nodes_num, (nids_size, )).unique().long().cuda()
                nids, _, _ = chunk_features._CAPI_split_index(total_nids)
                del total_nids
                torch.cuda.synchronize()
                start = time.time()
                _ = chunk_features._CAPI_local_index(nids)
                torch.cuda.synchronize()
                fact_time_log.append(time.time() - start)

            elif option == 'remote':
                total_nids = torch.randint(
                    0, cache_nodes_num, (nids_size, )).unique().long().cuda()
                _, nids, _ = chunk_features._CAPI_split_index(total_nids)
                del total_nids
                torch.cuda.synchronize()
                start = time.time()
                _ = chunk_features._CAPI_remote_index(nids)
                torch.cuda.synchronize()
                fact_time_log.append(time.time() - start)

            elif option == 'host':
                nids = torch.randint(0, feature_size,
                                     (nids_size, )).unique().long().cuda()
                torch.cuda.synchronize()
                start = time.time()
                _ = chunk_features._CAPI_host_index(nids)
                torch.cuda.synchronize()
                fact_time_log.append(time.time() - start)

        fact_time = numpy.mean(fact_time_log[3:]) * 1000

        if fact_time > valid_time_threshold:
            infer_time = nids.numel() * feature_dim * features.element_size(
            ) / 1024 / 1024 / 1024 / bandwidth * 1000
            valid_factor_log.append(fact_time / infer_time)
            valid_count += 1
            print(
                "rank {}, valid {}, #nids = {}, infer time = {:.2f}, fact_time = {:.2f}, penalty factor = {:.3f}"
                .format(rank, valid_count, nids.numel(), infer_time, fact_time,
                        fact_time / infer_time))
            nids_size = int(nids_size * 1.2)

        else:
            nids_size *= 10

        if nids_size > feature_size:
            break

    factor = numpy.mean(valid_factor_log)
    all_gather_list = [None for _ in range(world_size)]
    dist.all_gather_object(all_gather_list, factor)

    if rank == 0:
        print(
            "#GPU = {}, the penalty factor of {} loading time = {:.3f}".format(
                world_size, option, numpy.mean(all_gather_list)))


if __name__ == '__main__':
    feature_dim = 128
    feature_size = 10000000

    print("graph num nodes", feature_size)
    print("feature dim", feature_dim)

    world_size = 1
    features = torch.ones((world_size * feature_size, feature_dim)).float()
    mp.spawn(compute_loading_factor,
             args=(world_size, features, 1, 1300, 'local'),
             nprocs=world_size)

    world_size = 2
    features = torch.ones((world_size * feature_size, feature_dim)).float()
    mp.spawn(compute_loading_factor,
             args=(world_size, features, 1, 260, 'remote'),
             nprocs=world_size)

    world_size = 1
    features = torch.ones((world_size * feature_size, feature_dim)).float()
    mp.spawn(compute_loading_factor,
             args=(world_size, features, 5, 31, 'host'),
             nprocs=world_size)
