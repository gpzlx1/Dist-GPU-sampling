import argparse
import time
import numpy as np
import torch as th
import os
import torch.nn as nn
import torch.optim as optim
from models import DistSAGE, DistGAT
from utils.chunktensor_sampler import *
from utils.dataloader import SeedGenerator

from dist_graph import DistGraph

# Set environment variables
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])


def run(args, device, dist_graph, model):
    # create chunktensor sampler
    fan_out = [int(fanout) for fanout in args.fan_out.split(',')]
    if args.bias:
        sampler = ChunkTensorSampler(fan_out, dist_graph.chunk_indptr,
                                     dist_graph.chunk_indices,
                                     dist_graph.edata['probs'])
    else:
        sampler = ChunkTensorSampler(fan_out, dist_graph.chunk_indptr,
                                     dist_graph.chunk_indices)

    # Unpack data
    train_nid = dist_graph.ndata['train_ids']._CAPI_get_host_tensor()
    part = (train_nid.numel() + WORLD_SIZE - 1) // WORLD_SIZE
    train_nid = train_nid[part * WORLD_RANK:part * (WORLD_RANK + 1)]

    train_seedloader = SeedGenerator(train_nid,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     drop_last=False)

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # Training loop
    print("Begin Training")
    iteration_time_log = []
    for epoch in range(args.num_epochs):
        th.cuda.synchronize()
        start = time.time()
        with model.join():
            for it, seeds in enumerate(train_seedloader):
                frontier, seed_nodes, blocks = sampler.sample_blocks(seeds)
                frontier = frontier.cuda()
                seed_nodes = seed_nodes.cuda()
                blocks = [block.to(device) for block in blocks]
                batch_inputs = dist_graph.ndata['features']._CAPI_index(
                    frontier).to(device)
                batch_labels = dist_graph.ndata['labels']._CAPI_index(
                    seed_nodes).to(device).reshape(-1)
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(batch_pred, batch_labels.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                th.cuda.synchronize()
                end = time.time()
                iteration_time_log.append(end - start)

                start = time.time()

    avg_iteration_time = np.mean(iteration_time_log[5:])
    print(
        "Part {} | Model {} | Fan out {} | Sampling with bias {} | Iteration Time {:.4f} ms | Throughput {:.3f} seeds/sec"
        .format(dist.get_rank(), args.model, args.fan_out, args.bias,
                avg_iteration_time * 1000,
                args.batch_size / avg_iteration_time))


def main(args):
    dist.init_process_group(backend='nccl',
                            init_method="env://",
                            rank=WORLD_RANK,
                            world_size=WORLD_SIZE)
    th.ops.load_library(args.libdgs)
    local_group, groups = th.distributed.new_subgroups(args.num_gpu)

    dev_id = dist.get_rank(local_group)
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.cuda.current_device()

    create_dgs_communicator(args.num_gpu, local_group)

    if args.bias:
        dg = DistGraph(args.libdgs, args.root, args.graph_name,
                       WORLD_RANK - WORLD_RANK % args.num_gpu, WORLD_RANK,
                       local_group, ['ndata/features', 'edata/probs'],
                       args.feat_cache_rate, args.graph_cache_rate, args.bias)
    else:
        dg = DistGraph(args.libdgs, args.root, args.graph_name,
                       WORLD_RANK - WORLD_RANK % args.num_gpu, WORLD_RANK,
                       local_group, ['ndata/features'], args.feat_cache_rate,
                       args.graph_cache_rate, args.bias)

    if args.model == "graphsage":
        model = DistSAGE(dg.metadata['ndata/features'][1][1], 256,
                         dg.metadata['num_labels'])
    elif args.model == "gat":
        heads = [8, 8, 8]
        model = DistGAT(dg.metadata['ndata/features'][1][1], 32,
                        dg.metadata['num_labels'], heads)
    model = model.to(device)
    model = th.nn.parallel.DistributedDataParallel(model,
                                                   device_ids=[dev_id],
                                                   output_device=dev_id)

    dg.create_shared_graph()

    # Pack data and run
    data = dg
    run(args, device, data, model)

    for group in groups:
        th.distributed.destroy_process_group(group)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_name", type=str, help="graph name")
    parser.add_argument("--id", type=int, help="the partition id")
    parser.add_argument("--ip_config",
                        type=str,
                        help="The file for IP configuration")
    parser.add_argument("--num_gpu",
                        type=int,
                        default=1,
                        help="the number of GPU device.")
    parser.add_argument("--part_config",
                        type=str,
                        help="The path to the partition config file")
    parser.add_argument("--model",
                        default="graphsage",
                        choices=["graphsage", "gat"],
                        help="The model of training.")
    parser.add_argument("--bias",
                        action="store_true",
                        default=False,
                        help="Sample with bias.")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument('--fan_out', type=str, default='15,15,15')
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument('--root',
                        default='dataset/',
                        help='Path of the dataset.')
    parser.add_argument("--local_rank",
                        type=int,
                        help="get rank of the process")
    parser.add_argument('--libdgs',
                        default='../Dist-GPU-sampling/build/libdgs.so',
                        help='Path of libdgs.so')
    parser.add_argument(
        '--feat-cache-rate',
        default='1',
        type=float,
        help=
        'The gpu cache rate of features. If the gpu memory is not enough, cache priority: features > probs > indices > indptr'
    )
    parser.add_argument(
        '--graph-cache-rate',
        default='1',
        type=float,
        help=
        'The gpu cache rate of graph structure tensors. If the gpu memory is not enough, cache priority: features > probs > indices > indptr'
    )
    parser.add_argument(
        '--reserved-mem',
        default='3',
        type=float,
        help="The size of reserved memory (unit: GB) for model and training.")
    args = parser.parse_args()

    main(args)
