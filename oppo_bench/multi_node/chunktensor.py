import argparse
import time
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import dgl
from models import DistSAGE, DistGAT
from utils.load_graph import load_papers400m_sparse, load_ogb
from utils.chunktensor_sampler import *


def run(args, device, data, model, sampler):
    # Unpack data
    train_nid, g, chunk_features = data
    dataloader = dgl.dataloading.DistNodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # Training loop
    iteration_time_log = []
    for epoch in range(args.num_epochs):

        th.cuda.synchronize()
        start = time.time()
        with model.join():
            for it, (input_nodes, seeds, blocks) in enumerate(dataloader):
                batch_inputs = chunk_features._CAPI_index(input_nodes).to(
                    device)
                batch_labels = g.ndata["labels"][seeds].to(device)
                blocks = [block.to(device) for block in blocks]
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(batch_pred, batch_labels)
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
        .format(g.rank(), args.model, args.fan_out, args.bias,
                avg_iteration_time * 1000,
                args.batch_size / avg_iteration_time))


def main(args):
    dgl.distributed.initialize(args.ip_config)

    th.distributed.init_process_group(backend="gloo")

    g = dgl.distributed.DistGraph(args.graph_name,
                                  part_config=args.part_config)
    pb = g.get_partition_book()
    train_nid = dgl.distributed.node_split(g.ndata["train_mask"],
                                           pb,
                                           force_even=True)

    dev_id = g.rank() % args.num_gpu
    device = th.device("cuda:" + str(dev_id))
    th.cuda.set_device(dev_id)

    labels = g.ndata["labels"][np.arange(g.num_nodes())]
    num_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    in_feats = g.ndata["features"].shape[1]

    labels = g.ndata["labels"]
    g.ndata.clear()
    g.edata.clear()
    g.ndata["labels"] = labels

    if args.model == "graphsage":
        model = DistSAGE(in_feats, 256, num_classes)
    elif args.model == "gat":
        heads = [8, 8, 8]
        model = DistGAT(in_feats, 32, num_classes, heads)
    model = model.to(device)
    model = th.nn.parallel.DistributedDataParallel(model,
                                                   device_ids=[dev_id],
                                                   output_device=dev_id)

    # create chunk tensors
    th.cuda.reset_peak_memory_stats()

    # create each machine's process group
    th.ops.load_library(args.libdgs)
    local_group, groups = th.distributed.new_subgroups(args.num_gpu)
    create_dgs_communicator(args.num_gpu, local_group)

    available_mem = get_available_memory(
        dev_id,
        th.cuda.max_memory_reserved() +
        args.reserved_mem * 1024 * 1024 * 1024 + g.num_nodes(), local_group)

    # each group's root process load the whole graph
    rank_in_world = th.distributed.get_rank()
    if rank_in_world % args.num_gpu == 0:
        if args.graph_name == "ogbn-products":
            graph, _ = load_ogb("ogbn-products", root=args.root)
        elif args.graph_name == "ogbn-papers100M":
            graph, _ = load_ogb("ogbn-papers100M", root=args.root)
        elif args.graph_name == "ogbn-papers400M":
            graph, _ = load_papers400m_sparse(root=args.root)

        graph = graph.formats('csc')
        graph.create_formats_()

        features = graph.ndata.pop("features")
        indptr = graph.adj_sparse('csc')[0]
        indices = graph.adj_sparse('csc')[1]

        del graph
    else:
        features = None
        indptr = None
        indices = None

    # only rank 0 process generate the probs tensor, and broadcast it in the world
    if args.bias and rank_in_world == 0:
        probs = th.randn((g.num_edges(), )).float()
        broadcast_list = [probs]
    else:
        broadcast_list = [None]
    th.distributed.broadcast_object_list(broadcast_list, 0)
    probs = broadcast_list[0]

    th.distributed.barrier()

    # chunktensor cache
    if rank_in_world % args.num_gpu == 0:
        print("create chunk features")
    chunk_features = create_chunktensor(features,
                                        args.num_gpu,
                                        available_mem,
                                        cache_rate=args.feat_cache_rate,
                                        root_rank=rank_in_world -
                                        rank_in_world % args.num_gpu,
                                        local_group=local_group)
    if args.bias:
        if rank_in_world % args.num_gpu == 0:
            print("create chunk probs")
        chunk_probs = create_chunktensor(
            probs,
            args.num_gpu,
            available_mem - th.ops.dgs_ops._CAPI_get_current_allocated(),
            cache_rate=args.graph_cache_rate,
            root_rank=rank_in_world - rank_in_world % args.num_gpu,
            local_group=local_group)
    if rank_in_world % args.num_gpu == 0:
        print("create chunk indices")
    chunk_indices = create_chunktensor(
        indices,
        args.num_gpu,
        available_mem - th.ops.dgs_ops._CAPI_get_current_allocated(),
        cache_rate=args.graph_cache_rate,
        root_rank=rank_in_world - rank_in_world % args.num_gpu,
        local_group=local_group)
    if rank_in_world % args.num_gpu == 0:
        print("create chunk indptr")
    chunk_indptr = create_chunktensor(
        indptr,
        args.num_gpu,
        available_mem - th.ops.dgs_ops._CAPI_get_current_allocated(),
        cache_rate=args.graph_cache_rate,
        root_rank=rank_in_world - rank_in_world % args.num_gpu,
        local_group=local_group)

    # create chunktensor sampler
    fan_out = [int(fanout) for fanout in args.fan_out.split(',')]
    if args.bias:
        sampler = ChunkTensorSampler(fan_out, chunk_indptr, chunk_indices,
                                     chunk_probs)
    else:
        sampler = ChunkTensorSampler(fan_out, chunk_indptr, chunk_indices)

    # Pack data and run
    data = train_nid, g, chunk_features
    run(args, device, data, model, sampler)

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
