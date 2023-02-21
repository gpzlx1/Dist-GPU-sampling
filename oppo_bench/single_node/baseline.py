import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.optim
import torchmetrics.functional as MF
import dgl
import time
import numpy as np
from models import SAGE, GAT
from utils.load_graph import load_papers400m_sparse, load_ogb


def run(rank, world_size, data, args):
    graph, num_classes = data

    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',
                            'tcp://127.0.0.1:12347',
                            world_size=world_size,
                            rank=rank)

    if args.model == 'graphsage':
        model = SAGE(graph.ndata['features'].shape[1], 256, num_classes)
    elif args.model == 'gat':
        heads = [8, 8, 8]
        model = GAT(graph.ndata['features'].shape[1], 32, num_classes, heads)
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[rank],
                                                output_device=rank)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    train_idx = graph.nodes()[graph.ndata['train_mask'].bool()]

    # move ids to GPU
    train_idx = train_idx.to('cuda')

    if args.bias:
        # bias sampling
        sampler = dgl.dataloading.NeighborSampler(
            [int(fanout) for fanout in args.fan_out.split(',')],
            prob='probs',
            prefetch_node_feats=['features'],
            prefetch_labels=['labels'])
    else:
        # uniform sampling
        sampler = dgl.dataloading.NeighborSampler(
            [int(fanout) for fanout in args.fan_out.split(',')],
            prefetch_node_feats=['features'],
            prefetch_labels=['labels'])

    train_dataloader = dgl.dataloading.DataLoader(graph,
                                                  train_idx,
                                                  sampler,
                                                  device='cuda',
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  drop_last=False,
                                                  num_workers=0,
                                                  use_ddp=True,
                                                  use_uva=True)

    if rank == 0:
        print('start training...')
    iteration_time_log = []
    for epoch in range(args.num_epochs):
        model.train()

        torch.cuda.synchronize()
        start = time.time()
        for it, (input_nodes, output_nodes,
                 blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata['features']
            y = blocks[-1].dstdata['labels']
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y.long())
            opt.zero_grad()
            loss.backward()
            opt.step()

            torch.cuda.synchronize()
            end = time.time()
            iteration_time_log.append(end - start)

            if it % 20 == 0 and rank == 0 and args.print_train:
                acc = MF.accuracy(y_hat,
                                  y,
                                  task='multiclass',
                                  num_classes=num_classes)
                print('Epoch {} | Iteration {} | Loss {} | Acc {}'.format(
                    epoch, it, loss.item(), acc.item()))

            torch.cuda.synchronize()
            start = time.time()

    avg_iteration_time = np.mean(iteration_time_log[5:])
    all_gather_list = [None for _ in range(world_size)]
    dist.all_gather_object(all_gather_list, avg_iteration_time)
    avg_iteration_time = np.mean(all_gather_list)
    throughput = args.batch_size * world_size / avg_iteration_time.item()
    if rank == 0:
        print('Time per iteration {:.3f} ms | Throughput {:.3f} seeds/sec'.
              format(avg_iteration_time * 1000, throughput))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpu',
                        default='8',
                        type=int,
                        help='The number GPU participated in the training.')
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument('--root',
                        default='dataset/',
                        help='Path of the dataset.')
    parser.add_argument('--model',
                        default='graphsage',
                        choices=['graphsage', 'gat'],
                        help='The model of training.')
    parser.add_argument("--batch-size",
                        default="1000",
                        type=int,
                        help="The number of seeds of sampling.")
    parser.add_argument('--fan-out', type=str, default='15,15,15')
    parser.add_argument('--bias',
                        action='store_true',
                        default=False,
                        help="Sample with bias.")
    parser.add_argument(
        "--dataset",
        default="ogbn-papers400M",
        choices=["ogbn-products", "ogbn-papers100M", "ogbn-papers400M"])
    parser.add_argument('--print-train',
                        action='store_true',
                        default=False,
                        help="Whether to print loss and acc during training.")
    args = parser.parse_args()

    if args.dataset == "ogbn-products":
        graph, num_classes = load_ogb("ogbn-products", root=args.root)
    elif args.dataset == "ogbn-papers100M":
        graph, num_classes = load_ogb("ogbn-papers100M", root=args.root)
    elif args.dataset == "ogbn-papers400M":
        graph, num_classes = load_papers400m_sparse(root=args.root)

    graph = graph.formats('csc')
    graph.create_formats_()
    graph.edata.clear()

    n_procs = min(args.num_gpu, torch.cuda.device_count())

    if args.bias:
        graph.edata['probs'] = torch.randn((graph.num_edges(), )).float()

    print(
        'Dataset {} | GPU num {} | Model {} | Fan out {} | Batch size {} | Bias sampling {}'
        .format(args.dataset, n_procs, args.model,
                [int(fanout) for fanout in args.fan_out.split(',')],
                args.batch_size, args.bias))

    data = graph, num_classes

    import torch.multiprocessing as mp
    mp.spawn(run, args=(n_procs, data, args), nprocs=n_procs)
