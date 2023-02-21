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
from utils.chunktensor_sampler import *
from utils.compresser import Compresser


def run(rank, world_size, data, args, compresser):
    graph, num_classes, feature_dim = data

    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',
                            'tcp://127.0.0.1:12347',
                            world_size=world_size,
                            rank=rank)
    torch.ops.load_library(args.libdgs)
    if args.compress_feat:
        torch.ops.load_library(args.libbifeat)

    if args.model == 'graphsage':
        model = SAGE(feature_dim, 256, num_classes)
    elif args.model == 'gat':
        heads = [8, 8, 8]
        model = GAT(feature_dim, 32, num_classes, heads)
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[rank],
                                                output_device=rank)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    train_idx = graph.nodes()[graph.ndata['train_mask'].bool()]

    # move ids to GPU
    train_idx = train_idx.to('cuda')

    if args.model == 'graphsage':
        available_mem = get_available_memory(
            rank,
            torch.cuda.max_memory_reserved() + 3 * 1024 * 1024 * 1024 +
            graph.num_nodes())
    elif args.model == 'gat':
        available_mem = get_available_memory(
            rank,
            torch.cuda.max_memory_reserved() + 11 * 1024 * 1024 * 1024 +
            graph.num_nodes())

    create_dgs_communicator(world_size)

    if not args.compress_feat and not args.gpu_cache_full_feat:
        # chunktensor cache features
        if dist.get_rank() == 0:
            print("create chunk features")
        features = graph.ndata.pop('features')
        chunk_features = create_chunktensor(features, world_size,
                                            available_mem,
                                            args.feat_cache_rate)
        prefetch_node_feats = None
    else:
        if args.gpu_cache_full_feat:
            features = graph.ndata.pop('features').cuda()
            features_size = features.numel() * features.element_size()
            prefetch_node_feats = None
            if dist.get_rank() == 0:
                print("Every GPU cache the full features, size = {:.3f} GB".
                      format(features_size / 1024 / 1024 / 1024))
            available_mem = max(available_mem - features_size, 0)
        else:
            prefetch_node_feats = ['features']

    # chunktensor cache probs, indices, indptr
    if args.bias:
        if dist.get_rank() == 0:
            print("create chunk probs")
        probs = graph.edata.pop('probs')
        chunk_probs = create_chunktensor(
            probs, world_size,
            max(
                available_mem -
                torch.ops.dgs_ops._CAPI_get_current_allocated(), 0),
            args.graph_cache_rate)
    if dist.get_rank() == 0:
        print("create chunk indices")
    indices = graph.adj_sparse('csc')[1]
    chunk_indices = create_chunktensor(
        indices, world_size,
        max(available_mem - torch.ops.dgs_ops._CAPI_get_current_allocated(),
            0), args.graph_cache_rate)
    if dist.get_rank() == 0:
        print("create chunk indptr")
    indptr = graph.adj_sparse('csc')[0]
    chunk_indptr = create_chunktensor(
        indptr, world_size,
        max(available_mem - torch.ops.dgs_ops._CAPI_get_current_allocated(),
            0), args.graph_cache_rate)

    # create chunktensor sampler
    if args.bias:
        # bias sampling
        sampler = ChunkTensorSampler(
            [int(fanout) for fanout in args.fan_out.split(',')],
            chunk_indptr,
            chunk_indices,
            chunk_probs=chunk_probs,
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=['labels'])
    else:
        # uniform sampling
        sampler = ChunkTensorSampler(
            [int(fanout) for fanout in args.fan_out.split(',')],
            chunk_indptr,
            chunk_indices,
            prefetch_node_feats=prefetch_node_feats,
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
            if args.compress_feat:
                if args.gpu_cache_full_feat:
                    x = features.index_select(0, input_nodes)
                else:
                    x = blocks[0].srcdata['features']
                x = compresser.decompress(x, "cuda")
            else:
                x = chunk_features._CAPI_index(input_nodes)
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
    parser.add_argument('--libdgs',
                        default='../Dist-GPU-sampling/build/libdgs.so',
                        help='Path of libdgs.so')
    parser.add_argument(
        '--libbifeat',
        default='../GPU-Feature-Quantization/build/libbifeat.so',
        help='Path of libbifeat.so')
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
        "--compress-feat",
        action='store_true',
        default=False,
        help=
        "Compress features offline and decompress when loading. If this flag is set true, the features will not be cached by chunktensor."
    )
    parser.add_argument('--compress-mode',
                        type=str,
                        default='sq',
                        choices=['vq', 'sq'])
    parser.add_argument('--compress-width', type=int, default=1)
    parser.add_argument('--compress-length', type=int, default=4)
    parser.add_argument("--compress-feat-save-root",
                        type=str,
                        default=None,
                        help="The directory to save the compressed features.")
    parser.add_argument('--gpu-cache-full-feat',
                        action='store_true',
                        default=False,
                        help="Every gpu cache the full features.")
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

    feature_dim = graph.ndata["features"].shape[1]
    if args.compress_feat:
        print("Compressing features")
        compresser = Compresser(args.compress_mode, args.compress_length,
                                args.compress_width)
        graph.ndata["features"] = compresser.compress(
            graph.ndata.pop("features"),
            cache_root=args.compress_feat_save_root,
            dataset_name=args.dataset)
    else:
        compresser = None

    print(
        'Dataset {} | GPU num {} | Model {} | Fan out {} | Batch size {} | Bias sampling {}'
        .format(args.dataset, n_procs, args.model,
                [int(fanout) for fanout in args.fan_out.split(',')],
                args.batch_size, args.bias))

    data = graph, num_classes, feature_dim

    import torch.multiprocessing as mp
    mp.spawn(run, args=(n_procs, data, args, compresser), nprocs=n_procs)
