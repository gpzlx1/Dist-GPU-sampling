import argparse
import time
import torch
from dgl.data import RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset
import torch.distributed as dist
import numpy as np
from dataloader import SeedGenerator
from dgl.transforms.functional import to_block
import dgl
import os
from model import SAGE
import torch.nn.functional as F


def load_reddit(self_loop=True):
    # load reddit data
    data = RedditDataset(self_loop=self_loop)
    g = data[0]
    g.ndata["features"] = g.ndata.pop("feat")
    g.ndata["labels"] = g.ndata.pop("label")
    return g, data.num_classes


def load_ogb(name, root="dataset"):
    print("load", name)
    data = DglNodePropPredDataset(name=name, root=root)
    print("finish loading", name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    graph.ndata["features"] = graph.ndata.pop("feat")
    graph.ndata["labels"] = labels
    num_labels = len(
        torch.unique(labels[torch.logical_not(torch.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    train_mask = torch.zeros((graph.number_of_nodes(), ), dtype=torch.bool)
    train_mask[train_nid] = True
    val_mask = torch.zeros((graph.number_of_nodes(), ), dtype=torch.bool)
    val_mask[val_nid] = True
    test_mask = torch.zeros((graph.number_of_nodes(), ), dtype=torch.bool)
    test_mask[test_nid] = True
    graph.ndata["train_mask"] = train_mask
    graph.ndata["val_mask"] = val_mask
    graph.ndata["test_mask"] = test_mask
    print("finish constructing", name)
    return graph, num_labels


def load_ogbn_papers100m(root="dataset"):
    g = dgl.load_graphs(root + "/fast-papers100M")
    return g[0][0], None


def evaluation(type, dataset, batch_size, mode):
    local_rank = dist.get_rank()
    comm_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)

    g = None
    if (dataset == "reddit"):
        g, num_labels = load_reddit()
    elif (dataset == "ogbn-products"):
        g, num_labels = load_ogb(name="ogbn-products")
    elif (dataset == "ogbn-papers100M"):
        g, num_labels = load_ogbn_papers100m(root="/mnt/c/data")

    train_nid = g.nodes()[g.ndata["train_mask"].bool()]
    train_nid_num = train_nid.numel()
    each_gpu_seeds_num = int(train_nid_num / comm_size)
    if type == "int":
        g = g.formats(['csc']).int()
        train_nid = train_nid.int()
        train_nid = train_nid[local_rank *
                              each_gpu_seeds_num:(local_rank + 1) *
                              each_gpu_seeds_num]
    else:
        g = g.formats(['csc']).long()
        train_nid = train_nid.long()
        train_nid = train_nid[local_rank *
                              each_gpu_seeds_num:(local_rank + 1) *
                              each_gpu_seeds_num]

    print(g)
    feat = g.ndata['features'].to('cuda')
    in_size = feat.shape[1]
    label = g.ndata['labels'].to('cuda')

    g.ndata.clear()
    g.edata.clear()
    if mode == 'cuda':
        device = "cuda"
        g = g.to(device)
    elif mode == 'cpu':
        device = "cpu"
        g = g.to(device)
    elif mode == 'uva':
        device = "cuda"
        g.pin_memory_()

    model = SAGE(in_size, 256, num_labels).to('cuda')
    model = torch.nn.parallel.DistributedDataParallel(
        model, find_unused_parameters=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    avg_sample_time = bench(model, feat, label, opt, g, train_nid, batch_size,
                            device)

    #if local_rank == 0:
    #    print(
    #        "World Size {} | Mode {} | Dataset {} | Type {} | sampling time {:.3f} ms"
    #        .format(comm_size, mode, dataset, type, avg_sample_time))


def bench(model, feat, label, opt, g, train_nid, batch_size, device):
    for _ in range(3):
        for seeds in SeedGenerator(train_nid,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   device=device):
            output_node = seeds
            blocks = []
            for fan_out in [25, 15]:
                frontier = g.sample_neighbors(seeds, fan_out, replace=False)
                block = to_block(frontier, seeds)
                seeds = block.srcdata['_ID']
                blocks.insert(0, block)
            input_node = seeds

            sub_feature = feat[input_node.long()]
            sub_label = label[output_node.long()]
            y_hat = model(blocks, sub_feature)
            loss = F.cross_entropy(y_hat, sub_label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(loss)

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--type",
                        default="int",
                        choices=["int", "long"],
                        help="Type for tensor, choose from 'int' and 'long'.")
    parser.add_argument("--dataset",
                        default="reddit",
                        choices=["reddit", "ogbn-products", "ogbn-papers100M"],
                        help="The dataset to be sampled.")
    parser.add_argument("--batch-size",
                        default="5000",
                        type=int,
                        help="The number of seeds of sampling.")
    args = parser.parse_args()

    torch.manual_seed(1)
    torch.set_num_threads(1)
    dist.init_process_group(backend='nccl', init_method="env://")

    #evaluation(args.type, args.dataset, args.batch_size, 'cpu')
    #evaluation(args.type, args.dataset, args.batch_size, 'uva')
    evaluation(args.type, args.dataset, args.batch_size, 'cuda')