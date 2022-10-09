import argparse
import time
import torch
from dgl.data import RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset
import torch.distributed as dist
import numpy as np
from dataloader import SeedGenerator
from dgl.transforms.functional import to_block
from mpi4py import MPI
import os


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


def evaluation(type, dataset, batch_size, mode):
    torch.manual_seed(1)
    torch.set_num_threads(1)
    comm = MPI.COMM_WORLD
    torch.cuda.set_device(comm.Get_rank())
    os.environ["RANK"] = str(comm.Get_rank())
    os.environ["WORLD_SIZE"] = str(comm.Get_size())

    local_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12335"

    dist.init_process_group(backend='nccl', init_method="env://")

    g = None
    if (dataset == "reddit"):
        g, _ = load_reddit()
    elif (dataset == "ogbn-products"):
        g, _ = load_ogb(name="ogbn-products")
    elif (dataset == "ogbn-papers100M"):
        g, _ = load_ogb(name="ogbn-papers100M")

    train_nid = g.nodes()[g.ndata["train_mask"]]
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

    avg_sample_time = bench(g, train_nid, batch_size, device)
    print("Mode {} | Dataset {} | Type {} | sampling time {:.3f} ms".format(
        mode, dataset, type, avg_sample_time))


def bench(g, train_nid, batch_size, device):
    time_list = []
    for _ in range(3):
        torch.cuda.synchronize()
        start = time.time()
        for seeds in SeedGenerator(train_nid,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   device=device):
            for fan_out in [25, 15]:
                frontier = g.sample_neighbors(seeds, fan_out, replace=False)
                block = to_block(frontier, seeds)
                seeds = block.srcdata['_ID']

            dist.barrier()

        torch.cuda.synchronize()
        end = time.time()
        time_list.append(end - start)
    return np.mean(time_list[1:]) * 1000


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
    parser.add_argument("--mode",
                        default="cuda",
                        type=str,
                        choices=["cuda", "cpu", "uva"])
    args = parser.parse_args()

    evaluation(args.type, args.dataset, args.batch_size, args.mode)