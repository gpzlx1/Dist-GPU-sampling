import os
import argparse
import time
import torch
import torch.distributed as dist
from dgl.data import RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset
import numpy as np
import dgl
from dataloader import SeedGenerator


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


def evaluation(type, dataset, batch_size):
    torch.manual_seed(1)
    torch.ops.load_library("./build/libdgs.so")
    torch.ops.dgs_ops._CAPI_initialize()
    torch.set_num_threads(1)
    torch.cuda.set_device(torch.ops.dgs_ops._CAPI_get_rank())
    os.environ["RANK"] = str(torch.ops.dgs_ops._CAPI_get_rank())
    os.environ["WORLD_SIZE"] = str(torch.ops.dgs_ops._CAPI_get_size())

    local_rank = torch.ops.dgs_ops._CAPI_get_rank()
    comm_size = torch.ops.dgs_ops._CAPI_get_size()

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12335"

    dist.init_process_group(backend='nccl', init_method="env://")

    if (dataset == "reddit"):
        g, _ = load_reddit()
    elif (dataset == "ogbn-products"):
        g, _ = load_ogb(name="ogbn-products")
    elif (dataset == "ogbn-papers100M"):
        g, _ = load_ogbn_papers100m(root="/mnt/c/data")

    train_nid = g.nodes()[g.ndata["train_mask"].bool()]
    train_nid_num = train_nid.numel()
    each_gpu_seeds_num = int(train_nid_num / comm_size)
    if type == "int":
        indptr = g.adj_sparse("csc")[0].int()
        indices = g.adj_sparse("csc")[1].int()
        train_nid = train_nid.int()
        train_nid = train_nid[local_rank *
                              each_gpu_seeds_num:(local_rank + 1) *
                              each_gpu_seeds_num]
        type_size_in_bytes = 4
    else:
        indptr = g.adj_sparse("csc")[0].long()
        indices = g.adj_sparse("csc")[1].long()
        train_nid = train_nid.long()
        train_nid = train_nid[local_rank *
                              each_gpu_seeds_num:(local_rank + 1) *
                              each_gpu_seeds_num]
        type_size_in_bytes = 8

    indptr_cache_set = [0, 0.5, 1]
    indices_cache_set = [0, 0.5, 1]
    for indptr_cache in indptr_cache_set:
        for indices_cache in indices_cache_set:
            avg_sample_time = bench(indptr, indices, train_nid, batch_size,
                                    type_size_in_bytes, indptr_cache,
                                    indices_cache)
            print(
                "Device {} | Dataset {} | Type {} | indptr cache size {:.1f} | indices cache size {:.1f} | sampling time {:.3f} ms"
                .format(torch.ops.dgs_ops._CAPI_get_rank(), dataset, type,
                        indptr_cache, indices_cache, avg_sample_time))

    torch.ops.dgs_ops._CAPI_finalize()


def bench(indptr, indices, train_nid, batch_size, type_size, indptr_cache,
          indices_cache):

    chunk_indptr = torch.classes.dgs_classes.ChunkTensor(
        indptr,
        int((indptr.numel() * type_size * indptr_cache) /
            torch.ops.dgs_ops._CAPI_get_size()) + type_size)
    chunk_indices = torch.classes.dgs_classes.ChunkTensor(
        indices,
        int((indices.numel() * type_size * indices_cache) /
            torch.ops.dgs_ops._CAPI_get_size()) + type_size)

    time_list = []
    for _ in range(3):

        torch.cuda.synchronize()
        start = time.time()
        for seeds in SeedGenerator(train_nid,
                                   batch_size=batch_size,
                                   shuffle=True):
            for fan_out in [25, 15]:
                coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_with_chunk_tensor(
                    seeds, chunk_indptr, chunk_indices, fan_out, False)
                frontier, (coo_row,
                           coo_col) = torch.ops.dgs_ops._CAPI_tensor_relabel(
                               [seeds, coo_col], [coo_row, coo_col])
                seeds = frontier

            # simluation for sync grads
            dist.barrier()

        torch.cuda.synchronize()
        end = time.time()
        time_list.append(end - start)

    del chunk_indptr
    del chunk_indices
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
    args = parser.parse_args()

    evaluation(args.type, args.dataset, args.batch_size)