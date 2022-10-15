import argparse
from dgl.data import RedditDataset, AsNodePredDataset
import dgl.nn as dglnn
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import storage
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import tqdm


class SAGE(nn.Module):

    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'mean'))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata['feat']
        sampler = MultiLayerFullNeighborSampler(1,
                                                prefetch_node_feats=['feat'])
        dataloader = DataLoader(g,
                                torch.arange(g.num_nodes()).to(g.device),
                                sampler,
                                device=device,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device,
                pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0]:output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


def process_dataset(dataset, type):
    local_rank = dist.get_rank()
    comm_size = dist.get_world_size()

    g = dataset[0]

    feat = g.ndata.pop("feat")
    label = g.ndata.pop("label").cuda()

    train_idx = dataset.train_idx
    train_idx_num = train_idx.numel()
    each_gpu_seeds_num = int(train_idx_num / comm_size)
    train_idx = train_idx[local_rank * each_gpu_seeds_num:(local_rank + 1) *
                          each_gpu_seeds_num]

    if type == "int":
        g = g.formats(['csc']).int()
        train_idx = train_idx.int()
    else:
        g = g.formats(['csc']).long()
        train_idx = train_idx.long()
    g.ndata.clear()
    g.edata.clear()
    prob = torch.ones(g.num_edges()).float()
    g.edata['prob'] = prob

    return g, label, feat, train_idx, dataset.num_classes


def evaluation(g, label, feat, train_idx, batch_size, fan_out, model, mode):
    local_rank = dist.get_rank()
    train_device = torch.device(local_rank)

    if mode == 'cuda':
        g = g.to("cuda")
        train_idx = train_idx.to("cuda")
        use_uva = False
    elif mode == 'cpu':
        g = g.to("cpu")
        train_idx = train_idx.to("cpu")
        use_uva = False
    elif mode == 'uva':
        g = g.to("cpu")
        train_idx = train_idx.to("cuda")
        use_uva = True

    model = model.to(train_device)
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[local_rank],
                                                output_device=local_rank)

    # create sampler and dataloader
    sampler = NeighborSampler(fan_out, prob="prob")
    train_dataloader = DataLoader(g,
                                  train_idx,
                                  sampler,
                                  device=train_device,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=0,
                                  use_uva=use_uva)

    # pagraph cache
    cacher = storage.GraphCacheServer(feat, g.num_nodes(), gpuid=local_rank)
    cacher.auto_cache(g, None, 1, train_idx)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    sampling_time_list = []
    epoch_time_list = []
    for epoch in range(3):

        model.train()
        total_loss = 0
        sampling_time = 0

        torch.cuda.synchronize()
        epoch_start = time.time()
        sampling_start = time.time()

        for it, (input_nodes, output_nodes,
                 blocks) in enumerate(train_dataloader):
            sampling_time += time.time() - sampling_start

            x = cacher.fetch_data(input_nodes)
            y = label[output_nodes]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

            torch.cuda.synchronize()
            sampling_start = time.time()

        torch.cuda.synchronize()
        epoch_end = time.time()

        sampling_time_list.append(sampling_time)
        epoch_time_list.append(epoch_end - epoch_start)

    avg_sampling_time = np.mean(sampling_time_list[1:]) * 1000
    avg_epoch_time = np.mean(epoch_time_list[1:]) * 1000

    return avg_sampling_time, avg_epoch_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--type",
                        default="long",
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
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    if (args.dataset == "reddit"):
        dataset = AsNodePredDataset(RedditDataset(self_loop=True))
    elif (args.dataset == "ogbn-products"):
        dataset = AsNodePredDataset(
            DglNodePropPredDataset("ogbn-products", root="/data/nfs/"))
    elif (args.dataset == "ogbn-papers100M"):
        dataset = AsNodePredDataset(
            DglNodePropPredDataset("ogbn-papers100M", root="/data/nfs/"))

    g, label, feat, train_idx, num_classes = process_dataset(
        dataset, args.type)
    hidden_dim = 256
    model = SAGE(feat.shape[1], hidden_dim, num_classes)
    fanout = [5, 5, 5]
    mode_set = ["cpu", "uva", "cuda"]
    if dist.get_rank() == 0:
        print(
            "Model GraphSAGE | Hidden dim {} | Batch size {} | Fanout {} | World Size {} | Dataset {} | Type {}"
            .format(hidden_dim, args.batch_size, fanout, dist.get_world_size(),
                    args.dataset, args.type))
    for mode in mode_set:
        sampling_time, epoch_time = evaluation(g, label, feat, train_idx,
                                               args.batch_size, fanout, model,
                                               mode)
        if dist.get_rank() == 0:
            print("Mode {} | Sampling time {:.3f} ms | Epoch time {:.3f} ms".
                  format(mode, sampling_time, epoch_time))
