import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
import argparse
from sampler import gpu_sampler
import time
import numpy as np
import torch.distributed as dist
import os
import storage

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
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False,
                num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(), self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device, pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x) # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0]:output_nodes[-1]+1] = h.to(buffer_device)
            feat = y
        return y

def evaluate(model, graph, dataloader):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata['feat']
            ys.append(blocks[-1].dstdata['label'])
            y_hats.append(model(blocks, x))
    return MF.accuracy(torch.cat(y_hats), torch.cat(ys))


def train(args, dataset):
    torch.manual_seed(1)
    torch.set_num_threads(1)
    torch.ops.load_library("./build/libdgs.so")
    torch.ops.dgs_ops._CAPI_initialize()    
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12345"     
    os.environ["RANK"] = str(torch.ops.dgs_ops._CAPI_get_rank())
    os.environ["WORLD_SIZE"] = str(torch.ops.dgs_ops._CAPI_get_size())        
    dist.init_process_group(backend='nccl', init_method="env://")


    torch.set_num_threads(1)         
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print('rank:', rank)
    torch.cuda.set_device(rank)   
    device = torch.device(rank)

    g = dataset[0]
    # create GraphSAGE model
    in_size = g.ndata['feat'].shape[1]
    out_size = dataset.num_classes
    feat = g.ndata.pop("feat")
    label = g.ndata.pop("label").cuda()    
    cacher = storage.GraphCacheServer(feat, g.num_nodes(), None, rank)

    model = SAGE(in_size, 256, out_size).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)



    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    sampler = gpu_sampler([5, 10, 15], g, cache_percent_indptr=args.cpindptr, cache_percent_indices=args.cpindices, rank=rank, world_size=world_size)
    use_uva = True
    train_dataloader = DataLoader(g, train_idx, sampler, device=device,
                                  batch_size=1024, shuffle=True,
                                  drop_last=False, num_workers=0,
                                  use_uva=use_uva, use_ddp=True)

    val_dataloader = DataLoader(g, val_idx, sampler, device=device,
                                batch_size=1024, shuffle=True,
                                drop_last=False, num_workers=0,
                                use_uva=use_uva, use_ddp=True)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    
    time_log = []
    for epoch in range(10):
        model.train()
        total_loss = 0
        
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = cacher.fetch_data(input_nodes)
            y = label[output_nodes]
            if it==0:
                t0 = time.time()            
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            if epoch == 0 and it == 1:
                cacher.auto_cache(g, None, args.cpfeat, train_idx)             
            # acc = MF.accuracy(y_hat, y)

        time_log.append(time.time()-t0)
        # acc = evaluate(model, g, val_dataloader)
        if rank==0:
            print("Epoch {:05d} | Loss {:.4f} | Time {:.2f}"
                .format(epoch, total_loss / (it+1), time_log[-1]))
        
    if rank==0:
        with open("/home/yxma/graphstore/results/time_log.txt", "a+") as f:
            print(np.mean(time_log[5:]), "single_gpu_compact", "GraphSAGE", args.dataset, world_size, args.cpindptr, args.cpindices, sep="\t", file=f)     
    del sampler.chunk_indptr
    del sampler.chunk_indices
    torch.ops.dgs_ops._CAPI_finalize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='mixed', choices=['cpu', 'mixed', 'puregpu'],
                        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                             "'puregpu' for pure-GPU training.")
    parser.add_argument("--dataset", default='reddit')                             
    parser.add_argument("--cpindices", default=0, type=float, help="cache percentage of indices")                             
    parser.add_argument("--cpindptr", default=0, type=float, help="cache percentage of indptr")     
    parser.add_argument("--cpfeat", default=0, type=float, help="cache percentage of features")                              

    args = parser.parse_args()
    print('Loading data')


    if args.dataset=="reddit":
        dataset = AsNodePredDataset(dgl.data.RedditDataset(self_loop=True))
    elif args.dataset=="ogbn-products":
        dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products', root="/data/graph/ogbn-products"))
    else:
        print("wrong dataset")
        exit()
    # g = dataset[0]

    train(args, dataset)

