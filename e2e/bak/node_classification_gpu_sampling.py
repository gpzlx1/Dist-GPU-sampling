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

# def layerwise_infer(device, graph, nid, model, batch_size):
#     model.eval()
#     with torch.no_grad():
#         pred = model.inference(graph, device, batch_size) # pred in buffer_device
#         pred = pred[nid]
#         label = graph.ndata['label'][nid].to(pred.device)
#         return MF.accuracy(pred, label)

def train(args, device, g, dataset, model):
    # create sampler & dataloader


    feat = g.ndata.pop("feat").cuda()
    label = g.ndata.pop("label").cuda()

    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    sampler = gpu_sampler([5, 10, 15], g, cache_percent_indptr=args.cpindptr, cache_percent_indices=args.cpindices)
    use_uva = True
    train_dataloader = DataLoader(g, train_idx, sampler, device=device,
                                  batch_size=1024, shuffle=True,
                                  drop_last=False, num_workers=0,
                                  use_uva=use_uva)

    val_dataloader = DataLoader(g, val_idx, sampler, device=device,
                                batch_size=1024, shuffle=True,
                                drop_last=False, num_workers=0,
                                use_uva=use_uva)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    
    time_log = []
    for epoch in range(30):
        model.train()
        total_loss = 0
        
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            if it==0:
                t0 = time.time()
            # print(blocks, input_nodes.shape, output_nodes.shape)
            x = feat[input_nodes]
            y = label[output_nodes]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            # acc = MF.accuracy(y_hat, y)

        time_log.append(time.time()-t0)
        # acc = evaluate(model, g, val_dataloader)
        print("Epoch {:05d} | Loss {:.4f} | Time {:.2f}"
              .format(epoch, total_loss / (it+1), time_log[-1]))
        

    with open("/home/yxma/graphstore/results/time_log.txt", "a+") as f:
        print(np.mean(time_log[5:]), "single_gpu_compact", "GraphSAGE", args.dataset, args.cpindptr, args.cpindices, sep="\t", file=f)     

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='mixed', choices=['cpu', 'mixed', 'puregpu'],
                        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                             "'puregpu' for pure-GPU training.")
    parser.add_argument("--dataset", default='reddit')                             
    parser.add_argument("--cpindices", default=0, type=float)                             
    parser.add_argument("--cpindptr", default=0, type=float)                             
    args = parser.parse_args()
    # if not torch.cuda.is_available():
    #     args.mode = 'cpu'
    # print(f'Training in {args.mode} mode.')

    # load and preprocess dataset
    print('Loading data')


    if args.dataset=="reddit":
        dataset = AsNodePredDataset(dgl.data.RedditDataset(self_loop=True))
    elif args.dataset=="ogbn-products":
        dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products', root="/data/graph/ogbn-products"))
    else:
        print("wrong dataset")
        exit()
    g = dataset[0]
    # g = g.to('cuda' if args.mode == 'puregpu' else 'cpu')
    device = torch.device('cuda')

    # create GraphSAGE model
    in_size = g.ndata['feat'].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, 256, out_size).to(device)

    # model training
    print('Training...')
    train(args, device, g, dataset, model)

    # # test the model
    # print('Testing...')
    # acc = layerwise_infer(device, g, dataset.test_idx, model, batch_size=4096)
    # print("Test Accuracy {:.4f}".format(acc.item()))
