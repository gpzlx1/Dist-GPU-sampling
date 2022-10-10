import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.optim
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
# from dgl.utils import pin_memory_inplace, unpin_memory_inplace
from dgl.multiprocessing import shared_tensor
import time
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
import os

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        # self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(0.5)
        self.n_hidden = n_hidden
        self.n_classes = n_classes

    def _forward_layer(self, l, block, x):
        h = self.layers[l](block, x)
        if l != len(self.layers) - 1:
            h = F.relu(h)
            h = self.dropout(h)
        return h

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = self._forward_layer(l, blocks[l], h)
        return h

    def inference(self, g, device, batch_size):
        """
        Perform inference in layer-major order rather than batch-major order.
        That is, infer the first layer for the entire graph, and store the
        intermediate values h_0, before infering the second layer to generate
        h_1. This is done for two reasons: 1) it limits the effect of node
        degree on the amount of memory used as it only proccesses 1-hop
        neighbors at a time, and 2) it reduces the total amount of computation
        required as each node is only processed once per layer.

        Parameters
        ----------
            g : DGLGraph
                The graph to perform inference on.
            device : context
                The device this process should use for inference
            batch_size : int
                The number of items to collect in a batch.

        Returns
        -------
            tensor
                The predictions for all nodes in the graph.
        """
        g.ndata['h'] = g.ndata['features']
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1, prefetch_node_feats=['h'])
        dataloader = dgl.dataloading.DataLoader(
                g, torch.arange(g.num_nodes(), device=device, dtype=torch.int64), sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False,
                num_workers=0, use_ddp=True, use_uva=True)

        for l, layer in enumerate(self.layers):
            # in order to prevent running out of GPU memory, we allocate a
            # shared output tensor 'y' in host memory, pin it to allow UVA
            # access from each GPU during forward propagation.
            y = shared_tensor(
                    (g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes))
            # pin_memory_inplace(y)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader) \
                    if dist.get_rank() == 0 else dataloader:
                x = blocks[0].srcdata['h'].to(torch.float32)
                h = self._forward_layer(l, blocks[0], x)
                y[output_nodes] = h.to(y.device)
            # make sure all GPUs are done writing to 'y'
            dist.barrier()
            # if l > 0:
            #     unpin_memory_inplace(g.ndata['h'])
            if l + 1 < len(self.layers):
                # assign the output features of this layer as the new input
                # features for the next layer
                g.ndata['h'] = y
            else:
                # remove the intermediate data from the graph
                g.ndata.pop('h')
        return y


def train(rank, world_size, graph, num_classes):
    torch.cuda.set_device(rank)
    print('rank:', rank)
    dist.init_process_group('nccl', 'tcp://127.0.0.1:12348', world_size=world_size, rank=rank)
    print('init process group')
    model = SAGE(graph.ndata['features'].shape[1], 64, num_classes).cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    train_mask = graph.ndata.pop('train_mask')
    val_mask = graph.ndata.pop('val_mask')
    test_mask = graph.ndata.pop('test_mask')
    train_idx = train_mask.nonzero().squeeze().to(torch.int64)
    valid_idx = val_mask.nonzero().squeeze().to(torch.int64)
    test_idx = test_mask.nonzero().squeeze().to(torch.int64)
    # move ids to GPU
    # train_idx = train_idx.to('cuda')
    # valid_idx = valid_idx.to('cuda')
    # test_idx = test_idx.to('cuda')
    print('train_idx:', train_idx.shape)
    # For training, each process/GPU will get a subset of the
    # train_idx/valid_idx, and generate mini-batches indepednetly. This allows
    # the only communication neccessary in training to be the all-reduce for
    # the gradients performed by the DDP wrapper (created above).
    sampler = dgl.dataloading.NeighborSampler(
            [10,25], prob="weights", prefetch_node_feats=['features'], prefetch_labels=['labels'])
    train_dataloader = dgl.dataloading.DataLoader(
            graph, train_idx, sampler,
            device='cuda', batch_size=1000, shuffle=True, drop_last=False,
            num_workers=0, use_ddp=True, use_uva=False)
    valid_dataloader = dgl.dataloading.DataLoader(
            graph, valid_idx, sampler, device='cuda', batch_size=1000, shuffle=True,
            drop_last=False, num_workers=0, use_ddp=True,
            use_uva=False)
    print("start training")
    durations = []
    time_list = []
    for epoch in range(6):
        model.train()
        t0 = time.time()
        f0 = time.time()
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            f1 = time.time()
            if it==0 and epoch==0:
                print(blocks)
            x = blocks[0].srcdata['features'].to(torch.float32).cuda()
            y = blocks[-1].dstdata['labels'].to(torch.long).cuda()
            f2 = time.time()
            y_hat = model(blocks, x)
            f3 = time.time()
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            f4 = time.time()
            if it % 20 == 0 and rank == 0:
                acc = MF.accuracy(y_hat, y)
                mem = torch.cuda.max_memory_allocated() / 1000000
                print('Epoch {:05d} | Iter {:05d} | Loss {:.4f} | Accuracy {:.4f} | Mem {:.2f} MB'.format(
                    epoch, it, loss.item(), acc.item(), mem))
                # print('Loss', loss.item(), 'Acc', acc.item(), 'GPU Mem', mem, 'MB')
            time_list.append([f1-f0, f2-f1, f3-f2, f4-f3])
            f0 = time.time()
        tt = time.time()

        if rank == 0:
            print('Epoch {} | Time {}'.format(epoch, tt - t0))
            # print(tt - t0)
        durations.append(tt - t0)

        model.eval()
        ys = []
        y_hats = []
        for it, (input_nodes, output_nodes, blocks) in enumerate(valid_dataloader):
            with torch.no_grad():
                x = blocks[0].srcdata['features'].to(torch.float32)
                ys.append(blocks[-1].dstdata['labels'].long())
                y_hats.append(model.module(blocks, x))
        acc = MF.accuracy(torch.cat(y_hats), torch.cat(ys)) / world_size
        dist.reduce(acc, 0)
        if rank == 0:
            print('Validation acc:', acc.item())
            print()
        dist.barrier()


    if rank == 0:
        print(np.mean(durations[2:]), np.std(durations[2:]))

        if not os.path.exists('results'):
            os.makedirs('results')
        with open("results/time_log.txt", "a+") as f:
            for i in np.mean(time_list[5:], axis=0):
                print("{:.5f}".format(i), sep="\t", end="\t", file=f)
            print(np.mean(durations[2:]), "single_machine", "node_classification", "GraphSAGE", sep="\t", file=f)           
    # model.eval()
    # with torch.no_grad():
    #     # since we do 1-layer at a time, use a very large batch size
    #     pred = model.module.inference(graph, device='cuda', batch_size=2**16)
    #     if rank == 0:
    #         acc = MF.accuracy(pred[test_idx], graph.ndata['labels'][test_idx].long())
    #         print('Test acc:', acc.item())

if __name__ == '__main__':
    from load_graph import *    
    graph, num_classes = load_reddit()
    # graph, num_classes = load_ogb('ogbn-products', root="/data/graphData/original_dataset")
    # graph, num_classes = load_ogb('ogbn-papers100M', root="/data/graphData/original_dataset")
    # graph, num_classes = load_papers400m(root="/data/giant_graph/original_dataset")


    graph = graph.formats("csc")
    # graph.create_formats_()
    graph.ndata['labels'] = graph.ndata['labels'].long()
    graph.edata['weights'] = torch.rand(graph.number_of_edges())
    # graph.ndata['features'] = graph.ndata['features'].to(torch.float16)
    print(graph, num_classes)
    # use all available GPUs
    n_procs = 1
    # n_procs = torch.cuda.device_count()


    print("Spawning processes")
    import torch.multiprocessing as mp
    mp.spawn(train, args=(n_procs, graph, num_classes), nprocs=n_procs)



