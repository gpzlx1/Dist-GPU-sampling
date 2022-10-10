import dgl
import torch as th
import numpy as np
import os

def load_reddit():

    # from ogb.nodeproppred import DglNodePropPredDataset

    # print('load', name)
    # data = DglNodePropPredDataset(name="ogbn-products", root="/data/graphData/original_dataset")

    from dgl.data import RedditDataset

    # load reddit data
    data = RedditDataset(self_loop=True)
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    g.ndata.pop('feat')
    g.ndata.pop('label')      
    return g, data.num_classes

def load_pubmed():
    from dgl.data import PubmedGraphDataset

    # load reddit data
    data = PubmedGraphDataset()
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    g.ndata.pop('feat')
    g.ndata.pop('label')      
    return g, data.num_classes

def load_citeseer():
    from dgl.data import CiteseerGraphDataset

    # load reddit data
    data = CiteseerGraphDataset()
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    g.ndata.pop('feat')
    g.ndata.pop('label')      
    return g, data.num_classes

def load_cora():
    from dgl.data import CoraGraphDataset

    # load reddit data
    data = CoraGraphDataset()
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    g.ndata.pop('feat')
    g.ndata.pop('label')      
    return g, data.num_classes

def load_papers400m(root="dataset"):
    # from ogb.nodeproppred import DglNodePropPredDataset

    if os.path.exists(os.path.join(root, 'papers400M.dgl')):
        print('load papers400M.dgl')
        graph = dgl.load_graphs(os.path.join(root, 'papers400M.dgl'))[0][0]
        print(graph)
        print('finish loading papers400M.dgl')
        original_features = th.from_numpy(np.load(os.path.join(root, 'papers400M_features.npy'))).to(th.float16)
        graph.ndata['features'] = th.cat([original_features, original_features, original_features, original_features], dim=0)
        del original_features
        return graph, 172
    else:
        from ogb import nodeproppred
        print('load papers100m.')
        data = nodeproppred.DglNodePropPredDataset(name="ogbn-papers100M", root=root)
        print('finish loading papers100m.')
        splitted_idx = data.get_idx_split()
        original_graph, labels = data[0]
        original_labels = labels[:, 0]
        original_features = original_graph.ndata.pop('feat').to(th.float16)

        original_src, original_dst = original_graph.edges()
        # original_src = original_src.to(th.int32)
        # original_dst = original_dst.to(th.int32)
        # original_src, original_dst = th.cat([original_src, original_dst], dim=0), th.cat([original_dst, original_src], dim=0)

        n_nodes = original_graph.number_of_nodes()
        del original_graph
        # repeat original_graph for 4 times
        intra_src = th.cat([th.arange(n_nodes, dtype=th.int64).repeat(3).flatten(), th.arange(n_nodes, 2*n_nodes, dtype=th.int64).repeat(3).flatten(), 
                            th.arange(2*n_nodes, 3*n_nodes, dtype=th.int64).repeat(3).flatten(), th.arange(3*n_nodes, 4*n_nodes, dtype=th.int64).repeat(3).flatten()])
        intra_dst = th.cat([th.arange(n_nodes, 4*n_nodes, dtype=th.int64), th.cat([th.arange(0*n_nodes, 1*n_nodes, dtype=th.int64), th.arange(2*n_nodes, 4*n_nodes, dtype=th.int64)]), 
                            th.cat([th.arange(0*n_nodes, 2*n_nodes, dtype=th.int64), th.arange(3*n_nodes, 4*n_nodes, dtype=th.int64)]), th.arange(3*n_nodes, dtype=th.int64)])
        print(intra_src.shape)
        src = th.cat([original_src, original_src + n_nodes, original_src + 2*n_nodes, original_src + 3*n_nodes, intra_src])
        dst = th.cat([original_dst, original_dst + n_nodes, original_dst + 2*n_nodes, original_dst + 3*n_nodes, intra_dst])
        print(src.shape, dst.shape)
        graph = dgl.graph((src, dst))
        del src, dst, intra_src, intra_dst
        print("Generating formats")
        graph = graph.formats("csc")
        # graph = graph.int()
        # np.save(os.path.join(root, 'papers400M_features.npy'), original_features)
        # graph.ndata['features'] = th.cat([original_features, original_features, original_features, original_features], dim=0).to(th.float16)
        graph.ndata['labels'] = th.cat([original_labels, original_labels, original_labels, original_labels], dim=0)
        in_feats = original_features.shape[1]

        del original_features, original_labels

        num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

        # Find the node IDs in the training, validation, and test set.
        train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
        train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
        train_mask[train_nid] = True
        train_mask[train_nid + n_nodes] = True
        train_mask[train_nid + 2*n_nodes] = True
        train_mask[train_nid + 3*n_nodes] = True
        val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
        val_mask[val_nid] = True
        val_mask[val_nid + n_nodes] = True
        val_mask[val_nid + 2*n_nodes] = True
        val_mask[val_nid + 3*n_nodes] = True
        test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
        test_mask[test_nid] = True
        test_mask[test_nid + n_nodes] = True
        test_mask[test_nid + 2*n_nodes] = True
        test_mask[test_nid + 3*n_nodes] = True
        graph.ndata['train_mask'] = train_mask
        graph.ndata['val_mask'] = val_mask
        graph.ndata['test_mask'] = test_mask

        dgl.save_graphs(os.path.join(root, 'papers400M.dgl'), [graph])
        print('finish constructing papers400m', graph)

    return graph, num_labels


def load_ogb(name, root="dataset"):
    from ogb.nodeproppred import DglNodePropPredDataset

    print('load', name)
    data = DglNodePropPredDataset(name=name, root=root)
    print('finish loading', name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    graph.ndata['features'] = graph.ndata.pop('feat')
    graph.ndata['labels'] = labels
    in_feats = graph.ndata['features'].shape[1]
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    print('finish constructing', name)
    return graph, num_labels

def inductive_split(g):
    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata['train_mask'])
    val_g = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    test_g = g
    return train_g, val_g, test_g
