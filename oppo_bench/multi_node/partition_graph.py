import argparse
import time
import torch as th
import dgl
from utils.load_graph import load_papers400m_sparse, load_ogb

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--root',
                           default='dataset/',
                           help='Path of the dataset.')
    argparser.add_argument(
        "--dataset",
        default="ogbn-papers400M",
        choices=["ogbn-products", "ogbn-papers100M", "ogbn-papers400M"])
    argparser.add_argument("--num-parts",
                           type=int,
                           default=4,
                           help="number of partitions")
    argparser.add_argument("--part-method",
                           type=str,
                           default="metis",
                           help="the partition method")
    argparser.add_argument("--balance-train",
                           action="store_true",
                           help="balance the training size in each partition.")
    argparser.add_argument("--undirected",
                           action="store_true",
                           help="turn the graph into an undirected graph.")
    argparser.add_argument(
        "--balance-edges",
        action="store_true",
        help="balance the number of edges in each partition.")
    argparser.add_argument(
        "--num-trainers-per-machine",
        type=int,
        default=1,
        help="the number of trainers per machine. The trainer ids are stored\
                                in the node feature 'trainer_id'")
    argparser.add_argument("--output",
                           type=str,
                           default="data",
                           help="Output path of partitioned graph.")
    argparser.add_argument(
        "--bias",
        action="store_true",
        help=
        "For sampling with bias, generate probs tensor as edata before partition."
    )
    args = argparser.parse_args()

    start = time.time()
    if args.dataset == "ogbn-products":
        g, _ = load_ogb("ogbn-products", root=args.root)
    elif args.dataset == "ogbn-papers100M":
        g, _ = load_ogb("ogbn-papers100M", root=args.root)
    elif args.dataset == "ogbn-papers400M":
        g, _ = load_papers400m_sparse(root=args.root)

    print("load {} takes {:.3f} seconds".format(args.dataset,
                                                time.time() - start))
    print("|V|={}, |E|={}".format(g.number_of_nodes(), g.number_of_edges()))
    print("train: {}, valid: {}, test: {}".format(
        th.sum(g.ndata["train_mask"]),
        th.sum(g.ndata["val_mask"]),
        th.sum(g.ndata["test_mask"]),
    ))

    if args.bias:
        g.edata["probs"] = th.randn((g.num_edges(), )).float()

    if args.balance_train:
        balance_ntypes = g.ndata["train_mask"]
    else:
        balance_ntypes = None

    if args.undirected:
        sym_g = dgl.to_bidirected(g, readonly=True)
        for key in g.ndata:
            sym_g.ndata[key] = g.ndata[key]
        g = sym_g

    dgl.distributed.partition_graph(
        g,
        args.dataset,
        args.num_parts,
        args.output,
        part_method=args.part_method,
        balance_ntypes=balance_ntypes,
        balance_edges=args.balance_edges,
        num_trainers_per_machine=args.num_trainers_per_machine,
    )
