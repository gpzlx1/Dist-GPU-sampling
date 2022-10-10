torchrun --nproc_per_node 1 example/bench_graphsage_only_sampling_prob_dgl.py --dataset reddit >> log/dgl_probs.log
torchrun --nproc_per_node 2 example/bench_graphsage_only_sampling_prob_dgl.py --dataset reddit >> log/dgl_probs.log
torchrun --nproc_per_node 4 example/bench_graphsage_only_sampling_prob_dgl.py --dataset reddit >> log/dgl_probs.log
torchrun --nproc_per_node 1 example/bench_graphsage_only_sampling_prob_dgl.py --dataset ogbn-papers100M >> log/dgl_probs.log
torchrun --nproc_per_node 2 example/bench_graphsage_only_sampling_prob_dgl.py --dataset ogbn-papers100M >> log/dgl_probs.log
torchrun --nproc_per_node 4 example/bench_graphsage_only_sampling_prob_dgl.py --dataset ogbn-papers100M >> log/dgl_probs.log