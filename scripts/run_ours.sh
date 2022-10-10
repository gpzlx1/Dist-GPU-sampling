mpirun -n 1 python example/bench_graphsage_only_sampling_prob.py --dataset reddit >> log/ours_probs.log
mpirun -n 2 python example/bench_graphsage_only_sampling_prob.py --dataset reddit >> log/ours_probs.log
mpirun -n 4 python example/bench_graphsage_only_sampling_prob.py --dataset reddit >> log/ours_probs.log
mpirun -n 1 python example/bench_graphsage_only_sampling_prob.py --dataset ogbn-papers100M >> log/ours_probs.log
mpirun -n 2 python example/bench_graphsage_only_sampling_prob.py --dataset ogbn-papers100M >> log/ours_probs.log
mpirun -n 4 python example/bench_graphsage_only_sampling_prob.py --dataset ogbn-papers100M >> log/ours_probs.log