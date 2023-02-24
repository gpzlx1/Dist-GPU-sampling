# Distributed training
## Dataset

- ogbn-products

- ogbn-papers100M

- ogbn-papers400M

Notes: To generate `obgn-papers400M`, you should prepare `ogbn-papers100M`.

## How to run

### 1. Baseline

#### Step 0. Set up a workspace

To perform distributed training, data files and codes need to be accessed across multiple machines.

You can setup a distributed file system to handle this task, or you can just copy the data and codes to all the machines and locate them in the same directory.

#### Step 1. Partition the graph

```bash
python3 partition_graph.py \
--dataset ogbn-papers100M \
--root dataset/ \
--num-parts 2 \
--balance-train \
--balance-edges \
--num-trainers-per-machine 2 \
--bias
```

args:

- `--dataset` The graph to be partitioned, support `ogbn-products`, `ogbn-papers100M` and `ogbn-papers400M`.

- `--root` The directory which stores the dataset.

- `--num-parts` The number of partitions.

- `--balance-train` Balance the training size in each partition.

- `--balance-edges` Balance the number of edges in each partition.

- `--num-trainers-per-machine` The number of trainers per machine. The trainer ids are stored in the node feature 'trainer_id'

- `--bias` For sampling with bias, generate probs tensor as edata before partition.

#### Step 2. Set IP configuration file

Edit `ip_config,txt`, add the IPs of all the machines that will participate in the training. For example:

```text
192.168.1.51
192.168.1.52
```

**Users need to make sure that the master node can ssh to all the other nodes without password authentication.**

#### Step 3. Launch the distributed training

```bash
python3 ~/workspace/dgl/tools/launch.py \
--workspace ~/workspace/dgs-bench/oppo_bench/multi_node/ \
--num_trainers 2 \
--num_samplers 0 \
--num_servers 1 \
--part_config data/ogbn-products.json \
--ip_config ip_config.txt \
"python3 baseline.py --graph_name ogbn-products --ip_config ip_config.txt --batch_size 1000 --num_gpu 2 --model graphsage --bias"
```

args:

- `--part_config` The metadata file of the partitioned graph.

- `--num_trainers` The number of trainer processes per machine, should be the same value with `--num_gpu`

- `--num_samplers` The number of sampler processes per trainer process.

- `--num_servers` The number of server processes per machine.

- `--num_gpu` The number of gpus participated in the training per machine.

### 2. Chunktensor

#### Step 1. Set socket interface name

```shell
export NCCL_SOCKET_IFNAME=${your socket interface name}
```

You can get your socket interface name with command `ifconfig -a`

#### Step 2. Launch the distributed training

Run this command on every node of the cluster:

```shell
OMP_NUM_THREADS=${#CPUs} torchrun --nproc_per_node 2 \
--master_port 12345 \
--nnodes=${number of nodes} \
--node_rank=${rank of this node} \
--master_addr=${IP address of master node} \
chunktensor.py --graph_name ogbn-papers400M --root /data --libdgs ../../build/libdgs.so --num_gpu 8 --bias --feat_cache_rate 1 --graph_cache_rate 1
```

args:

- `--nproc_per_node` the number of trainers on this node, should be equal to `--num_gpu`.

- `--master_port` has to be a free port on master node (node with rank 0).

- `--graph_name` the dataset. Note that each node will load the dataset.

- `--root` the directory which stores the dataset. 

- `--libdgs` the directory of `libdgs.so`.

- `--num_gpu` the number of gpus participated in the training per node.

- `--bias` randomly generate probs as edata and sample with bias.

- `--graph_cache_rate`

  The cache rate of graph structure tensors.

  Note: this is the cache rate of **all the gpus on one node**. For example, if `--num_gpu 2` and `--graph_cache_rate 0.4` are set, the cache rate of each gpu on one node is `0.2`.

- `--feat_cache_rate`

  The cache rate of features.

  Note: If gpu memory is not large enough to achieve the given cache rate, cache priority: features > probs > indices > indptr.
