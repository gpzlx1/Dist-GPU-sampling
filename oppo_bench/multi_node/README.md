# Distributed training

## Dependencies

- CUDA 11.3

- PyTorch 1.12.1

- DGL 0.9.1

- NumPy 1.23.5

- OGB 1.3.5

## Dataset

- ogbn-products

- ogbn-papers100M

- ogbn-papers400M

## How to run

### Step 0. Set up a workspace

To perform distributed training, data files and codes need to be accessed across multiple machines.

You can setup a distributed file system to handle this task, or you can just copy the data and codes to all the machines and locate them in the same directory.

### Step 1. Partition the graph

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

### Step 2. Set IP configuration file

Edit `ip_config,txt`, add the IPs of all the machines that will participate in the training. For example:

```text
192.168.1.51
192.168.1.52
```

**Users need to make sure that the master node can ssh to all the other nodes without password authentication.**

### Step 3. Launch the distributed training

#### Baseline

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

#### Chunktensor

```bash
python3 ~/workspace/dgl/tools/launch.py \
--workspace ~/workspace/dgs-bench/oppo_bench/multi_node/ \
--num_trainers 2 \
--num_samplers 0 \
--num_servers 1 \
--part_config data/ogbn-products.json \
--ip_config ip_config.txt \
"python3 chunktensor.py --graph_name ogbn-products --root dataset/ --ip_config ip_config.txt --batch_size 1000 --num_gpu 2 --model graphsage --bias --libdgs ../../../Dist-GPU-sampling/build/libdgs.so --graph-cache-rate 1 --feat-cache-rate 1"
```

args:

- `--root` The directory which stores the dataset. Add this arg because the whole graph need to be loaded one more time on each machine to create the chunktensors.

- `--libdgs` Same with single node version.

- `--graph-cache-rate` Same with single node version. Note: this means the total cache rate of all the gpus on **each machine**.

- `--feat-cache-rate` Same with single node version. Note: this means the total cache rate of all the gpus on **each machine**.