# Dist-GPU-sampling

## ChunkTensor
```Hybrid-Memory Graph Store (HMGStore)``` supports using memory from different GPUs and CPU as a shared memory system. Based on `HMGStore`, we develop `ChunkTensor` which enable storing graph dataset cross different devices and host to reduce data transmission and improve GNN training performance.

A case for ChunkTensor usage:
```python
c_a = torch.classes.dgs_classes.ChunkTensor([100], torch.int64, capacity_per_gpu=200)
if dist.get_rank(local_subgroup) == 0:
    a = torch.arange(100).long()
    c_a._CAPI_load_from_tensor(a)
```

Note: Currently, this repo is only supported on systems where the GPUs have peer-to-peer access to each other.

## Install 
Requirement:
* CUDA >= 11.3
* NCCL >= 2.x

Install python dependencies.
```shell
$ pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116
$ pip install --pre dgl -f https://data.dgl.ai/wheels/cu116/repo.html
```

Install the system packages for building the shared library.
```shell
$ sudo apt-get update
$ sudo apt-get install -y build-essential python3-dev make cmake
```

Download the source files.
```shell
$ git clone git@github.com:gpzlx1/Dist-GPU-sampling.git
$ git checkout v0.1
```

Build.
```shell
$ mkdir build && cd build
$ cmake ..
$ make -j16
```

After building, it will generate `libdgs.so` in `${WORKSPACE}/build`.

## Run demo
```shell
cd ${WORKSPACE}
$ torchrun --nproc_per_node 1 example/demo.py
[rank=0] Create ChunkTensor
[rank=0] Load data
[rank=0] Print HostTensor in ChunkTensor:
tensor([25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
        61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
        79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
        97, 98, 99])
[rank=0] Print DeviceTensor in ChunkTensor:
tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24], device='cuda:0')
```

## Single Node Benchmark

See [oppo_bench/single_node](Dist-GPU-sampling/oppo_bench/single_node/README.md)

## Multi Nodes BenchmarkSee 
See [oppo_bench/multi_node](Dist-GPU-sampling/oppo_bench/multi_node/README.md)