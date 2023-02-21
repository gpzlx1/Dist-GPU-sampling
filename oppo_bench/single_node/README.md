# Single node training

## Dependencies

- CUDA 11.3

- PyTorch 1.12.1

- DGL 0.9.1

- NumPy 1.23.5

- OGB 1.3.5

- NCCL 2.10.3

## Dataset

- ogbn-products

- ogbn-papers100M

- ogbn-papers400M

## How to run

### Baseline

```bash
python3 baseline.py \
--num-gpu 8 \
--num-epochs 3 \
--dataset ogbn-papers400M \
--root dataset/ \
--model graphsage \
--batch-size 1000 \
--fan-out 15,15,15 \
--bias \
--print-train
```

args:

- `--num-gpu` The number of GPUs, default = 8

- `--num-epochs` The number of epochs of training, default = 1

- `--dataset` The dataset, support `ogbn-products`, `ogbn-papers100M` and `ogbn-papers400M`, default = `ogbn-papers400M`

- `--root` The directory which stores the dataset

- `--model` The model of training, support `graphsage` and `gat`, default = `graphsage`

- `--batch-size` The number of seeds in each iteration, default = 1000

- `--fan-out` The fanout of sampler, default = `15,15,15`

- `--bias` Sample neighbors with bias

- `--print-train` Print loss and accuracy when training

### Chunktensor version

In this script, with *Chunktensor*, the graph structure tensors (indptr, indices, probs) can be cached across multi gpus' memory, to accelerate sampling.

And there are 2 methods to accelerate feature loading:

- Chunktensor cache features

- Compress features by quantization

These 2 methods need different args:

- Chunktensor cache features

  ```bash
  python3 chunktensor.py \
  --num-gpu 8 \
  --num-epochs 3 \
  --dataset ogbn-papers400M \
  --root dataset/ \
  --model graphsage \
  --batch-size 1000 \
  --fan-out 15,15,15 \
  --libdgs ../Dist-GPU-sampling/build/libdgs.so \
  --graph-cache-rate 1 \
  --feat-cache-rate 1 \
  --bias \
  --print-train
  ```

  args:

  - `--libdgs` The directory of `libdgs.so` (the compiled library of *chunktensor*)

  - `--graph-cache-rate`

    The cache rate of graph structure tensors

    Note: this is the cache rate of all the gpus. For example, if `--num-gpu 2` and `--graph-cache-rate 0.4` are set, the cache rate of each gpu is `0.2`.

  - `--feat-cache-rate`

    The cache rate of features

  Note: If gpu memory is not large enough to achieve the given cache rate, cache priority: features > probs > indices > indptr.

- Compress features by quantization

  ```bash
  python3 chunktensor.py \
  --num-gpu 8 \
  --num-epochs 3 \
  --dataset ogbn-papers400M \
  --root dataset/ \
  --model graphsage \
  --batch-size 1000 \
  --fan-out 15,15,15 \
  --libdgs ../Dist-GPU-sampling/build/libdgs.so \
  --graph-cache-rate 1 \
  --libbifeat ../GPU-Feature-Quantization/build/libbifeat.so \
  --compress-feat \
  --compress-mode vq \
  --compress-width 12 \
  --compress-length 1024 \
  --compress-feat-save-root data/ \
  --bias \
  --print-train
  ```

  args:

  - `--libbifeat` The directory of `libbifeat.so` (the compiled library of *feature quantization*)

  - `--compress-feat` Enable feature compressing

  - `--compress-mode` The mode of feature compressing, can be `sq` or `vq`, deafult = `sq`

    - `sq` means scalar quantization

    - `vq` means vector quantization

  - `--compress-length`
  
    - If mode is `sq`, length means the number of bit to use, can be 1,2,4...16,32, if length is 32, no quantization would be done

    - If mode is `vq`, length means the number of codebook entries, normally select big numbers like 1024, 2048, 8192, note that the larger the length is, the slower compressing would be

  - `--compress-width`
  
    For `vq` mode only, the width of each codebook entries, the features would be split into Ceil (feature_dim / width) parts for vector quantization

  - `--compress-feat-save-root` If set, the compressed featurs will be saved in this dierctory

  - `--gpu-cache-full-feat` If set, every gpu will cache the full features.