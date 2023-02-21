import torch as th
import math
import numpy as np
import tqdm
import os

from .kmeans import kmeans, get_centers, kmeans_predict
from .packbits import packbits, unpackbits


class Compresser(object):

    def __init__(self, mode="sq", length=1, width=1, device="cpu"):
        """
        Parameters:
        - mode: "vq" or "sq", selecting vector quantization or scalar quantization
        - length: 
            - if mode is sq, length mean the number of bit to use, can be 1,2,4...16,32, if length is 32, no quantization would be done.
            - if mode is vq, length mean the number of codebook entries, normally select big numbers like 1024, 2048, 8192, note that larger the length is, the slower vq would be
        - width: for vq mode only, the width of each codebook entries, the features would be split into Ceil(feature_dim / width) parts for vector quantization
        - device: the device used for compressing, only work for vq, advise on cpu because gpu isn't much faster, it is also used as default device for dequantization
        """
        self.mode = mode
        self.length = length
        self.width = width
        self.device = device
        self.feat_dim = -1
        self.quantized = False
        self.codebooks = None
        self.info = None
        self.fn = None

    def compress(self,
                 features,
                 cache_root=None,
                 dataset_name=None,
                 batch_size=-1):
        """
        Parameters:
        - features : the features to be quantized
        - cache_root : specify the directory where quantized result saved. If cache file exists, compresser will directly use the result.
        - dataset_name: dataset name, to generate the filename of cached result
        - batch_size: for vq only, read and quantize a batch of nodes each time, only giant datasets needs , doesn't affect training.
        """
        self.batch_size = batch_size
        # get cache file name and load if exist
        if cache_root and dataset_name:
            self.fn = os.path.join(
                cache_root, dataset_name + "_" + self.mode + "_" +
                str(self.length) + "_" + str(self.width) + ".pkl")
            if os.path.exists(self.fn):
                (compressed, self.codebooks, self.quantized, self.info,
                 self.feat_dim) = th.load(self.fn)
                return compressed

        # quantize features
        if self.mode == "sq":
            compressed = self.sq_compresser(features)
        elif self.mode == "vq":
            if self.batch_size <= 0:
                compressed = self.vq_compresser(features)
            else:
                compressed = self.vq_compresser_batch(features)
        else:
            raise ValueError("compression mode must be 'sq' or 'vq'")

        # save quantization result
        if self.quantized and self.fn:
            th.save((compressed, self.codebooks, self.quantized, self.info,
                     self.feat_dim), self.fn)

        return compressed

    def vq_compresser(self, features):
        # vector quantization
        self.quantized = True
        self.feat_dim = features.shape[1]
        print("in total ", math.ceil(features.shape[1] / self.width), " parts")
        self.codebooks = th.empty((math.ceil(features.shape[1] / self.width),
                                   self.length, self.width))

        # select data type to save memory
        if self.length <= 256:
            dtype = th.uint8
        elif self.length <= 32768:
            dtype = th.int16
        else:
            dtype = th.int32
        cluster_ids = th.empty(
            (features.shape[0], math.ceil(features.shape[1] / self.width)),
            dtype=dtype)

        # quantize part by part
        for i in range(math.ceil(features.shape[1] / self.width)):
            print("quantizing part ", i)
            X = features[:, i * self.width:i * self.width + self.width]
            dist = X.norm(dim=1, p=2)
            method = "cosine"
            out_num = self.length - 1
            rim = th.quantile(dist[:50000], 0.3 / self.length)
            cluster_ids_x = th.empty(X.shape[0], dtype=th.int32)
            inner = th.lt(dist, rim)
            cluster_ids_x[inner] = self.length - 1
            out = th.ge(dist, rim)
            cluster_ids_o, cluster_centers_o = kmeans(X=X[out],
                                                      num_clusters=out_num,
                                                      distance=method,
                                                      tol=3e-2 * out_num,
                                                      device=self.device)
            cluster_ids_x[out] = cluster_ids_o.to(th.int32)
            self.codebooks[i, :, :features.shape[1] - i * self.width] = th.cat(
                (cluster_centers_o, th.ones(
                    (1, cluster_centers_o.shape[1])).mul_(1e-4)))
            cluster_ids[:, i] = cluster_ids_x
        return cluster_ids

    def vq_compresser_batch(self, features):
        # vector quantization with batching, save memory
        batch_size = self.batch_size
        self.quantized = True
        self.feat_dim = features.shape[1]

        print("in total ", math.ceil(features.shape[1] / self.width), " parts")
        self.codebooks = th.empty((math.ceil(features.shape[1] / self.width),
                                   self.length, self.width))

        # select data type to save memory
        if self.length <= 256:
            dtype = th.uint8
        elif self.length <= 32768:
            dtype = th.int16
        else:
            dtype = th.int32

        # use part of feature nodes to get codebooks
        perm = th.randperm(features.shape[0])
        for i in range(math.ceil(features.shape[1] / self.width)):
            print("quantizing part ", i)
            X = th.tensor(features[perm[:300000],
                                   i * self.width:i * self.width + self.width],
                          dtype=th.float32)
            dist = X.norm(dim=1, p=2)
            method = "cosine"
            out_num = self.length
            rim = th.quantile(dist[:50000], 0.3 / self.length)
            cluster_ids_x = th.empty(X.shape[0], dtype=th.int32)
            out = th.ge(dist, rim)
            cluster_centers_o = get_centers(X=X[out],
                                            num_clusters=out_num,
                                            distance=method,
                                            tol=3e-2 * out_num,
                                            device=self.device)
            self.codebooks[i, :, :features.shape[1] -
                           i * self.width] = cluster_centers_o
        del X
        cluster_ids = th.empty(
            (features.shape[0], math.ceil(features.shape[1] / self.width)),
            dtype=dtype)

        # quantize a batch of nodes each time
        for j in tqdm.trange(math.ceil(features.shape[0] / batch_size),
                             mininterval=1):
            start = j * batch_size
            end = (j + 1) * batch_size
            features_ = th.tensor(features[start:end, :], dtype=th.float32)
            # quantize part by part
            for i in range(math.ceil(features.shape[1] / self.width)):
                method = "cosine"
                X = features_[:, i * self.width:i * self.width + self.width]
                cluster_ids_x = kmeans_predict(X,
                                               self.codebooks[i],
                                               method,
                                               device=self.device)
                cluster_ids[start:end, i] = cluster_ids_x
        del features_
        del features
        return cluster_ids

    def sq_compresser(self, features):
        # scalar quantization
        self.feat_dim = features.shape[1]
        if not th.is_tensor(features):
            features = th.tensor(features, dtype=th.float16)

        # decide quantizing or not
        if self.length == 32 or (self.length == 16
                                 and features.dtype == th.float16):
            self.quantized = False
            return features
        else:
            self.quantized = True

        emin = 0
        emax = 0
        drange = 2**(self.length - 1)

        if self.length < 8:
            dtype = th.uint8
        elif self.length == 8:
            dtype = th.int8
        elif self.length <= 16:
            dtype = th.int16
        else:
            dtype = th.int32

        if self.length < 8:
            # pack bits
            tfeat_dim = int(math.ceil(self.feat_dim / 8 * self.length))
        else:
            tfeat_dim = self.feat_dim
        t_features = th.empty((features.shape[0], tfeat_dim), dtype=dtype)

        epsilon = 1e-5
        print("start compressing, precision=", self.length)
        # get the max and min value to clip
        perm = th.randperm(features.shape[0])
        sample = features[perm[:100000]]
        fmin = max(np.percentile(np.abs(sample), 0.5), epsilon)
        fmax = max(np.percentile(np.abs(sample), 99.5), 2 * epsilon)
        fmin = th.tensor(fmin)
        fmax = th.tensor(fmax)

        # quantize by batch
        quantize_batch_size = 1000000 if self.batch_size <= 0 else self.batch_size
        for start in tqdm.trange(0, features.shape[0], quantize_batch_size):
            end = min(features.shape[0], start + quantize_batch_size)
            features_ = features[start:end].to(th.float32)
            sign = th.sign(features_)
            if self.length == 1:
                # 1-bit quantization
                features_ = th.where(sign <= 0, 0, 1)
            else:
                features_ = th.abs(features_)
                features_ = th.clip(features_, fmin, fmax)
                exp = th.log2(features_)
                emin = th.log2(fmin)
                emax = th.log2(fmax).add(epsilon)

                exp = th.floor((exp - emin) / (emax - emin) * drange)
                if self.length < 8:
                    features_ = th.where(sign <= 0, drange - 1 - exp,
                                         exp + drange)
                else:
                    features_ = th.where(sign <= 0, -1 - exp, exp)

            if self.length < 8:
                t_features[start:end] = packbits(features_.to(th.uint8),
                                                 mask=(1 << self.length) - 1)
            elif self.length == 8:
                t_features[start:end] = th.tensor(features_).to(th.int8)
            elif self.length <= 16:
                t_features[start:end] = th.tensor(features_).to(th.int16)
            else:
                t_features[start:end] = th.tensor(features_).to(th.int32)
            del features_

        mean = features[:10000].float().norm(1).div(features[:10000].shape[0] *
                                                    features.shape[1])
        if mean < 0.05:
            mean += 0.05
        info = th.zeros(4)
        info[0] = emin
        info[1] = emax
        info[2] = mean
        info[3] = drange
        self.info = info
        del features
        return t_features

    def decompress(self, compressed_features, device=None):
        """
        Parameters:
        - compressed_features: features to be dequantized 
        - device: device to perform dequantization, features are loaded into device and dequantize.
        """
        if device is None:
            device = self.device
        else:
            self.device = device
        if self.quantized:
            if self.mode == "vq":
                return self.vq_decompresser(compressed_features, device)
            elif self.mode == "sq":
                return self.sq_decompresser(compressed_features, device)
            else:
                raise ValueError("mode should be 'vq' or 'sq'")
        else:
            return compressed_features.to(th.float32).to(device)

    def vq_decompresser(self, compressed_features, device):
        # vector dequantization
        compressed_features = compressed_features.to(device).to(th.int64)
        self.codebooks = self.codebooks.to(device)
        if device == 'cuda':
            decompressed = th.ops.bifeat_ops._CAPI_vq_decompress(
                compressed_features, self.codebooks, self.feat_dim)
        else:
            num_parts = self.codebooks.shape[0]
            width = self.width
            decompressed = th.empty(
                (compressed_features.shape[0], self.feat_dim),
                dtype=th.float32,
                device=device)
            # dequantize by index selecting
            for i in range(num_parts - 1):
                h = i * width
                t = (i + 1) * width
                decompressed[:, h:t] = th.index_select(
                    self.codebooks[i], 0, compressed_features[:, i].flatten())
            decompressed[:, (num_parts - 1) * width:] = th.index_select(
                self.codebooks[num_parts - 1, :, :self.feat_dim -
                               (num_parts - 1) * width], 0,
                compressed_features[:, num_parts - 1].flatten())
        return decompressed

    def sq_decompresser(self, compressed_features, device):
        # scalar dequantization
        self.info = self.info.to(device)
        emin = self.info[0]
        emax = self.info[1]
        mean = self.info[2]
        drange = self.info[3]

        exp = compressed_features.to(device)
        if self.length < 8:
            exp = unpackbits(exp,
                             mask=2 * drange - 1,
                             shape=[exp.shape[0], self.feat_dim],
                             dtype=th.uint8,
                             device=device)
        if self.length > 1:
            if self.length < 8:
                # unsigned to signed
                exp = exp.to(th.float32) - drange
            exp = exp.add(0.5)
            sign = th.sign(exp)
            decompressed = th.exp2(exp.abs_().mul_(
                (emax - emin) / drange).add_(emin)).mul_(sign)
        else:
            # 1-bit dequantization
            decompressed = (exp.to(th.float32).sub_(0.5)).mul_(2 * mean)
        return decompressed
