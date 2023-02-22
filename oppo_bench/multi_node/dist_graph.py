import torch
import dgl
from utils.load_graph import load_papers400m_sparse, load_ogb
from utils.chunktensor_sampler import get_available_memory


class DistGraph(object):

    def __init__(self,
                 libdgs_path,
                 graph_path,
                 graph_name,
                 root,
                 rank,
                 world_group,
                 cached_feature_name,
                 feat_cache_rate,
                 graph_cache_rate,
                 with_bias=False):
        torch.ops.load_library(libdgs_path)

        self._tmp = {}
        self.root = root
        self.rank = rank
        self.world_group = world_group
        self.ndata = {}
        self.edata = {}

        self.cached_feature_name = cached_feature_name
        if self.root == self.rank:
            print("Feature name for cached: ", self.cached_feature_name)

        self.feat_cache_rate = float(feat_cache_rate)
        self.graph_cache_rate = float(graph_cache_rate)

        self._tmp = {}

        self._g = None
        metadata = None
        broadcast_list = [None]
        if root == rank:
            if graph_name == "ogbn-products":
                g, num_labels = load_ogb("ogbn-products", root=graph_path)
            elif graph_name == "ogbn-papers100M":
                g, num_labels = load_ogb("ogbn-papers100M", root=graph_path)
            elif graph_name == "ogbn-papers400M":
                g, num_labels = load_papers400m_sparse(root=graph_path)

            if with_bias:
                g.edata['probs'] = torch.randn((g.num_edges(), )).abs().float()

            metadata = self._generate_meta(g, num_labels)
            self._g = g
            broadcast_list = [metadata]

        torch.distributed.broadcast_object_list(broadcast_list, root,
                                                world_group)
        self.metadata = broadcast_list[0]

        if self.rank == self.root:
            print(self.metadata)

    def create_shared_graph(self):
        self._available_mem = get_available_memory(
            torch.cuda.current_device(),
            torch.cuda.max_memory_reserved() + 3 * 1024 * 1024 * 1024 +
            self.metadata['num_nodes'], self.world_group)

        self._parse_meta(self.metadata)
        self._create_graph_chunktensor()

        del self._g
        del self._tmp

        if self.root == self.rank:
            print("ndata", self.ndata.keys())
            print("edata", self.edata.keys())

    def _generate_meta(self, g, num_labels):
        metadata = {}
        metadata['num_nodes'] = g.num_nodes()
        metadata['num_edges'] = g.num_edges()
        metadata['num_labels'] = num_labels
        metadata['graph_type'] = g.nodes().dtype

        for key, value in g.ndata.items():
            if key.endswith('mask'):
                self._tmp[key] = torch.nonzero(value).reshape(-1)
                metadata['ndata/' + key] = (self._tmp[key].dtype,
                                            list(self._tmp[key].size()))
            else:
                metadata['ndata/' + key] = (value.dtype, list(value.size()))

        for key, value in g.edata.items():
            metadata['edata/' + key] = (value.dtype, list(value.size()))

        return metadata

    def _create_graph_chunktensor(self):
        if self.root == self.rank:
            print("Cache indices:")
        self.chunk_indices = self._create_chunktensor(
            [self.metadata['num_edges']], self.metadata['graph_type'],
            self.graph_cache_rate)

        if self.root == self.rank:
            print("Cache indptr:")
        self.chunk_indptr = self._create_chunktensor(
            [self.metadata['num_nodes'] + 1], self.metadata['graph_type'],
            self.graph_cache_rate)

        if self.root == self.rank:
            indptr, indices, _ = self._g.adj_sparse('csc')
            self.chunk_indptr._CAPI_load_from_tensor(indptr)
            self.chunk_indices._CAPI_load_from_tensor(indices)
        torch.distributed.barrier()

    def _parse_meta(self, metadata):
        self.num_nodes = metadata
        self.num_edges = metadata
        self.num_labels = metadata

        for key, meta in metadata.items():
            if key.startswith('edata/') or key.startswith('ndata/'):
                self._create_feat_chunk_tensor(key, meta)

    def _create_feat_chunk_tensor(self, key, meta):
        data_type = key[0:key.find('/')]
        tensor_name = key[key.find('/') + 1:]

        chunktensor = None
        if key in self.cached_feature_name:
            if self.root == self.rank:
                print("Cache {}:".format(key))
            chunktensor = self._create_chunktensor(meta[1], meta[0],
                                                   self.feat_cache_rate)
        else:
            chunktensor = self._create_chunktensor(meta[1], meta[0], 0)

        if self.rank == self.root:
            if data_type == 'ndata':
                if key.endswith('mask'):
                    chunktensor._CAPI_load_from_tensor(self._tmp[tensor_name])
                else:
                    chunktensor._CAPI_load_from_tensor(
                        self._g.ndata[tensor_name])
            else:
                chunktensor._CAPI_load_from_tensor(self._g.edata[tensor_name])
        torch.distributed.barrier()

        if data_type == 'ndata':
            if tensor_name.endswith('mask'):
                tensor_name = tensor_name.replace('mask', 'ids')
            self.ndata[tensor_name] = chunktensor
        else:
            self.edata[tensor_name] = chunktensor

    def _create_chunktensor(self, shape, type, rate):
        total_size = torch.tensor([], dtype=type).element_size()
        for k in shape:
            total_size = total_size * k

        available_mem = max(
            self._available_mem -
            torch.ops.dgs_ops._CAPI_get_current_allocated(), 0)
        cached_size_per_gpu = int(
            min(
                total_size * rate //
                torch.distributed.get_world_size(self.world_group),
                available_mem))
        chunk_tensor = torch.classes.dgs_classes.ChunkTensor(
            shape, type, cached_size_per_gpu)
        if self.rank == self.root and cached_size_per_gpu > 0:
            print(
                "Cache size per GPU {:.3f} GB, all gpus total cache rate = {:.3f}"
                .format(
                    cached_size_per_gpu / 1024 / 1024 / 1024,
                    cached_size_per_gpu *
                    torch.distributed.get_world_size(self.world_group) /
                    total_size))
        return chunk_tensor