import torch
import random
import numpy as np
from torch_geometric.transforms import BaseTransform
import torch, torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, WLConv, GCNConv, GATConv, GINConv, global_mean_pool
from torch_geometric.utils import * 

from models.tinhofer import *

np.seterr(over='ignore')


class EdgeConstant(BaseTransform):
    def __init__(self, value: float = 1.0):
        self.value = value

    def __call__(self, data):
        c = torch.full((data.edge_index.shape[1], 1), self.value, dtype=torch.float)
        data.edge_attr = c
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(value={self.value})'


class NodeAttributesToFloat(BaseTransform):
    def __call__(self, graph: Data):
        if graph.x is not None:
            graph.x = graph.x.float()
        return graph
        

class ToAbsolute(BaseTransform):
    def __init__(self, dims=1, edge_labels=False):
        self.dims = dims
        pass

    def __call__(self, data):
        tmp = torch.abs(data.x[:, -1]) 
        tmp = (tmp * 1) * (2 ** self.dims) 
        tmp = tmp.to(int)

        mask = 2 ** torch.arange(self.dims - 1, -1, -1).to(tmp.device, int)
        tmp = tmp.unsqueeze(-1).bitwise_and(mask).ne(0).float()

        data.x = torch.cat([data.x[:, :-1], tmp.to(data.x.device)], dim=-1)
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class ExpandRWPE(BaseTransform):
    def __init__(self, dims):
        self.dims = dims
        pass

    def __call__(self, data):
        tmp = data.x[:, -1]
        tmp = tmp  * (2 ** self.dims) 
        tmp = tmp.to(int)

        mask = 2 ** torch.arange(self.dims - 1, -1, -1).to(tmp.device, int)
        tmp = tmp.unsqueeze(-1).bitwise_and(mask).ne(0).float()

        data.x = torch.cat([data.x[:, :-1], tmp.to(data.x.device)], dim=-1)
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class ToBinaryLabels(BaseTransform):
    def __init__(self, dims=1, edge_labels=False):
        pass

    def __call__(self, data):
        if data.y == 2: data.y -= 1
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class OneHotSqrtDegree(BaseTransform):
    def __init__(
        self,
        max_degree: int,
    ) -> None:
        self.max_degree = max_degree

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        idx, x = data.edge_index[0], data.x
        deg = torch.sqrt(degree(idx, data.num_nodes)).to(dtype=torch.long)
        deg = one_hot(deg, num_classes=self.max_degree + 1)

        if x is not None:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
        else:
            data.x = deg

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.max_degree})'




# taken from https://github.com/bjfranks/IRNI
class RNI(BaseTransform):
    def __init__(self, prob=1, dims=1, edge_labels=False, cat=True):
        self.prob = prob
        self.dims = dims
        self.cat = cat

    def __call__(self, data):
        if self.dims != 0:
            x = data.x

            c = torch.rand((data.num_nodes, self.dims), dtype=torch.float)
            n = torch.full((data.num_nodes, self.dims), 0,  dtype=torch.float)
            r = torch.rand((data.num_nodes, 1), dtype=torch.float)
            c = torch.where(r < self.prob, c, n)

            if x is not None and self.cat:
                x = x.view(-1, 1) if x.dim() == 1 else x
                data.x = torch.cat([x, c.to(x.dtype).to(x.device)], dim=-1)
            else:
                data.x = c

        return data

    def __repr__(self):
        return '{}(prob={}, dims={}, cat={})'.format(self.__class__.__name__, self.prob, self.dims, self.cat)


class RP(BaseTransform):
    def __init__(self, dims=6, max_individ_labels=None):
        self.dims = dims
        self.max_individ_labels = 2 ** dims
        if max_individ_labels: self.max_individ_labels = max_individ_labels

    def __call__(self, data):
        o = torch.full((data.num_nodes, self.dims), 0, dtype=data.x.dtype)
        mask = 2 ** torch.arange(self.dims - 1, -1, -1).to(o.device, int)
        order = torch.from_numpy(np.random.permutation(data.num_nodes))
        order = order % self.max_individ_labels
        for node in range(data.num_nodes):
            o[node] = order[node].unsqueeze(-1).bitwise_and(mask).ne(0).float()
                    
        data.x = torch.cat([data.x, o.to(data.x.device)], dim=-1)
        return data

    def __repr__(self):
        return '{}(depth={}, fill={})'.format(self.__class__.__name__, self.depth, self.fill)



