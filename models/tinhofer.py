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


np.seterr(over='ignore')


def mash(input):
    output = torch.sum(input*torch.tensor([2 ** (input.shape[1]-1-i) for i in range(input.shape[1])]), dim=1, dtype=torch.int64) 
    return output



class TinhoferW(BaseTransform):
    def __init__(self, output_dim, k_weak):
        self.output_dim = output_dim
        self.k_weak = k_weak
        

    def forward(self, data, batch=None):
        x1 = mash(data.x)
        edge_index = torch_geometric.utils.to_undirected(data.edge_index)
        edge_index = sort_edge_index(edge_index, num_nodes=data.x.size(0), sort_by_row=False)
        row, col = edge_index[0], edge_index[1]
        deg = degree(col, data.x.size(0), dtype=torch.long).tolist()

        color_classes = None
        if self.k_weak == 0: color_classes = x1.clone()

        # break symmetry in orbits of size > 1
        while True:
            # color refinement
            for i_cr in range(1, min(16, x1.shape[0])):
                out = []
                for node, neighbors in zip(x1.tolist(), x1[row].split(deg)):
                    hashx = hash(tuple([node] + neighbors.sort()[0].tolist()))
                    out.append(hashx)
                x1 = torch.tensor(out, device=x1.device)

                if color_classes == None and (i_cr == self.k_weak): color_classes = x1.clone()
            if color_classes == None: color_classes = x1.clone() # for smaller graphs
            # end color refinement 
            
            uniq, inv, counts = torch.unique(x1, return_inverse=True, return_counts=True)
            orbit_size = counts[ inv ]
            orbits2_pos = torch.nonzero(orbit_size > 1, as_tuple=True)[0]

            if orbits2_pos.shape[0] > 0:
                xs = x1[orbits2_pos]
                idx = torch.argmin(xs)
                x1[ orbits2_pos[idx] ] += 1 # hopefully this is unique 
            else: #all of size 1
                break
            

        x_out = torch.zeros((x1.shape[0], self.output_dim), device=x1.device)
        tmp = torch.zeros((x1.shape[0]), dtype=int, device=x1.device)
        mask = 2 ** torch.arange(self.output_dim - 1, -1, -1).to(x_out.device, int)

        uniq, counts = torch.unique(color_classes, return_counts=True)
        for color in uniq:
            nodes = torch.nonzero(color_classes == color, as_tuple=True)[0]
            ind_colors = x1[nodes]
            order = torch.argsort(torch.argsort( ind_colors )) 
            order = order % (2 ** self.output_dim)
            x_out[nodes] = order.unsqueeze(-1).bitwise_and(mask).ne(0).float()
            tmp[nodes] = order

        return x_out


    def __call__(self, data):
        x1 = self.forward(data)
        data.x = torch.cat([data.x, x1], dim=-1)
        return data


