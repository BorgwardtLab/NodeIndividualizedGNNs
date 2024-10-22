import torch
import torch_geometric
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.data import Data, Batch, Dataset
import torch.nn.functional as F

from models.models import *



def explode_graph(graph, layers, transform):
    neigh_loader = NeighborLoader(graph, num_neighbors=[-1]*layers, subgraph_type='induced', disjoint=False, transform=transform)
    tmp = []
    for neigh in neigh_loader:
        del neigh.batch_size, neigh.n_id, neigh.e_id, neigh.input_id, neigh.y, neigh.num_nodes
        tmp.append(neigh)
    ego_graph = Batch.from_data_list(tmp)
    ego_graph.y = graph.y
    ego_graph.ego = ego_graph.batch
    return Data(x=ego_graph.x, edge_index=ego_graph.edge_index, edge_attr=ego_graph.edge_attr, y=graph.y, ego=ego_graph.batch)



class EGONN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, pattern_channels, out_channels, layers, use_max_aggr=False):
        super().__init__()
        self.num_classes = out_channels

        self.convs = torch.nn.ModuleList()
        for _ in range(layers):
            self.convs.append(GraphConv(in_channels, hidden_channels, aggr='add', bias=True))
            in_channels = hidden_channels

        assert layers > 0
        self.mlp = torch_geometric.nn.MLP([pattern_channels, 2*out_channels, out_channels])
        self.aggr = torch_geometric.nn.aggr.SoftmaxAggregation(t=1, learn=False, semi_grad=True)
        self.down = torch_geometric.nn.dense.linear.Linear(hidden_channels, pattern_channels, bias=False)
        

    def forward_node(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = self.down(x)
        return x

    def forward(self, data):
        ego = data.ego + data.batch*(torch.max(data.ego)+1)
        batch = torch_geometric.nn.global_max_pool(data.batch, ego)

        xi = self.forward_node(data.x, data.edge_index)
        x = torch_geometric.nn.global_add_pool(xi, ego)
        x = self.aggr(x, batch)
        lgits = self.mlp(x)
        return lgits


