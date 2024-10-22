from typing import List
import torch
import torch_geometric

from torch_geometric.nn.conv import MessagePassing
from torch_sparse import matmul



class GraphConv(MessagePassing):
    def __init__(
            self,
            in_channels,
            out_channels,
            aggr: str = 'add',
            bias: bool = True,
            **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_rel = torch_geometric.nn.dense.linear.Linear(in_channels, out_channels, bias=bias)
        self.lin_root = torch_geometric.nn.dense.linear.Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        out = self.lin_rel(out)

        x_r = x
        out += self.lin_root(x_r)

        return out

    def message(self, x_j):
        return x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)



class GraphConvMLP(MessagePassing):
    def __init__(
            self,
            in_channels,
            out_channels,
            aggr: str = 'add',
            bias: bool = True,
            **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_rel = torch_geometric.nn.MLP([in_channels, out_channels, out_channels])
        self.lin_root = torch_geometric.nn.MLP([in_channels, out_channels, out_channels])

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        out = self.lin_rel(out)

        x_r = x
        out += self.lin_root(x_r)

        return out

    def message(self, x_j):
        return x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)




class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, layers, use_max_aggr=False):
        super().__init__()
        self.num_classes = out_channels

        self.convs = torch.nn.ModuleList()
        for _ in range(layers):
            self.convs.append(GraphConv(in_channels, hidden_channels, aggr='add', bias=True))
            in_channels = hidden_channels

        if layers != 0:
            self.mlp = torch_geometric.nn.MLP([hidden_channels, hidden_channels, out_channels])
        else:
            self.mlp = torch_geometric.nn.MLP([in_channels, hidden_channels, out_channels])

        self.aggr = torch_geometric.nn.global_add_pool
        if use_max_aggr: self.aggr = torch_geometric.nn.global_max_pool

    def forward(self, data):
        x = data.x 
        for conv in self.convs:
            x = conv(x, data.edge_index).relu()
        x = self.aggr(x, data.batch)

        return self.mlp(x)



class NetMLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, layers, use_max_aggr=False):
        super().__init__()
        self.num_classes = out_channels

        self.convs = torch.nn.ModuleList()
        for _ in range(layers):
            self.convs.append(GraphConvMLP(in_channels, hidden_channels, aggr='add', bias=True))
            in_channels = hidden_channels

        if layers != 0:
            self.mlp = torch_geometric.nn.MLP([hidden_channels, hidden_channels, out_channels])
        else:
            self.mlp = torch_geometric.nn.MLP([in_channels, hidden_channels, out_channels])

        self.aggr = torch_geometric.nn.global_add_pool
        if use_max_aggr: self.aggr = torch_geometric.nn.global_max_pool

    def forward(self, data):
        x = data.x 
        for conv in self.convs:
            x = conv(x, data.edge_index).relu()
        x = self.aggr(x, data.batch)

        return self.mlp(x)


