import torch
import networkx as nx
from torch_geometric.utils import to_networkx

def graph_representation(data, num_iterations):

    G = to_networkx(data, to_undirected=True)

    for node in G.nodes:
        G.nodes[node]['label'] = str(data.x[node].tolist())
    
    for _ in range(num_iterations):
        new_labels = {}
        for node in G.nodes:
            neighbor_labels = sorted([G.nodes[neighbor]['label'] for neighbor in G.neighbors(node)])
            current_label = G.nodes[node]['label']
            new_labels[node] = str(hash((current_label, tuple(neighbor_labels))))
        nx.set_node_attributes(G, new_labels, 'label')

    labels = sorted(list(nx.get_node_attributes(G, 'label').values()))
    return str(hash(tuple(labels)))
