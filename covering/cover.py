import argparse


import torch, torchvision
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import numpy as np
import random
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
from sklearn.metrics import roc_auc_score


from torch_geometric.nn import WLConv, WLConvContinuous
import ot

import sklearn
from sklearn.cluster import KMeans
import sklearn_extra
from sklearn_extra.cluster import KMedoids


import models.models as models
import models.transforms as transforms
import utils



class WWL(torch.nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList([WLConvContinuous() for _ in range(num_layers)])

    def forward(self, x, edge_index, batch=None):
        xs = []
        for conv in self.convs:
            x = conv(x, edge_index)
            xs.append(x)
        x = torch.cat(xs, dim=-1)
        return x



def cline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--ind-dim", type=int, default=6)
    parser.add_argument("--ind-type", type=str, default='None')
    parser.add_argument('--k-weak', type=int, default=0)
    parser.add_argument("--dataset", type=str, default='None')
    

    args = parser.parse_args()
    args.cuda = True

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.device = torch.device(torch.cuda.current_device()) \
        if (torch.cuda.is_available() and args.cuda) else torch.device('cpu')
    print(args.device)

    torch.set_printoptions(precision=3, linewidth=300)

    if args.ind_type == 'Tinhofer': args.ind_type += str(args.k_weak)

    
    return args


def main():
    args = cline()
    print(args)
    name = args.dataset
    train_d = TUDataset(root='../data', name=name)

    trs = [transforms.EdgeConstant()]
    if args.dataset in ['COLLAB', 'IMDB-BINARY']:
        trs.append(torch_geometric.transforms.Constant()) 
    
    if args.ind_type == 'RNI':
        trs.append( transforms.RNI(prob=1.0, dims=args.ind_dim) )
    elif args.ind_type == 'RP':
        trs.append( transforms.RP(dims=args.ind_dim) ) 
    elif args.ind_type[:-1] == 'Tinhofer':
        trs.append( transforms.TinhoferW(output_dim=args.ind_dim, k_weak=args.k_weak) )
    elif args.ind_type == 'LPE':
        trs.append( torch_geometric.transforms.AddLaplacianEigenvectorPE(k=1, attr_name=None, is_undirected=True)  )
        trs.append( transforms.ToAbsolute(dims = args.ind_dim) )
    
    if args.dataset in ['COLLAB', 'IMDB-BINARY']:
        trs.append(transforms.OneHotSqrtDegree(max_degree=24)) 

    dataset = TUDataset(root='../data', name=name, transform=torchvision.transforms.Compose(trs)).shuffle()
    train_dataset = dataset[len(dataset) // 10:]
    test_dataset = dataset[:len(dataset) // 10]


    train = train_dataset


    model = WWL(num_layers=args.layers)
    data = Batch.from_data_list(train)
    points = model(data.x, data.edge_index)
    points = points.to(torch.float64)

    points = torch_geometric.utils.unbatch(points, data.batch)
    d = torch.zeros((len(points), len(points)))
    for i in range(len(points)):
        if i%100 == 0: print(i, flush=True)
        for j in range(i+1, len(points)):
            costs = ot.dist(points[i].numpy(), points[j].numpy(), metric='euclidean')
            d[i, j] = ot.emd2([], [], costs, numItermax=1000, check_marginals=False)
            d[j, i] = d[i, j]
    d = d / torch.max(d)
    d[d < 0.0] = 0.0
    n_points = len(points)
    

    for i in range(0, 100):
        k = int(np.sqrt(np.sqrt(2)) ** i + 0.01)
        if k > n_points: break

        kmeans = KMedoids(n_clusters=k, random_state=0, metric='precomputed', init='k-medoids++').fit(d)
        centers = kmeans.medoid_indices_
        labels = kmeans.labels_
        costs_1 = np.zeros(k)
        costs_max = np.zeros(k)
        for p in range(len(points)):
            dist = d[ p , centers[labels[p]] ]
            costs_1[labels[p]] += dist
            costs_max[labels[p]] = max( costs_max[labels[p]], dist) 

        cost_1 = np.sum(costs_1) / n_points
        cost_max = np.max(costs_max)
        
        print(k, cost_1, cost_max)



if __name__ == "__main__":
    main()
