import argparse

import torch, torchvision
import torch_geometric
from torch_geometric.loader import DataLoader
import numpy as np
import random
from torch_geometric.datasets import TUDataset, LRGBDataset
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
import copy
import networkx as nx
from torch_geometric.utils.convert import from_networkx
import pickle

import models.models as models
import models.transforms as transforms
import models.wl as wl
import utils


def cline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--ind-dim", type=int, default=8)
    parser.add_argument("--ind-type", type=str, default='None')
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=128)
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


def to_torch(g,c):
    nx.set_node_attributes(g, dict.fromkeys(g.nodes(), [1.0]), "label")
    G = from_networkx(g, group_node_attrs=['label'])
    G.y = torch.tensor(c)
    return G



def main():

    args = cline()

    args.logger = SummaryWriter(log_dir=f'logs/expressivity_{args.dataset}_ind-type={args.ind_type}_layers={args.layers}_seed={args.seed}')

    # read graphs
    if args.dataset == 'MCF-7':
        dataset = []
        data = TUDataset(root='data', name=args.dataset)
        dataset += [x for x in data]
        selected_graphs_idcs = [17967, 17973, 17965, 17969, 17966, 17970, 3134, 13431, 17972, 17974, 17940, 17942, 17968, 17971]
        dataset = [dataset[i] for i in selected_graphs_idcs]
    elif args.dataset in 'Peptides-func':
        dataset = []
        data = LRGBDataset(root='data', name=args.dataset, split='train')
        dataset += [x for x in data]
        data = LRGBDataset(root='data', name=args.dataset, split='val')
        dataset += [x for x in data]
        data = LRGBDataset(root='data', name=args.dataset, split='test')
        dataset += [x for x in data]
        selected_graphs_idcs = [6184, 13799, 14413, 5504, 13491, 4249, 11354, 4454, 6501, 5881, 6023, 11063, 11803, 8350, 14318]
        dataset = [dataset[i] for i in selected_graphs_idcs]
    elif args.dataset in ['cycles-pin', 'csl-pin']:
        file = open('data/expressivity/'+args.dataset+'.pickle', 'rb')
        graphs, classes = pickle.load(file)
        classes = [c%2 for c in classes]
        dataset = [to_torch(g,c) for (g,c) in zip(graphs, classes)]
    else:
        raise Exception('Dataset not present')
    
    # assign binary classes  
    for i,g in enumerate(dataset):
        g.y = torch.tensor(i%2)

    for g in dataset:
        print(g)
    
    # setup necessary node and edge label transforms
    trs = [transforms.EdgeConstant(), transforms.NodeAttributesToFloat()] #torch_geometric.transforms.Constant()
    if args.ind_type == 'RNI':
        trs.append( transforms.RNI(prob=1.0, dims=args.ind_dim) )
    elif args.ind_type == 'RP':
        trs.append( transforms.RandPerm(dims=args.ind_dim) ) 
    elif args.ind_type[:-1] == 'Tinhofer':
        trs.append( transforms.TinhoferW(output_dim=args.ind_dim, k_weak=args.k_weak) )
    transform = torchvision.transforms.Compose(trs) 
    
    # generate train and test data
    train = [transform(x) for x in dataset]
    train = train * int((5 * args.batch_size / len(train)) +1)
    test = [transform(x) for x in dataset]

    print('number of graphs:', len(test))
    wl_hashes = set()
    for g in test:
        g_hash = wl.graph_representation(g, args.layers)
        wl_hashes.add(g_hash)
    print(f'number of graphs wl-distinguishable after {args.layers} iterations:', len(wl_hashes))


    model = models.NetMLP(train[0].x.shape[1], args.hidden, out_channels=2, layers=args.layers).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    batch_size = args.batch_size
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

    for epoch in range(1000):

        loss = utils.train_step(model, train_loader, optimizer, device=args.device)
        args.logger.add_scalar("Loss/Train", loss, epoch)

        labels, preds = utils.test_step(model, train_loader, device=args.device)
        train_roc = roc_auc_score(labels, preds)
        train_acc = accuracy_score(labels, preds, normalize=True)

        labels, preds = utils.test_step(model, test_loader, device=args.device)
        test_roc = roc_auc_score(labels, preds)
        test_acc = accuracy_score(labels, preds, normalize=True)

        args.logger.add_scalar("AUROC/Train", train_roc, epoch)
        args.logger.add_scalar("AUROC/Test", test_roc, epoch)
        args.logger.add_scalar("AUROC/Difference", train_roc - test_roc, epoch)
        args.logger.add_scalar("Acc/Train", train_acc, epoch)
        args.logger.add_scalar("Acc/Test", test_acc, epoch)
        args.logger.add_scalar("Acc/Difference", train_acc - test_acc, epoch)

        if epoch%1 == 0: 
            print("Epoch", epoch)
            print(train_roc, test_roc, ' ', train_acc, test_acc, ' ', loss.item())

    args.logger.flush()
    args.logger.close()


if __name__ == "__main__":
    main()
