import argparse

import torch, torchvision
import torch_geometric
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import numpy as np
import random
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
import networkx as nx
from torch_geometric.utils.convert import from_networkx

import pickle

import models.models as models
import models.transforms as transforms
import utils


def cline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--ind-dim", type=int, default=6)
    parser.add_argument("--ind-type", type=str, default='None')
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument('--k-weak', type=int, default=0)
    

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
    nx.set_node_attributes(g, dict.fromkeys(g.nodes(), [1.0]), "v_label")
    nx.set_edge_attributes(g, dict.fromkeys(g.edges(), [1.0]), "e_label")
    G = from_networkx(g, group_node_attrs=['v_label'], group_edge_attrs=['e_label'])
    G.y = torch.tensor(c)
    return G


def generate_dataset(graphs, classes, N):
    
    torch_graphs = []
    for _ in range(N):
        for g,c in zip(graphs,classes):

            g_copy_edges = list(g.edges())
            random.shuffle(g_copy_edges)
            g_copy = nx.Graph(g_copy_edges)

            permuation = list(g_copy.nodes())
            random.shuffle(permuation)
            map = dict(zip(list(g_copy.nodes()), permuation))
            g_copy = nx.relabel_nodes(g_copy, map)

            tg = to_torch(g_copy, c)
            torch_graphs.append(tg)
    
    return torch_graphs


def main():

    args = cline()

    ntest = 1

    for n in [41]: 

        for ntrain in [1]: 

            args.logger = SummaryWriter(log_dir=f'runs/csl_own/csl_pin_indiv{args.ind_type}/acc_layers{args.layers}_dim{args.ind_dim}_hidden{args.hidden}_lr{args.lr}_size{n}_train{ntrain}_test{ntest}')

            for ii in range(5):

                print('>', n, ntrain, ii)

                filename = f'csl_{n}_pin'
                nmb_classes = 2
                file = open('../data/csl_own/'+filename+'.pickle', 'rb')
                graphs, classes = pickle.load(file)
                classes = [c%2 for c in classes]
                nmb_graphs = len(graphs)
                file.close()

                print('number of graphs', len(graphs))
                for g,c in zip(graphs, classes):
                    print(g, 'class', c)

                train_d = generate_dataset(graphs, classes, ntrain)
                random.shuffle(train_d)
                test_d = generate_dataset(graphs, classes, ntest)
                random.shuffle(test_d)


                print('train size', len(train_d), 'test size', len(test_d))
                print('0s in train', len([int(g.y) for g in train_d if int(g.y)==0]))
                print('0s in test ', len([int(g.y) for g in test_d  if int(g.y)==0]))

                
                
                trs = [ transforms.EdgeConstant()] #torch_geometric.transforms.Constant()

                if args.ind_type == 'RNI':
                    trs.append( transforms.RNI(prob=1.0, dims=args.ind_dim) )
                elif args.ind_type == 'RP':
                    trs.append( transforms.RP(dims=args.ind_dim) ) 
                elif args.ind_type[:-1] == 'Tinhofer':
                    trs.append( transforms.TinhoferW(output_dim=args.ind_dim, k_weak=args.k_weak) )
                    
                print(args.ind_type)

                transform=torchvision.transforms.Compose(trs) 
                train = [ transform(x) for x in train_d]
                train = train * int((5 * args.batch_size / len(train)) +1)
                test = [ transform(x) for x in test_d]

                graph0 = train[0]
                print(graph0.x)


                model = models.NetMLP(train[0].x.shape[1], args.hidden, out_channels=nmb_classes, layers=args.layers).to(args.device)

                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

                batch_size = args.batch_size
                train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)


                for epoch in range(1000):
                    if epoch%1 == 0: print("Epoch", epoch)
                    

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
                    print(train_roc, test_roc, ' ', train_acc, test_acc, ' ', loss.item())


                args.logger.flush()
                args.logger.close()  


if __name__ == "__main__":
    main()
