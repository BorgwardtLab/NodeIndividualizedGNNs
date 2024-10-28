import argparse

import torch, torchvision
import torch_geometric
from torch_geometric.loader import DataLoader
import numpy as np
import random
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
import networkx as nx
from torch_geometric.utils.convert import from_networkx
import torch.optim as optim
import pickle

import models.models as models
import models.transforms as transforms
import utils


def cline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--ind-dim", type=int, default=6)
    parser.add_argument("--ind-type", type=str, default='None')
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument('--k-weak', type=int, default=0)
    parser.add_argument('--train-size', type=int, default=1000)
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


def get_permuted_copy(data):
    num_nodes = data.num_nodes
    perm = torch.randperm(num_nodes)
    inverse_perm = torch.zeros_like(perm)
    inverse_perm[perm] = torch.arange(num_nodes)
    permuted_data = torch_geometric.data.Data()
    permuted_data.x = data.x[perm]
    permuted_data.edge_index = inverse_perm[data.edge_index] 
    permuted_data.y = data.y
    return permuted_data



def main():

    args = cline()

    args.logger = SummaryWriter(log_dir=f'logs/vc-dim_{args.dataset}_ind_type={args.ind_type}_layers={args.layers}_seed={args.seed}')

    # load graphs and assign binary labels
    file = open(f'data/vc-dim/{args.dataset}.pickle', 'rb')
    graphs, classes = pickle.load(file)
    classes = [c%2 for c in classes]
    dataset = [to_torch(g,c) for (g,c) in zip(graphs, classes)]

    # generate test and train data
    test_d = []
    for _ in range(1000):
        test_d += [get_permuted_copy(d) for d in dataset]
    random.shuffle(test_d)

    train_d = []
    for _ in range(args.train_size):
        train_d += [get_permuted_copy(d) for d in dataset]
    random.shuffle(train_d)

    # transform graphs
    trs = [ transforms.EdgeConstant()] #torch_geometric.transforms.Constant()
    if args.ind_type == 'RNI':
        trs.append( transforms.RNI(prob=1.0, dims=args.ind_dim) )
    elif args.ind_type == 'RP':
        trs.append( transforms.RP(dims=args.ind_dim) ) 
    elif args.ind_type[:-1] == 'Tinhofer':
        trs.append( transforms.TinhoferW(output_dim=args.ind_dim, k_weak=args.k_weak) )
    transform = torchvision.transforms.Compose(trs) 

    # generate train and test data
    train = [ transform(x) for x in train_d]
    train = train * int((5 * args.batch_size / len(train)) +1)
    test = [ transform(x) for x in test_d]


    model = models.Net(train[0].x.shape[1], args.hidden, out_channels=2, layers=args.layers).to(args.device)

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
