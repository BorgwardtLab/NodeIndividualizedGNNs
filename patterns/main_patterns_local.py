import argparse
import numpy as np
import random

from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
import torch, torchvision
import torch_geometric
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter


import models.models as models
import models.egonn as egonn
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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument('--k-weak', type=int, default=0)
    parser.add_argument("--dataset", type=str, default='None')
    parser.add_argument('--resample', action='store_true')
    

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

    assert(args.ind_type in {'None', 'RP', 'Tinhofer', 'LPE', 'RWPE'})
    if args.ind_type == 'Tinhofer': args.ind_type += str(args.k_weak)

    dataset = args.dataset
    args.logger = SummaryWriter(log_dir=f'runs/{dataset}/layers{args.layers}_hid{args.hidden}_indiv{args.ind_dim}/local_indiv{args.ind_type}_res{args.resample}_seed{args.seed}')

    return args


def main():
    args = cline()

    
    trs = [torch_geometric.transforms.Constant(), transforms.EdgeConstant()]
    if args.ind_type == 'RP':
        trs.append( transforms.RP(dims=args.ind_dim) )
    elif args.ind_type[:-1] == 'Tinhofer':
        trs.append( transforms.TinhoferW(output_dim=args.ind_dim, k_weak=args.k_weak) )
    
    transform=torchvision.transforms.Compose(trs) 

    train_d, test_d = torch.load(f'../data/{args.dataset}.pt')

    train_egos = []
    for graph in train_d:
        train_egos.append( egonn.explode_graph(graph, args.layers, transform) )
    test_egos = []
    for graph in test_d:
        test_egos.append( egonn.explode_graph(graph, args.layers, transform) )    
    
    train_loader = DataLoader(train_egos, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_egos, batch_size=args.batch_size)
    


    model = egonn.EGONN(test_egos[0].x.shape[1], args.hidden, pattern_channels=1, out_channels=2, layers=1).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    epochs = 1000

    for epoch in range(epochs):
        print("Epoch", epoch)
        
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
