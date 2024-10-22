import numpy as np
import random

from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
import torch, torchvision
import torch_geometric
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.data import Data, Batch, Dataset
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score



class ListDataset(Dataset):
    def __init__(self, data, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.data = data

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]





def train_step(model, loader, optimizer, device):
    model.train()
    for data in loader:
        optimizer.zero_grad()
        
        data = data.to(device)
        lgits = model(data)

        loss = F.cross_entropy(lgits, data.y)
        
        loss.backward()
        optimizer.step()
    return loss


def test_step(model, loader, device):
    model.eval()
    labels = []
    preds = torch.empty(0)
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            lgits = model(data)
        
        pred = torch.argmax(lgits, dim=1).cpu()
        
        preds = torch.cat((preds, pred), dim=0)
        labels += data.y.cpu().tolist()
    return labels, preds.tolist()