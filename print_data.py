import argparse
import numpy as np
import random
import pandas as pd
import sys, os

from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
import torch, torchvision
import torch_geometric
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator



def log2pandas(path):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
    return runlog_data



root = sys.argv[1]

out_tr = []
out_te = []
out_df = []

for seed in [0, 1, 2, 3, 4]:
    try:
        dirn = root+str(seed)
        files = os.listdir(dirn)
        print(files)
        df = log2pandas(os.path.join(dirn, files[-1]))

        loss = df.loc[df['metric'] == 'Loss/Train']['value'].to_numpy()
        train_acc = df.loc[df['metric'] == 'Acc/Train']['value'].to_numpy()
        test_acc = df.loc[df['metric'] == 'Acc/Test']['value'].to_numpy()
        
        i = np.argmin(-train_acc*100 + loss)
        out_tr.append(train_acc[i])
        out_te.append(test_acc[i])
        out_df.append(train_acc[i] - test_acc[i])
        print(train_acc[i], test_acc[i])
    except:
        pass


print("%1.3f %1.3f" % (np.mean(out_tr), np.std(out_tr)))
print("%1.3f %1.3f" % (np.mean(out_te), np.std(out_te)))
print("%1.3f %1.3f" % (np.mean(out_df), np.std(out_df)))

print(r"%.1f \tsmall{%.1f} & %.1f \tsmall{%.1f}" % (np.mean(out_tr)*100, np.std(out_tr)*100, np.mean(out_te)*100, np.std(out_te)*100))

print(r"%.1f \tsmall{%.1f} & %.1f \tsmall{%.1f}" % (np.mean(out_te)*100, np.std(out_te)*100, np.mean(out_df)*100, np.std(out_df)*100))