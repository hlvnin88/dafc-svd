"""DataLoader cho MMAF."""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class _DS(Dataset):
    def __init__(self, feats, labels):
        self.n = len(labels)
        self.y = torch.tensor(labels, dtype=torch.float32).view(-1,1)
        self.f = [torch.tensor(np.nan_to_num(a).astype(np.float32)) for a in feats]
    def __len__(self): return self.n
    def __getitem__(self, i): return [x[i] for x in self.f], self.y[i]

def make_loader(feats, labels, bs=32, shuffle=True):
    return DataLoader(_DS(feats, labels), batch_size=bs, shuffle=shuffle)