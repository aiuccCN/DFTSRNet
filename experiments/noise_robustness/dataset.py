import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data_dir, mode='train', use_preprocessed=True):
        suffix = '_preprocessed.npy' if use_preprocessed else '.npy'
        data_path = os.path.join(data_dir, f"{mode}_data{suffix}")
        label_path = os.path.join(data_dir, f"{mode}_labels.npy")

        self.data = torch.tensor(np.load(data_path), dtype=torch.float32).unsqueeze(1)  # [N, 1, L]
        self.labels = torch.tensor(np.load(label_path), dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_dataloaders(data_dir, batch_size=64, use_preprocessed=True):
    train_dataset = MyDataset(data_dir, mode='train', use_preprocessed=use_preprocessed)
    val_dataset = MyDataset(data_dir, mode='val', use_preprocessed=use_preprocessed)
    test_dataset = MyDataset(data_dir, mode='test', use_preprocessed=use_preprocessed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader