# -*- coding: utf-8 -*-
"""
加载 CWRU Dataset_A / B / C 的 *.npy 文件并构造成 PyTorch Dataset
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class NpyBearingDataset(Dataset):
    def __init__(self, root, split):
        """
        root : Dataset_A 目录
        split: 'train' | 'val' | 'test'
        """
        self.data  = np.load(os.path.join(root, f"{split}_data_preprocessed.npy"))
        self.label = np.load(os.path.join(root, f"{split}_labels.npy"))
        # 转为 float32 / long
        self.data  = self.data.astype(np.float32)
        self.label = self.label.astype(np.int64)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # [1, 1024]
        return torch.from_numpy(self.data[idx][None, :]), self.label[idx]


def get_dataloaders(base_dir, batch_size=128):
    """返回 train / val / test 三个 DataLoader"""
    train_ds = NpyBearingDataset(base_dir, "train")
    val_ds   = NpyBearingDataset(base_dir, "val")
    test_ds  = NpyBearingDataset(base_dir, "test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader
