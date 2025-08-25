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

def get_dataloaders(data_root, datasets, batch_size=64, use_preprocessed=True):
    """获取数据加载器
    Args:
        data_root: 数据根目录
        datasets: 数据集列表，如['A']
        batch_size: 批次大小
        use_preprocessed: 是否使用预处理数据
    """
    # 使用第一个数据集
    dataset_name = datasets[0] if isinstance(datasets, list) else datasets
    data_dir = os.path.join(data_root, f"Dataset_{dataset_name}")
    
    train_dataset = MyDataset(data_dir, mode='train', use_preprocessed=use_preprocessed)
    val_dataset = MyDataset(data_dir, mode='val', use_preprocessed=use_preprocessed)
    test_dataset = MyDataset(data_dir, mode='test', use_preprocessed=use_preprocessed)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader