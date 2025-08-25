import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# 检查原始数据形状
x = np.load(r'E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\数据集\西储大学实验数据集\Dataset_A\train_data_preprocessed.npy')
print('Original data shape:', x.shape)

# 转换为tensor
x_tensor = torch.from_numpy(x).float()
print('Tensor shape:', x_tensor.shape)

# 创建DataLoader
y = np.load(r'E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\数据集\西储大学实验数据集\Dataset_A\train_labels.npy')
y_tensor = torch.from_numpy(y).long()
ds = TensorDataset(x_tensor, y_tensor)
loader = DataLoader(ds, batch_size=2)

# 检查batch形状
for batch_x, batch_y in loader:
    print('Batch X shape:', batch_x.shape)
    print('Batch Y shape:', batch_y.shape)
    break