import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import warnings
import random
import torch
from pathlib import Path
from dataset_loader import get_dataloaders
from train_utils import train_epoch, eval_epoch
from models_lib import CNN1D, WDCNN, InceptionTime1D, VMD_CNN
from model import DFTSRNet
from config import Config
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pywt

warnings.filterwarnings("ignore")
device = Config.DEVICE

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_deep_for_curves(model, loaders, epochs=200):
    """运行深度学习模型获取训练曲线"""
    tr_loader, va_loader, te_loader = loaders
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val, best_state = 0, None
    
    acc_history = []
    loss_history = []
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, tr_loader, opt, device)
        _, val_acc = eval_epoch(model, va_loader, device)
        
        acc_history.append(val_acc)
        loss_history.append(train_loss)
        
        if val_acc > best_val:
            best_val, best_state = val_acc, model.state_dict()
    
    model.load_state_dict(best_state)
    _, test_acc = eval_epoch(model, te_loader, device)
    
    return test_acc, acc_history, loss_history

def train_and_save_curves():
    """
    训练五个模型并将训练曲线数据保存到CSV文件
    与compare.py的参数和逻辑对齐
    """
    
    # 设置随机种子（与compare.py一致）
    set_seed(42)
    
    # 模型列表（与compare.py一致）
    MODELS = {
        '1D-CNN': CNN1D,
        'WDCNN': WDCNN,
        'Inception': InceptionTime1D,
        'VMD+CNN': VMD_CNN,
        'DFTSRNet': lambda: DFTSRNet(num_classes=10)
    }
    
    # 数据集A路径
    BASE_PATH = Config.BASE_DATA_PATH
    dataset_a_path = os.path.join(BASE_PATH, 'Dataset_A')
    
    print("开始训练并收集数据集A的训练曲线...")
    
    # 收集训练曲线数据
    all_curves = {}
    loaders = get_dataloaders(dataset_a_path, batch_size=128)
    
    for name, ctor in MODELS.items():
        print(f"正在训练 {name}...")
        test_acc, acc_hist, loss_hist = run_deep_for_curves(ctor(), loaders, epochs=200)
        all_curves[name] = (acc_hist, loss_hist)
        print(f"  {name} 测试准确率: {test_acc:.4f}")
    
    # 准备保存数据
    epochs = list(range(1, 201))
    
    # 保存准确率数据
    acc_data = {'Epoch': epochs}
    for model_name, (acc_hist, _) in all_curves.items():
        acc_data[model_name] = acc_hist
    
    acc_df = pd.DataFrame(acc_data)
    
    # 保存损失数据
    loss_data = {'Epoch': epochs}
    for model_name, (_, loss_hist) in all_curves.items():
        # 确保损失数据是数值而不是元组
        if isinstance(loss_hist[0], tuple):
            loss_data[model_name] = [loss[0] if isinstance(loss, tuple) else loss for loss in loss_hist]
        else:
            loss_data[model_name] = loss_hist
    
    loss_df = pd.DataFrame(loss_data)
    
    # 创建输出目录
    OUTPUT_DIR = r'e:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\对比实验\对比试验结果\训练曲线'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存CSV文件
    acc_csv_path = os.path.join(OUTPUT_DIR, 'training_accuracy_DatasetA.csv')
    loss_csv_path = os.path.join(OUTPUT_DIR, 'training_loss_DatasetA.csv')
    
    acc_df.to_csv(acc_csv_path, index=False)
    loss_df.to_csv(loss_csv_path, index=False)
    
    print(f"准确率数据已保存到: {acc_csv_path}")
    print(f"损失数据已保存到: {loss_csv_path}")
    print("\n训练数据收集完成！")

if __name__ == "__main__":
    train_and_save_curves()