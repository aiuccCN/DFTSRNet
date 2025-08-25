# -*- coding: utf-8 -*-
"""
L1L2损失比率训练脚本
在Dataset_A上训练DFTSRNet、WDCNN、1D-CNN三个模型
使用早停，最大100个epoch
记录最后10个epoch的train/test loss并计算比值
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import json
import time
from collections import defaultdict

# 添加路径
BASE_DIR = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\新版可解释实验\20250808\基尼"
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))  # 添加上级目录以导入model.py

# 导入模型
from models_lib import WDCNN, CNN1D
from model import DFTSRNet

# 数据和输出路径
DATA_DIR = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\数据集\西储大学实验数据集\Dataset_A"
OUT_DIR = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\新版可解释实验\20250808\基尼\LIL2\结果"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[Info] Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据加载函数
def make_loader(split, batch_size=64, shuffle=True):
    """加载数据并添加通道维度"""
    x = np.load(os.path.join(DATA_DIR, f"{split}_data_preprocessed.npy"))  # (N,L)
    y = np.load(os.path.join(DATA_DIR, f"{split}_labels.npy"))             # (N,)
    
    # 添加通道维度: (N,L) -> (N,1,L)
    x = x[:, np.newaxis, :]
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

# 早停类
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

# 训练函数
def train_model(model, model_name, train_loader, val_loader, max_epochs=50, lr=0.001):
    """训练单个模型"""
    print(f"\n===== Training {model_name} =====")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(max_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            if model_name == 'DFTSRNet':
                outputs = model(batch_x)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # 取第一个输出作为logits
            else:
                outputs = model(batch_x)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                if model_name == 'DFTSRNet':
                    outputs = model(batch_x)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                else:
                    outputs = model(batch_x)
                
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        # 计算平均损失和准确率
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch < 5:
            print(f"Epoch [{epoch+1}/{max_epochs}] - "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 早停检查
        if early_stopping(avg_val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

# 计算最后10个epoch的损失比值
def calculate_l1_l2_ratio(history):
    """计算最后10个epoch的train/test损失比值"""
    train_losses = history['train_loss']
    val_losses = history['val_loss']
    
    # 取最后10个epoch
    last_10_train = train_losses[-10:] if len(train_losses) >= 10 else train_losses
    last_10_val = val_losses[-10:] if len(val_losses) >= 10 else val_losses
    
    # 计算平均值
    avg_train_loss = np.mean(last_10_train)
    avg_val_loss = np.mean(last_10_val)
    
    # 计算比值
    ratio = avg_train_loss / (avg_val_loss + 1e-12)
    
    return avg_train_loss, avg_val_loss, ratio

# 主函数
def main():
    print("Loading data...")
    train_loader = make_loader("train", batch_size=64, shuffle=True)
    val_loader = make_loader("val", batch_size=64, shuffle=False)
    test_loader = make_loader("test", batch_size=64, shuffle=False)
    
    # 模型配置
    models_config = {
        'DFTSRNet': lambda: DFTSRNet(in_channels=1, num_classes=10),
        'WDCNN': lambda: WDCNN(num_cls=10, in_channels=1),
        '1D-CNN': lambda: CNN1D(num_cls=10, in_channels=1)
    }
    
    results = []
    
    for model_name, model_builder in models_config.items():
        try:
            print(f"\n{'='*50}")
            print(f"Training {model_name}")
            print(f"{'='*50}")
            
            # 创建模型
            model = model_builder()
            
            # 训练模型
            start_time = time.time()
            trained_model, history = train_model(
                model, model_name, train_loader, val_loader, 
                max_epochs=100, lr=0.001
            )
            training_time = time.time() - start_time
            
            # 计算最后10个epoch的损失比值
            avg_train_loss, avg_val_loss, ratio = calculate_l1_l2_ratio(history)
            
            # 在测试集上评估
            trained_model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    
                    if model_name == 'DFTSRNet':
                        outputs = trained_model(batch_x)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                    else:
                        outputs = trained_model(batch_x)
                    
                    loss = nn.CrossEntropyLoss()(outputs, batch_y)
                    test_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += batch_y.size(0)
                    test_correct += (predicted == batch_y).sum().item()
            
            test_loss = test_loss / len(test_loader)
            test_acc = 100.0 * test_correct / test_total
            
            # 保存结果
            result = {
                'Model': model_name,
                'Training_Time': round(training_time, 2),
                'Total_Epochs': len(history['train_loss']),
                'Best_Val_Loss': round(min(history['val_loss']), 4),
                'Best_Val_Acc': round(max(history['val_acc']), 2),
                'Test_Loss': round(test_loss, 4),
                'Test_Acc': round(test_acc, 2),
                'Last10_Train_Loss': round(avg_train_loss, 4),
                'Last10_Val_Loss': round(avg_val_loss, 4),
                'L1_L2_Ratio': round(ratio, 4)
            }
            
            results.append(result)
            
            print(f"\n{model_name} Training Results:")
            print(f"  Training Time: {training_time:.2f}s")
            print(f"  Total Epochs: {len(history['train_loss'])}")
            print(f"  Best Val Loss: {min(history['val_loss']):.4f}")
            print(f"  Best Val Acc: {max(history['val_acc']):.2f}%")
            print(f"  Test Loss: {test_loss:.4f}")
            print(f"  Test Acc: {test_acc:.2f}%")
            print(f"  Last 10 Epochs - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"  L1/L2 Ratio: {ratio:.4f}")
            
            # 保存模型
            model_path = os.path.join(OUT_DIR, f"{model_name}_trained.pth")
            torch.save(trained_model.state_dict(), model_path)
            print(f"  Model saved to: {model_path}")
            
            # 保存训练历史
            history_path = os.path.join(OUT_DIR, f"{model_name}_history.json")
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            print(f"  History saved to: {history_path}")
            
        except Exception as e:
            print(f"[Error] Failed to train {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存汇总结果
    if results:
        # 保存为JSON
        results_path = os.path.join(OUT_DIR, "L1L2_training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # 保存为CSV
        import pandas as pd
        df = pd.DataFrame(results)
        csv_path = os.path.join(OUT_DIR, "L1L2_training_results.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"\n{'='*60}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'='*60}")
        print(df.to_string(index=False))
        print(f"\nResults saved to:")
        print(f"  JSON: {results_path}")
        print(f"  CSV: {csv_path}")
    else:
        print("[Warning] No successful training results to save.")

if __name__ == "__main__":
    main()