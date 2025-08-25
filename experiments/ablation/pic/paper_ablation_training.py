#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文消融实验专用训练脚本
训练+AFBS、w/o_AFBS、w/o_FiLM、Full四个模型
生成CSV数据和论文用图表
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import random
from datetime import datetime

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from dataset import get_dataloaders
from focal_loss import FocalLoss
from ablation_models import create_ablation_model

# 设置matplotlib字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300

def set_random_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, model_name, seed):
    """训练模型并记录训练过程"""
    model.train()
    best_val_acc = 0.0
    
    # 记录训练过程
    train_losses = []
    train_accs = []
    val_accs = []
    
    print(f"\nTraining {model_name} with seed {seed}...")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        # 计算准确率
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100. * train_correct / train_total
        epoch_val_acc = 100. * val_correct / val_total
        
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)
        
        # 更新最佳验证准确率
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {epoch_train_loss:.4f}, '
                  f'Train Acc: {epoch_train_acc:.2f}%, Val Acc: {epoch_val_acc:.2f}%')
    
    return best_val_acc, train_losses, train_accs, val_accs

def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()
    
    test_acc = 100. * test_correct / test_total
    return test_acc

def run_single_experiment(model_name, dataset_name, seed, save_dir):
    """运行单个实验"""
    print(f"\n=== Running {model_name} on Dataset {dataset_name} with seed {seed} ===")
    
    # 设置随机种子
    set_random_seed(seed)
    
    # 数据加载
    data_dir = os.path.join(Config.DATA_ROOT, f"Dataset_{dataset_name}")
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, Config.BATCH_SIZE)
    
    # 模型创建
    model = create_ablation_model(model_name, Config.NUM_CLASSES).to(Config.DEVICE)
    
    # 损失函数和优化器
    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    # 训练
    best_val_acc, train_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, Config.DEVICE, Config.NUM_EPOCHS, model_name, seed
    )
    
    # 测试
    test_acc = evaluate_model(model, test_loader, Config.DEVICE)
    
    print(f"Test Accuracy: {test_acc:.4f}%")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}%")
    
    return {
        'model': model_name,
        'dataset': dataset_name,
        'seed': seed,
        'test_accuracy': test_acc,
        'val_accuracy': best_val_acc,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

def save_training_data_to_csv(all_results, save_dir):
    """保存训练数据到CSV文件"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 为每个模型创建CSV文件
    for result in all_results:
        model_name = result['model']
        seed = result['seed']
        
        # 创建DataFrame
        df = pd.DataFrame({
            'epoch': range(1, len(result['train_losses']) + 1),
            'train_loss': result['train_losses'],
            'train_accuracy': result['train_accs'],
            'val_accuracy': result['val_accs']
        })
        
        # 保存CSV
        safe_model_name = model_name.replace('/', '_').replace('+', 'plus_')
        csv_filename = f'{safe_model_name}_seed{seed}_training_data.csv'
        csv_path = os.path.join(save_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"Training data saved: {csv_path}")
    
    # 创建汇总CSV
    summary_data = []
    for result in all_results:
        summary_data.append({
            'model': result['model'],
            'seed': result['seed'],
            'test_accuracy': result['test_accuracy'],
            'best_val_accuracy': result['val_accuracy']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(save_dir, 'training_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Training summary saved: {summary_path}")

def create_paper_figures(all_results, save_dir):
    """创建论文用图表"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 按模型分组
    model_groups = {}
    for result in all_results:
        model_name = result['model']
        if model_name not in model_groups:
            model_groups[model_name] = []
        model_groups[model_name].append(result)
    
    # 图1: +AFBS vs w/o_AFBS (4个子图)
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 8))
    fig1.suptitle('AFBS Ablation Study', fontsize=14)
    
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']
    metrics = [('train_losses', 'Training Loss'), ('val_accs', 'Validation Accuracy (%)'), 
               ('train_accs', 'Training Accuracy (%)'), ('test_accuracy', 'Test Accuracy (%)')]
    
    for idx, (metric, ylabel) in enumerate(metrics):
        row, col = idx // 2, idx % 2
        ax = axes1[row, col]
        
        # 绘制+AFBS
        if '+AFBS' in model_groups:
            for result in model_groups['+AFBS']:
                if metric == 'test_accuracy':
                    # 测试准确率显示为水平线
                    ax.axhline(y=result[metric], label=f'+AFBS (seed {result["seed"]})', 
                              linestyle='-', alpha=0.7)
                else:
                    ax.plot(result[metric], label=f'+AFBS (seed {result["seed"]})', 
                           linestyle='-', alpha=0.7)
        
        # 绘制w/o_AFBS
        if 'w/o_AFBS' in model_groups:
            for result in model_groups['w/o_AFBS']:
                if metric == 'test_accuracy':
                    ax.axhline(y=result[metric], label=f'w/o AFBS (seed {result["seed"]})', 
                              linestyle='--', alpha=0.7)
                else:
                    ax.plot(result[metric], label=f'w/o AFBS (seed {result["seed"]})', 
                           linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{subplot_labels[idx]} {ylabel}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    fig1_path = os.path.join(save_dir, 'afbs_ablation_comparison.png')
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"AFBS ablation figure saved: {fig1_path}")
    
    # 图2: w/o_FiLM vs Full (4个子图)
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    fig2.suptitle('FiLM Ablation Study', fontsize=14)
    
    for idx, (metric, ylabel) in enumerate(metrics):
        row, col = idx // 2, idx % 2
        ax = axes2[row, col]
        
        # 绘制w/o_FiLM
        if 'w/o_FiLM' in model_groups:
            for result in model_groups['w/o_FiLM']:
                if metric == 'test_accuracy':
                    ax.axhline(y=result[metric], label=f'w/o FiLM (seed {result["seed"]})', 
                              linestyle='--', alpha=0.7)
                else:
                    ax.plot(result[metric], label=f'w/o FiLM (seed {result["seed"]})', 
                           linestyle='--', alpha=0.7)
        
        # 绘制Full
        if 'Full' in model_groups:
            for result in model_groups['Full']:
                if metric == 'test_accuracy':
                    ax.axhline(y=result[metric], label=f'Full (seed {result["seed"]})', 
                              linestyle='-', alpha=0.7)
                else:
                    ax.plot(result[metric], label=f'Full (seed {result["seed"]})', 
                           linestyle='-', alpha=0.7)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{subplot_labels[idx]} {ylabel}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    fig2_path = os.path.join(save_dir, 'film_ablation_comparison.png')
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"FiLM ablation figure saved: {fig2_path}")

def main():
    """主函数"""
    print("Starting Paper Ablation Experiments...")
    print(f"Device: {Config.DEVICE}")
    
    # 目标模型
    target_models = ['+AFBS', 'w/o_AFBS', 'w/o_FiLM', 'Full']
    datasets = ['A']  # 仅Dataset A
    seeds = [42, 97, 20]  # 使用Config中的种子
    
    # 输出目录
    save_dir = r'e:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\消融实验\论文用'
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = []
    
    # 运行所有实验
    for dataset in datasets:
        for model_name in target_models:
            for seed in seeds:
                try:
                    result = run_single_experiment(model_name, dataset, seed, save_dir)
                    all_results.append(result)
                except Exception as e:
                    print(f"Error in {model_name}_{dataset}_seed{seed}: {str(e)}")
                    continue
    
    # 保存训练数据到CSV
    print("\n=== Saving Training Data to CSV ===")
    save_training_data_to_csv(all_results, save_dir)
    
    # 创建论文用图表
    print("\n=== Creating Paper Figures ===")
    create_paper_figures(all_results, save_dir)
    
    print("\n=== Paper Ablation Experiments Completed! ===")
    print(f"Results saved to: {save_dir}")
    print("Generated files:")
    print("- Training data CSV files for each model and seed")
    print("- training_summary.csv: Summary of all experiments")
    print("- afbs_ablation_comparison.png: AFBS ablation study figure")
    print("- film_ablation_comparison.png: FiLM ablation study figure")

if __name__ == "__main__":
    main()