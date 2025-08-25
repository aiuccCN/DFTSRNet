import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from config import Config
from dataset import get_dataloaders
from focal_loss import FocalLoss
from ablation_models import create_ablation_model

def set_random_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    """训练模型并记录训练过程"""
    model.train()
    best_val_acc = 0.0
    
    # 记录训练过程
    train_losses = []
    train_accs = []
    val_accs = []
    
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
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        # 记录数据
        train_losses.append(epoch_train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {epoch_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
    
    return best_val_acc, train_losses, train_accs, val_accs

def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy, y_true, y_pred

def plot_training_curves(train_losses, train_accs, val_accs, model_name, seed, save_dir):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 训练损失曲线
    ax1.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Training Loss (Seed {seed})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 准确率曲线
    ax2.plot(train_accs, label='Train Accuracy', color='green', linewidth=2)
    ax2.plot(val_accs, label='Validation Accuracy', color='orange', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{model_name} - Accuracy Curves (Seed {seed})')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    safe_model_name = model_name.replace('/', '_')
    save_path = os.path.join(save_dir, f'training_curves_{safe_model_name}_seed{seed}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def plot_combined_curves(all_results, save_dir):
    """绘制相同模型不同种子的合并曲线"""
    models = list(set([r['model'] for r in all_results]))
    
    for model_name in models:
        model_results = [r for r in all_results if r['model'] == model_name]
        
        if len(model_results) == 0:
            continue
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 训练损失曲线
        for result in model_results:
            seed = result['seed']
            train_losses = result['train_losses']
            ax1.plot(train_losses, label=f'Seed {seed}', alpha=0.7, linewidth=2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{model_name} - Training Loss (All Seeds)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 验证准确率曲线
        for result in model_results:
            seed = result['seed']
            val_accs = result['val_accs']
            ax2.plot(val_accs, label=f'Seed {seed}', alpha=0.7, linewidth=2)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title(f'{model_name} - Validation Accuracy (All Seeds)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        safe_model_name = model_name.replace('/', '_')
        save_path = os.path.join(save_dir, f'combined_curves_{safe_model_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Combined curves saved: {save_path}")

def plot_confusion_matrix(y_true, y_pred, model_name, seed, save_dir):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(len(np.unique(y_true))),
                yticklabels=range(len(np.unique(y_true))))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{model_name} - Confusion Matrix (Seed {seed})')
    
    safe_model_name = model_name.replace('/', '_')
    save_path = os.path.join(save_dir, f'confusion_matrix_{safe_model_name}_seed{seed}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path, cm

def save_confusion_matrix_text(cm, model_name, seed, save_dir):
    """保存混淆矩阵的文字版本"""
    safe_model_name = model_name.replace('/', '_')
    save_path = os.path.join(save_dir, f'confusion_matrix_{safe_model_name}_seed{seed}.txt')
    
    with open(save_path, 'w') as f:
        f.write(f"Confusion Matrix for {model_name} (Seed {seed})\n")
        f.write("=" * 50 + "\n\n")
        f.write("Matrix Format:\n")
        f.write("Rows: True Labels\n")
        f.write("Columns: Predicted Labels\n\n")
        
        # 写入矩阵
        f.write("Confusion Matrix:\n")
        for i, row in enumerate(cm):
            f.write(f"Class {i}: {' '.join(f'{val:4d}' for val in row)}\n")
        
        # 计算每类的精确率、召回率
        f.write("\nPer-class Statistics:\n")
        f.write("-" * 30 + "\n")
        
        for i in range(len(cm)):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            f.write(f"Class {i}:\n")
            f.write(f"  Precision: {precision:.4f}\n")
            f.write(f"  Recall: {recall:.4f}\n")
            f.write(f"  F1-Score: {f1:.4f}\n")
            f.write(f"  Support: {cm[i, :].sum()}\n\n")
    
    return save_path

def run_single_experiment(model_name, dataset_name, seed):
    """运行单个实验"""
    print(f"\n=== Running {model_name} on Dataset {dataset_name} with seed {seed} ===")
    
    # 确保结果目录存在
    os.makedirs('results/classification_reports', exist_ok=True)
    os.makedirs('results/confusion_matrices', exist_ok=True)
    os.makedirs('results/training_curves', exist_ok=True)
    os.makedirs('results/saved_models', exist_ok=True)
    
    # 设置随机种子
    set_random_seed(seed)
    
    # 数据加载
    data_dir = os.path.join(Config.DATA_ROOT, f"Dataset_{dataset_name}")
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, Config.BATCH_SIZE)
    
    # 模型创建
    safe_model_name = model_name.replace('/', '_')
    model = create_ablation_model(model_name, Config.NUM_CLASSES).to(Config.DEVICE)
    
    # 损失函数和优化器
    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    # 训练
    best_val_acc, train_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, Config.DEVICE, Config.NUM_EPOCHS
    )
    
    # 测试
    test_acc, y_true, y_pred = evaluate_model(model, test_loader, Config.DEVICE)
    
    # 绘制训练曲线
    curve_path = plot_training_curves(train_losses, train_accs, val_accs, model_name, seed, 'results/training_curves')
    print(f"Training curves saved: {curve_path}")
    
    # 绘制混淆矩阵
    cm_path, cm = plot_confusion_matrix(y_true, y_pred, model_name, seed, 'results/confusion_matrices')
    print(f"Confusion matrix saved: {cm_path}")
    
    # 保存混淆矩阵文字版
    cm_text_path = save_confusion_matrix_text(cm, model_name, seed, 'results/confusion_matrices')
    print(f"Confusion matrix text saved: {cm_text_path}")
    
    # 保存分类报告
    report = classification_report(y_true, y_pred, digits=4)
    report_path = f'results/classification_reports/{safe_model_name}_{dataset_name}_seed{seed}.txt'
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    with open(report_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write("\n" + "="*50 + "\n")
        f.write(report)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Results saved to {report_path}")
    
    return {
        'model': model_name,
        'dataset': dataset_name, 
        'seed': seed,
        'test_accuracy': test_acc,
        'val_accuracy': best_val_acc,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'y_true': y_true,
        'y_pred': y_pred
    }

def main():
    """主函数"""
    print("Starting Ablation Experiments with Visualization...")
    print(f"Device: {Config.DEVICE}")
    print(f"Models: {Config.ABLATION_MODELS}")
    print(f"Datasets: {Config.DATASETS}")
    print(f"Seeds: {Config.RANDOM_SEEDS}")
    
    results = []
    
    # 运行所有实验
    for dataset in Config.DATASETS:
        for model_name in Config.ABLATION_MODELS:
            for seed in Config.RANDOM_SEEDS:
                try:
                    result = run_single_experiment(model_name, dataset, seed)
                    results.append(result)
                except Exception as e:
                    print(f"Error in {model_name}_{dataset}_seed{seed}: {str(e)}")
                    continue
    
    # 绘制合并曲线
    print("\n=== Generating Combined Curves ===")
    plot_combined_curves(results, 'results/training_curves')
    
    # 保存详细结果
    if len(results) > 0:
        # 准备DataFrame数据（排除不能序列化的字段）
        df_data = []
        for r in results:
            df_row = {
                'model': r['model'],
                'dataset': r['dataset'],
                'seed': r['seed'],
                'test_accuracy': r['test_accuracy'],
                'val_accuracy': r['val_accuracy'],
                'total_params': r['total_params'],
                'trainable_params': r['trainable_params']
            }
            df_data.append(df_row)
        
        df = pd.DataFrame(df_data)
        df.to_csv('results/detailed_results.csv', index=False)
        
        # 生成汇总统计
        summary_stats = df.groupby(['model', 'dataset']).agg({
            'test_accuracy': ['mean', 'std'],
            'val_accuracy': ['mean', 'std'],
            'total_params': 'first'
        }).round(4)
        
        summary_stats.to_csv('results/summary_statistics.csv')
        
        print("\n=== Experiment Summary ===")
        print(summary_stats)
        
        # 保存最终汇总表
        pivot_table = df.pivot_table(
            values='test_accuracy', 
            index='model', 
            columns=['dataset', 'seed'], 
            aggfunc='mean'
        ).round(4)
        
        pivot_table.to_csv('results/ablation_results_table.csv')
        print("\nDetailed results saved to results/ablation_results_table.csv")
    
    print("\n=== All Visualizations Generated ===")
    print("- Training curves: results/training_curves/")
    print("- Confusion matrices: results/confusion_matrices/")
    print("- Classification reports: results/classification_reports/")
    print("\nAblation experiments with visualization completed!")

if __name__ == "__main__":
    main()