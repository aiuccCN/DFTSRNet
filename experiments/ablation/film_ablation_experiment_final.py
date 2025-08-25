import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import os
import sys  # 添加这行导入
from datetime import datetime
import random
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入必要的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config
from dataset import get_dataloaders
from focal_loss import FocalLoss
from ablation_models import AdaptiveFreqBandSelect, ModalGraphReasoning

# 设置随机种子
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 纯CNN模块（无FiLM调制）
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.resblock = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2)
        )
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # 权重初始化
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    
    def forward(self, fused_signal, graph_summary=None):
        # 忽略graph_summary，直接处理信号
        x = F.relu(self.bn1(self.conv1(fused_signal)))
        res = self.resblock(x)
        x = F.relu(x + res)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        out = self.classifier(x)
        return out

# FiLM调制CNN模块
class FiLMCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.film_gamma = nn.Linear(Config.GNN_HIDDEN, in_channels)
        self.film_beta = nn.Linear(Config.GNN_HIDDEN, in_channels)
        
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.resblock = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2)
        )
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # 权重初始化
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.film_gamma.weight)
        nn.init.xavier_uniform_(self.film_beta.weight)
    
    def forward(self, fused_signal, graph_summary):
        # FiLM调制
        gamma = self.film_gamma(graph_summary).unsqueeze(-1)  # [B, C, 1]
        beta = self.film_beta(graph_summary).unsqueeze(-1)
        modulated = fused_signal * (1 + gamma) + beta
        
        # CNN处理
        x = F.relu(self.bn1(self.conv1(modulated)))
        res = self.resblock(x)
        x = F.relu(x + res)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        out = self.classifier(x)
        return out

# 带FiLM的完整模型
class FullFiLMModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.afbs = AdaptiveFreqBandSelect(1, Config.NUM_MODALS, Config.SEQ_LEN)
        self.mgr = ModalGraphReasoning(1, Config.GNN_HIDDEN, Config.GNN_HIDDEN, 4)
        self.film_cnn = FiLMCNN(1, num_classes)
    
    def forward(self, x):
        fused, modals, _ = self.afbs(x)
        graph_feat, _ = self.mgr(modals)
        graph_summary = graph_feat.mean(dim=1)  # [B, hidden]
        out = self.film_cnn(fused, graph_summary)
        return out

# 无FiLM的模型（保持AFBS和MGR，但用纯CNN）
class NoFiLMModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.afbs = AdaptiveFreqBandSelect(1, Config.NUM_MODALS, Config.SEQ_LEN)
        self.mgr = ModalGraphReasoning(1, Config.GNN_HIDDEN, Config.GNN_HIDDEN, 4)
        self.simple_cnn = SimpleCNN(1, num_classes)
    
    def forward(self, x):
        fused, modals, _ = self.afbs(x)
        graph_feat, _ = self.mgr(modals)  # 计算但不使用
        # 不使用graph_summary进行FiLM调制
        out = self.simple_cnn(fused)
        return out

# 训练函数
def train_model(model, train_loader, val_loader, num_epochs, device, model_name):
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    train_losses = []
    val_accuracies = []
    
    model.to(device)
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}: Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return train_losses, val_accuracies

# 评估函数
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_preds)
    return accuracy, all_targets, all_preds

# 保存混淆矩阵
def save_confusion_matrix(y_true, y_pred, model_name, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = os.path.join(save_dir, 'confusion_matrices')
    os.makedirs(cm_path, exist_ok=True)
    plt.savefig(os.path.join(cm_path, f'confusion_matrix_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存文本格式
    np.savetxt(os.path.join(cm_path, f'confusion_matrix_{model_name}.txt'), cm, fmt='%d')

# 保存分类报告
def save_classification_report(y_true, y_pred, model_name, save_dir):
    report = classification_report(y_true, y_pred, digits=4)
    
    report_path = os.path.join(save_dir, 'classification_reports')
    os.makedirs(report_path, exist_ok=True)
    
    with open(os.path.join(report_path, f'classification_report_{model_name}.txt'), 'w') as f:
        f.write(f'Classification Report - {model_name}\n')
        f.write('=' * 50 + '\n')
        f.write(report)

# 保存训练曲线
def save_training_curves(results_dict, save_dir):
    curves_path = os.path.join(save_dir, 'training_curves')
    os.makedirs(curves_path, exist_ok=True)
    
    # 训练损失对比
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for model_name, data in results_dict.items():
        plt.plot(data['train_losses'], label=f'{model_name}')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 验证准确率对比
    plt.subplot(1, 2, 2)
    for model_name, data in results_dict.items():
        plt.plot(data['val_accuracies'], label=f'{model_name}')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(curves_path, 'training_curves_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 运行单个实验
def run_single_experiment(model_name, model_class, dataset_name, seed, save_dir):
    print(f"\n{'='*60}")
    print(f"Running {model_name} on {dataset_name} (seed: {seed})")
    print(f"{'='*60}")
    
    set_random_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据加载
    train_loader, val_loader, test_loader = get_dataloaders(
        Config.DATA_ROOT, dataset_name, Config.BATCH_SIZE
    )
    
    # 模型创建
    model = model_class(num_classes=Config.NUM_CLASSES)
    
    # 训练
    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader, Config.NUM_EPOCHS, device, model_name
    )
    
    # 测试评估
    test_accuracy, y_true, y_pred = evaluate_model(model, test_loader, device)
    
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    
    # 保存结果
    save_confusion_matrix(y_true, y_pred, f"{model_name}_{dataset_name}_seed{seed}", save_dir)
    save_classification_report(y_true, y_pred, f"{model_name}_{dataset_name}_seed{seed}", save_dir)
    
    # 保存模型
    model_save_path = os.path.join(save_dir, 'saved_models')
    os.makedirs(model_save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_save_path, f'{model_name}_{dataset_name}_seed{seed}.pth'))
    
    return {
        'model_name': model_name,
        'dataset': dataset_name,
        'seed': seed,
        'test_accuracy': test_accuracy,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies
    }

# 主实验函数
def main():
    # 实验配置
    models_to_test = {
        'Full_with_FiLM': FullFiLMModel,
        'w/o_FiLM': NoFiLMModel
    }
    
    datasets = ['A', 'B', 'C']
    seeds = [42, 123, 456]
    
    # 结果保存目录
    save_dir = r'e:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\消融实验\results\film_test_results'
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = []
    training_curves_data = {}
    
    # 运行所有实验
    for dataset_name in datasets:
        for seed in seeds:
            for model_name, model_class in models_to_test.items():
                result = run_single_experiment(
                    model_name, model_class, dataset_name, seed, save_dir
                )
                all_results.append(result)
                
                # 收集训练曲线数据
                key = f"{model_name}_{dataset_name}"
                if key not in training_curves_data:
                    training_curves_data[key] = {
                        'train_losses': [],
                        'val_accuracies': []
                    }
                training_curves_data[key]['train_losses'].append(result['train_losses'])
                training_curves_data[key]['val_accuracies'].append(result['val_accuracies'])
    
    # 计算平均训练曲线
    averaged_curves = {}
    for key, data in training_curves_data.items():
        averaged_curves[key] = {
            'train_losses': np.mean(data['train_losses'], axis=0),
            'val_accuracies': np.mean(data['val_accuracies'], axis=0)
        }
    
    # 保存训练曲线
    save_training_curves(averaged_curves, save_dir)
    
    # 创建详细结果表
    df_detailed = pd.DataFrame(all_results)
    df_detailed.to_csv(os.path.join(save_dir, 'detailed_results.csv'), index=False)
    
    # 创建汇总统计
    summary_stats = df_detailed.groupby(['model_name', 'dataset']).agg({
        'test_accuracy': ['mean', 'std', 'min', 'max']
    }).round(4)
    summary_stats.columns = ['mean_accuracy', 'std_accuracy', 'min_accuracy', 'max_accuracy']
    summary_stats.to_csv(os.path.join(save_dir, 'summary_statistics.csv'))
    
    # 创建消融结果对比表
    pivot_table = df_detailed.pivot_table(
        values='test_accuracy', 
        index='dataset', 
        columns='model_name', 
        aggfunc=['mean', 'std']
    ).round(4)
    pivot_table.to_csv(os.path.join(save_dir, 'ablation_results_table.csv'))
    
    print(f"\n{'='*60}")
    print("FiLM Ablation Experiment Completed!")
    print(f"Results saved to: {save_dir}")
    print(f"{'='*60}")
    
    # 打印汇总结果
    print("\nSummary Results:")
    print(summary_stats)
    
    # 计算FiLM模块的平均改进
    full_results = df_detailed[df_detailed['model_name'] == 'Full_with_FiLM']['test_accuracy']
    no_film_results = df_detailed[df_detailed['model_name'] == 'w/o_FiLM']['test_accuracy']
    
    improvement = full_results.mean() - no_film_results.mean()
    print(f"\nFiLM Module Average Improvement: {improvement:.4f} ({improvement*100:.2f}%)")

if __name__ == "__main__":
    main()