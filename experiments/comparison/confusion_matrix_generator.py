# confusion_matrix_generator.py
# ----------------------------------------------------------
# 生成指定模型的混淆矩阵
# 支持模型：SVM+Wavelet、Inception、VMD+CNN、DFTSRNet
# ----------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import joblib
import pywt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
import seaborn as sns

from dataset_loader import get_dataloaders
from models_lib import InceptionTime1D, VMD_CNN
from model import DFTSRNet
from config import Config
from train_utils import eval_epoch

device = Config.DEVICE
BASE_PATH = Config.BASE_DATA_PATH
DATASETS = {d: os.path.join(BASE_PATH, f'Dataset_{d}') for d in ['A']}

# 输出目录
OUTPUT_DIR = r'e:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\对比实验\对比试验结果'
MODEL_DIR = os.path.join(OUTPUT_DIR, 'saved_models')
CONFUSION_DIR = os.path.join(OUTPUT_DIR, 'confusion_matrices')
os.makedirs(CONFUSION_DIR, exist_ok=True)

# 类别标签映射（根据你的数据集调整）
CLASS_NAMES = {
    'A': ['Normal', 'Ball_007', 'Ball_014', 'Ball_021', 'IR_007', 'IR_014', 'IR_021', 'OR_007', 'OR_014', 'OR_021'],
    'B': ['Normal', 'Ball_007', 'Ball_014', 'Ball_021', 'IR_007', 'IR_014', 'IR_021', 'OR_007', 'OR_014', 'OR_021'],
    'C': ['Normal', 'Ball_007', 'Ball_014', 'Ball_021', 'IR_007', 'IR_014', 'IR_021', 'OR_007', 'OR_014', 'OR_021']
}

# 指定要生成混淆矩阵的模型
TARGET_MODELS = ['SVM+Wavelet', 'Inception', 'VMD+CNN', 'DFTSRNet']

def wp_feat(sig, wav='db4', lvl=3):
    """小波包特征提取"""
    wp = pywt.WaveletPacket(sig, wav, 'symmetric', maxlevel=lvl)
    return np.array([np.sqrt(np.sum(n.data**2)) for n in wp.get_level(lvl, 'freq')])

def load_svm_model(model_path, dataset_root):
    """加载SVM模型并进行预测"""
    # 加载模型
    clf = joblib.load(model_path)
    
    # 加载测试数据
    te_x = np.load(os.path.join(dataset_root, 'test_data_preprocessed.npy'))
    te_y = np.load(os.path.join(dataset_root, 'test_labels.npy'))
    
    # 提取特征
    te_f = np.array([wp_feat(x) for x in te_x])
    
    # 预测
    y_pred = clf.predict(te_f)
    
    return te_y, y_pred

def load_deep_model_and_predict(model_path, model_class, dataset_root):
    """加载深度学习模型并进行预测"""
    # 创建模型实例
    if model_class == DFTSRNet:
        model = model_class(num_classes=10)
    else:
        model = model_class()
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 加载数据
    _, _, te_loader = get_dataloaders(dataset_root, batch_size=128)
    
    # 预测
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for data, labels in te_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    return np.array(y_true), np.array(y_pred)

def plot_confusion_matrix(y_true, y_pred, class_names, model_name, dataset_name, seed, save_path):
    """绘制并保存混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    # 使用优雅的蓝白渐变配色，专业且清晰
    sns.heatmap(cm, annot=True, fmt='d', 
                cmap='Blues',  # 经典蓝白渐变，专业清晰
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'shrink': 0.8},
                annot_kws={'size': 11, 'weight': 'bold'},  # 数字加粗更清晰
                linewidths=0.5, linecolor='white')  # 添加白色网格线
    
    # 美化坐标轴
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    # 设置更好的布局
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 计算准确率
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    
    return cm, accuracy

def generate_classification_report(y_true, y_pred, class_names, model_name, dataset_name, seed):
    """生成分类报告"""
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    
    # 保存报告
    report_path = os.path.join(CONFUSION_DIR, f'classification_report_{model_name}_{dataset_name}_seed{seed}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f'Classification Report - {model_name}\n')
        f.write(f'Dataset {dataset_name}, Seed {seed}\n')
        f.write('=' * 50 + '\n\n')
        f.write(report)
    
    return report_path

def main():
    """主函数"""
    print("开始生成混淆矩阵...")
    
    # 模型类映射
    model_classes = {
        'Inception': InceptionTime1D,
        'VMD+CNN': VMD_CNN,
        'DFTSRNet': DFTSRNet
    }
    
    results_summary = []
    
    for seed in [42, 97, 20]:
        print(f"\n处理种子 {seed}...")
        
        for dataset_name in ['A', 'B', 'C']:
            dataset_root = DATASETS[dataset_name]
            class_names = CLASS_NAMES[dataset_name]
            
            print(f"  数据集 {dataset_name}:")
            
            for model_name in TARGET_MODELS:
                print(f"    {model_name}...")
                
                try:
                    if model_name == 'SVM+Wavelet':
                        # SVM模型
                        model_path = os.path.join(MODEL_DIR, f'{model_name}_{dataset_name}_seed{seed}.pkl')
                        if not os.path.exists(model_path):
                            print(f"      警告: 模型文件不存在 {model_path}")
                            continue
                        
                        y_true, y_pred = load_svm_model(model_path, dataset_root)
                    
                    else:
                        # 深度学习模型
                        model_path = os.path.join(MODEL_DIR, f'{model_name}_{dataset_name}_seed{seed}.pth')
                        if not os.path.exists(model_path):
                            print(f"      警告: 模型文件不存在 {model_path}")
                            continue
                        
                        model_class = model_classes[model_name]
                        y_true, y_pred = load_deep_model_and_predict(model_path, model_class, dataset_root)
                    
                    # 生成混淆矩阵
                    cm_save_path = os.path.join(CONFUSION_DIR, f'confusion_matrix_{model_name}_{dataset_name}_seed{seed}.png')
                    cm, accuracy = plot_confusion_matrix(y_true, y_pred, class_names, 
                                                        model_name, dataset_name, seed, cm_save_path)
                    
                    # 生成分类报告
                    report_path = generate_classification_report(y_true, y_pred, class_names, 
                                                               model_name, dataset_name, seed)
                    
                    results_summary.append({
                        'model': model_name,
                        'dataset': dataset_name,
                        'seed': seed,
                        'accuracy': accuracy,
                        'confusion_matrix_path': cm_save_path,
                        'report_path': report_path
                    })
                    
                    print(f"      完成! 准确率: {accuracy:.4f}")
                    
                except Exception as e:
                    print(f"      错误: {str(e)}")
                    continue
    
    # 保存结果汇总
    summary_path = os.path.join(CONFUSION_DIR, 'confusion_matrix_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('混淆矩阵生成结果汇总\n')
        f.write('=' * 50 + '\n\n')
        
        for result in results_summary:
            f.write(f"模型: {result['model']}\n")
            f.write(f"数据集: {result['dataset']}\n")
            f.write(f"种子: {result['seed']}\n")
            f.write(f"准确率: {result['accuracy']:.4f}\n")
            f.write(f"混淆矩阵: {result['confusion_matrix_path']}\n")
            f.write(f"分类报告: {result['report_path']}\n")
            f.write('-' * 30 + '\n')
    
    print(f"\n全部完成!")
    print(f"混淆矩阵保存在: {CONFUSION_DIR}")
    print(f"结果汇总保存在: {summary_path}")
    print(f"共生成 {len(results_summary)} 个混淆矩阵")

if __name__ == '__main__':
    main()