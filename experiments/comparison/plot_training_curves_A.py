import matplotlib.pyplot as plt
import numpy as np
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

def wp_feat(sig, wav='db4', lvl=3):
    wp = pywt.WaveletPacket(sig, wav, 'symmetric', maxlevel=lvl)
    return np.array([np.sqrt(np.sum(n.data**2)) for n in wp.get_level(lvl,'freq')])

def run_svm_for_curves(root):
    """运行SVM获取准确率，用于训练曲线"""
    tr_x = np.load(os.path.join(root,'train_data_preprocessed.npy'))
    va_x = np.load(os.path.join(root,'val_data_preprocessed.npy'))
    te_x = np.load(os.path.join(root,'test_data_preprocessed.npy'))
    tr_y = np.load(os.path.join(root,'train_labels.npy'))
    va_y = np.load(os.path.join(root,'val_labels.npy'))
    te_y = np.load(os.path.join(root,'test_labels.npy'))
    tr_f = np.array([wp_feat(x) for x in tr_x])
    va_f = np.array([wp_feat(x) for x in va_x])
    te_f = np.array([wp_feat(x) for x in te_x])
    best_c = max([0.1,1,10], key=lambda c:accuracy_score(va_y,SVC(C=c,gamma='auto').fit(tr_f,tr_y).predict(va_f)))
    clf = SVC(C=best_c,gamma='auto').fit(np.vstack([tr_f,va_f]), np.hstack([tr_y,va_y]))
    return accuracy_score(te_y, clf.predict(te_f))

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

def plot_training_curves():
    """
    绘制五个模型在数据集A上的训练曲线和损失函数
    使用2×1子图布局：上图显示准确率曲线，下图显示损失曲线
    与compare.py的参数和逻辑对齐
    """
    
    # 设置随机种子（与compare.py一致）
    set_seed(42)
    
    # 设置matplotlib参数
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = 12
    
    # 定义模型颜色（与compare.py完全一致）
    palette = {
        '1D-CNN': (33/255, 158/255, 188/255),
        'WDCNN': (19/255, 103/255, 131/255),
        'Inception': (2/255, 48/255, 74/255),
        'VMD+CNN': (254/255, 183/255, 5/255),
        'DFTSRNet': 'red'
    }
    
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
    
    # 提取准确率和损失数据
    acc_plot = {m: curves[0] for m, curves in all_curves.items()}
    loss_plot = {m: curves[1] for m, curves in all_curves.items()}
    
    # 创建2x1子图布局：上图显示准确率曲线，下图显示损失曲线
    xs = np.arange(1, 201)
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 绘制准确率曲线
    for m, c in acc_plot.items():
        style = {'lw': 4, 'color': palette[m]} if m == 'DFTSRNet' else {'lw': 1.5, 'color': palette[m]}
        ax[0].plot(xs, c, label=m, **style)
    
    ax[0].set_ylabel('Accuracy', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
    ax[0].legend(loc='lower right', fontsize=12, prop={'family': 'Times New Roman', 'weight': 'bold'})
    ax[0].grid(True, alpha=0.3)
    ax[0].tick_params(axis='both', which='major', labelsize=14)
    
    # 绘制损失曲线
    for m, l in loss_plot.items():
        style = {'lw': 4, 'color': palette[m]} if m == 'DFTSRNet' else {'lw': 1.5, 'color': palette[m]}
        ax[1].plot(xs, l, **style)
    
    ax[1].set_xlabel('Epoch', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
    ax[1].set_ylabel('Loss', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
    ax[1].legend(loc='upper right', fontsize=12, prop={'family': 'Times New Roman', 'weight': 'bold'})
    ax[1].grid(True, alpha=0.3)
    ax[1].tick_params(axis='both', which='major', labelsize=14)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片（与compare.py一致的路径和参数）
    OUTPUT_DIR = r'e:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\对比实验\对比试验结果'
    curve_path = os.path.join(OUTPUT_DIR, '训练曲线', 'training_curves_DatasetA.png')
    os.makedirs(os.path.dirname(curve_path), exist_ok=True)
    
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"训练曲线图已保存到: {curve_path}")
    print("\n训练曲线绘制完成！")

if __name__ == "__main__":
    plot_training_curves()