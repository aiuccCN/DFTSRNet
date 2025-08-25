import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# 设置matplotlib参数（参考对比实验脚本）
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.unicode_minus'] = False

def create_afbs_confusion_matrix():
    """
    创建AFBS混淆矩阵（根据用户提供的第一个图片数据）
    """
    cm = np.array([
        [25, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 5, 8, 12, 0, 0, 0, 0, 0, 0],
        [0, 0, 25, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 25, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 25, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 25, 0, 0, 0, 0],
        [0, 0, 25, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 4, 0, 21, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 25, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 25]
    ])
    return cm

def create_mgr_confusion_matrix():
    """
    创建MGR混淆矩阵（根据用户提供的第二个图片数据）
    """
    cm = np.array([
        [25, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 25, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 24, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 25, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 25, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 25, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 25, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 25, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 25, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 25]
    ])
    return cm

def create_wo_mgr_confusion_matrix():
    """
    创建W/o MGR混淆矩阵（根据用户提供的第三个图片数据）
    """
    cm = np.array([
        [25, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 9, 16, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 25, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 25, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 25, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 25, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 25, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 25, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 25, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 25]
    ])
    return cm



def plot_single_confusion_matrix(cm, title, save_path):
    """
    绘制单个混淆矩阵图（参考对比实验中单个子图的尺寸）
    """
    # 计算单个子图的尺寸：对比实验总尺寸19cm×18cm，2×2布局
    # 单个子图约为 (19/2.54)/2 × (18/2.54)/2 英寸
    single_width = (19/2.54) / 2
    single_height = (18/2.54) / 2
    fig, ax = plt.subplots(figsize=(single_width, single_height))
    
    # 类别标签
    class_labels = [str(i) for i in range(10)]
    
    # 归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 使用与对比实验完全相同的绘图风格
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                vmin=0, vmax=1,
                xticklabels=class_labels, yticklabels=class_labels,
                cbar=False, ax=ax, square=True, annot_kws={'size': 10})
    
    # 设置坐标轴标签（与对比实验一致，10号字体不加粗）
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)
    
    # 设置刻度标签字体
    ax.tick_params(axis='both', labelsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    print(f"单图混淆矩阵已保存到: {save_path}")
    plt.close()

def plot_combined_confusion_matrices(cm1, cm2, title1, title2, save_path):
    """
    绘制左右合并的混淆矩阵图（参考对比实验的布局）
    """
    # 计算合并图的尺寸：两个单图水平拼接
    single_width = (19/2.54) / 2
    single_height = (18/2.54) / 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(single_width * 2, single_height))
    
    # 类别标签
    class_labels = [str(i) for i in range(10)]
    
    # 归一化混淆矩阵
    cm1_normalized = cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]
    cm2_normalized = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]
    
    # 绘制第一个混淆矩阵 (a) - 使用与对比实验完全相同的设置
    sns.heatmap(cm1_normalized, annot=True, fmt='.2f', cmap='Blues',
                vmin=0, vmax=1,
                xticklabels=class_labels, yticklabels=class_labels,
                cbar=False, ax=ax1, square=True, annot_kws={'size': 10})
    
    ax1.set_xlabel('Predicted Label', fontsize=10)
    ax1.set_ylabel('True Label', fontsize=10)
    ax1.tick_params(axis='both', labelsize=10)
    
    # 添加子图标注 (a)（参考对比实验的位置）
    ax1.text(-0.1, 1.05, '(a)', fontsize=10, color='black', 
             transform=ax1.transAxes, ha='left', va='bottom')
    
    # 绘制第二个混淆矩阵 (b) - 使用与对比实验完全相同的设置
    sns.heatmap(cm2_normalized, annot=True, fmt='.2f', cmap='Blues',
                vmin=0, vmax=1,
                xticklabels=class_labels, yticklabels=class_labels,
                cbar=False, ax=ax2, square=True, annot_kws={'size': 10})
    
    ax2.set_xlabel('Predicted Label', fontsize=10)
    ax2.set_ylabel('True Label', fontsize=10)
    ax2.tick_params(axis='both', labelsize=10)
    
    # 添加子图标注 (b)（参考对比实验的位置）
    ax2.text(-0.1, 1.05, '(b)', fontsize=10, color='black', 
             transform=ax2.transAxes, ha='left', va='bottom')
    
    # 调整布局（参考对比实验的间距设置）
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.12, hspace=0.3, left=0.08, right=0.95, top=0.92, bottom=0.08)
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    print(f"合并混淆矩阵已保存到: {save_path}")
    plt.close()

def main():
    # 输出路径
    base_path = r'e:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\消融实验\results\film_test_results\confusion_matrices'
    output_path = os.path.join(base_path, "论文配图", "结果")
    os.makedirs(output_path, exist_ok=True)
    
    # 直接使用硬编码的混淆矩阵数据
    confusion_matrices = {
        'AFBS': create_afbs_confusion_matrix(),
        'MGR': create_mgr_confusion_matrix(),
        'W/o_MGR': create_wo_mgr_confusion_matrix()
    }
    
    print("使用硬编码的混淆矩阵数据")
    for name in confusion_matrices.keys():
        print(f"成功加载 {name} 的混淆矩阵")
    
    # 生成第一张图：单图 AFBS
    if 'AFBS' in confusion_matrices:
        single_save_path = os.path.join(output_path, 'AFBS_confusion_matrix_seed42.png')
        plot_single_confusion_matrix(
            confusion_matrices['AFBS'],
            'AFBS - Confusion Matrix (Seed 42)',
            single_save_path
        )
    
    # 生成第二张图：左右合并 MGR vs W/o MGR
    if 'MGR' in confusion_matrices and 'W/o_MGR' in confusion_matrices:
        combined_save_path = os.path.join(output_path, 'MGR_comparison_confusion_matrix_seed42.png')
        plot_combined_confusion_matrices(
            confusion_matrices['W/o_MGR'],  # 左图 (a)
            confusion_matrices['MGR'],      # 右图 (b)
            'W/o MGR - Confusion Matrix (Seed 42)',
            'MGR - Confusion Matrix (Seed 42)',
            combined_save_path
        )

if __name__ == '__main__':
    main()