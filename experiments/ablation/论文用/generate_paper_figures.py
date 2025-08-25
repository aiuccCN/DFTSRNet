import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob

# 设置matplotlib参数
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

plt.rcParams['figure.dpi'] = 300

# 定义专业配色方案
# 方案1: 学术蓝橙配色（推荐）
COLOR_SCHEME_1 = {
    'colors': ['#1f77b4', '#ff7f0e', '#2ca02c'],  # 蓝、橙、绿
    'name': 'Academic Blue-Orange'
}

# 方案2: 深色专业配色
COLOR_SCHEME_2 = {
    'colors': ['#2E4057', '#048A81', '#F18F01'],
    'name': '深色专业配色',
    'description': '深沉专业的配色方案，适合技术类论文'
}

# 方案3: 科研期刊常用配色
COLOR_SCHEME_3 = {
    'colors': ['#003f5c', '#58508d', '#bc5090'],  # 深蓝、紫、粉
    'name': 'Journal Standard'
}

# 当前使用的配色方案
CURRENT_SCHEME = COLOR_SCHEME_2

def load_training_data():
    """加载所有训练数据CSV文件"""
    data = {}
    csv_files = glob('*_training_data.csv')
    
    for file in csv_files:
        # 从文件名提取模型名和种子
        basename = os.path.splitext(file)[0]
        parts = basename.split('_seed')
        model_name = parts[0].replace('_', '/')
        seed_part = parts[1].split('_')[0]  # 只取种子数字部分
        seed = int(seed_part)
        
        if model_name not in data:
            data[model_name] = {}
        
        df = pd.read_csv(file)
        data[model_name][seed] = df
    
    return data

def create_paper_figures(data):
    """创建论文图表"""
    
    # 创建输出目录
    output_dir = '论文消融图2'
    os.makedirs(output_dir, exist_ok=True)
    
    # 图1: +AFBS vs w/o AFBS
    fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8))
    fig1.suptitle('', fontsize=12)  # 无标题
    
    models_group1 = ['plus/AFBS', 'w/o/AFBS']
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']
    
    subplot_idx = 0
    for model in models_group1:
        if model in data:
            # 训练损失子图
            row = subplot_idx // 2
            col = subplot_idx % 2
            ax = axes1[row, col]
            
            for i, seed in enumerate(data[model]):
                df = data[model][seed]
                color = CURRENT_SCHEME['colors'][i % len(CURRENT_SCHEME['colors'])]
                ax.plot(df['epoch'], df['train_loss'], alpha=0.8, label=f'Seed {seed}', 
                       color=color, linewidth=1.5)
            
            ax.set_xlabel('Epoch', fontweight='normal')
            ax.set_ylabel('Training Loss', fontweight='normal')
            ax.text(-0.1, 1.05, subplot_labels[subplot_idx], transform=ax.transAxes, fontsize=12, 
                   fontweight='normal', va='bottom', ha='left')
            ax.legend(frameon=False)
            ax.grid(True, alpha=0.3)
            subplot_idx += 1
            
            # 验证准确率子图
            row = subplot_idx // 2
            col = subplot_idx % 2
            ax = axes1[row, col]
            
            for i, seed in enumerate(data[model]):
                df = data[model][seed]
                color = CURRENT_SCHEME['colors'][i % len(CURRENT_SCHEME['colors'])]
                ax.plot(df['epoch'], df['val_accuracy'], alpha=0.8, label=f'Seed {seed}', 
                       color=color, linewidth=1.5)
            
            ax.set_xlabel('Epoch', fontweight='normal')
            ax.set_ylabel('Validation Accuracy (%)', fontweight='normal')
            ax.text(-0.1, 1.05, subplot_labels[subplot_idx], transform=ax.transAxes, fontsize=12, 
                   fontweight='normal', va='bottom', ha='left')
            ax.legend(frameon=False)
            ax.grid(True, alpha=0.3)
            subplot_idx += 1
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'AFBS_comparison_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'AFBS_comparison_training_curves.svg'), bbox_inches='tight')
    plt.close()
    
    # 图2: w/o FiLM vs Full
    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
    fig2.suptitle('', fontsize=12)  # 无标题
    
    models_group2 = ['w/o/FiLM', 'Full']
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']  # 重新定义subplot_labels
    
    subplot_idx = 0
    for model in models_group2:
        if model in data:
            # 训练损失子图
            row = subplot_idx // 2
            col = subplot_idx % 2
            ax = axes2[row, col]
            
            for i, seed in enumerate(data[model]):
                df = data[model][seed]
                color = CURRENT_SCHEME['colors'][i % len(CURRENT_SCHEME['colors'])]
                ax.plot(df['epoch'], df['train_loss'], alpha=0.8, label=f'Seed {seed}', 
                       color=color, linewidth=1.5)
            
            ax.set_xlabel('Epoch', fontweight='normal')
            ax.set_ylabel('Training Loss', fontweight='normal')
            ax.text(-0.1, 1.05, subplot_labels[subplot_idx], transform=ax.transAxes, fontsize=12, 
                   fontweight='normal', va='bottom', ha='left')
            ax.legend(frameon=False)
            ax.grid(True, alpha=0.3)
            subplot_idx += 1
            
            # 验证准确率子图
            row = subplot_idx // 2
            col = subplot_idx % 2
            ax = axes2[row, col]
            
            for i, seed in enumerate(data[model]):
                df = data[model][seed]
                color = CURRENT_SCHEME['colors'][i % len(CURRENT_SCHEME['colors'])]
                ax.plot(df['epoch'], df['val_accuracy'], alpha=0.8, label=f'Seed {seed}', 
                       color=color, linewidth=1.5)
            
            ax.set_xlabel('Epoch', fontweight='normal')
            ax.set_ylabel('Validation Accuracy (%)', fontweight='normal')
            ax.text(-0.1, 1.05, subplot_labels[subplot_idx], transform=ax.transAxes, fontsize=12, 
                    fontweight='normal', va='bottom', ha='left')
            ax.legend(frameon=False)
            ax.grid(True, alpha=0.3)
            subplot_idx += 1
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'FiLM_Full_comparison_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'FiLM_Full_comparison_training_curves.svg'), bbox_inches='tight')
    plt.close()
    
    print("Paper figures created successfully!")
    print(f"Generated files in '{output_dir}' directory:")
    print("- AFBS_comparison_training_curves.png")
    print("- AFBS_comparison_training_curves.svg")
    print("- FiLM_Full_comparison_training_curves.png")
    print("- FiLM_Full_comparison_training_curves.svg")

if __name__ == "__main__":
    print("Loading training data...")
    data = load_training_data()
    print(f"Loaded data for models: {list(data.keys())}")
    
    print("Creating paper figures...")
    create_paper_figures(data)