import torch
import os
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

class Config:
    # 数据路径
    DATA_ROOT = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\数据集\西储大学实验数据集"
    OUTPUT_DIR = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\消融实验\results"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据参数
    INPUT_CHANNELS = 1
    SEQ_LEN = 1024
    NUM_CLASSES = 10
    
    # 模型参数
    NUM_MODALS = 6  # AFBS模态数
    GNN_HIDDEN = 32
    FILM_DIM = 32
    
    # 训练参数
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50  # 消融实验用较少epoch
    WEIGHT_DECAY = 1e-4
    
    # 实验设置
    RANDOM_SEEDS = [42, 97, 20]
    DATASETS = ['A']  # 仅Dataset A
    
    # 消融模型列表
    ABLATION_MODELS = [
        'Base_CNN',
        '+AFBS', 
        '+MGR',
        '+FiLM',
        'Full',
        'w/o_AFBS',
        'w/o_MGR', 
        'w/o_FiLM'
    ]
    
    # 可视化设置
    PLOT_STYLE = 'seaborn-v0_8'
    FIGURE_DPI = 300
    FIGURE_FORMAT = 'png'

# 确保输出目录存在
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# 设置matplotlib样式
plt.style.use('default')  # 使用默认样式
plt.rcParams['figure.dpi'] = Config.FIGURE_DPI
plt.rcParams['savefig.format'] = Config.FIGURE_FORMAT
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3