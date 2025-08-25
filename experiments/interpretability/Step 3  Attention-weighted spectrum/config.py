import torch, os

class Config:
    # 基础路径设置
    BASE_DIR = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\不同噪声水平实验"
    OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "results")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集配置
    DATASETS = [
        "Dataset_B_Full_Noisy_-2dB",
        "Dataset_B_Full_Noisy_-4dB", 
        "Dataset_B_Full_Noisy_0dB",
        "Dataset_B_Full_Noisy_2dB",
        "Dataset_B_SNR_-2dB",
        "Dataset_B_SNR_-4dB",
        "Dataset_B_SNR_0dB",
        "Dataset_B_SNR_2dB"
    ]

    # 数据 / 任务
    INPUT_CHANNELS = 1
    SEQ_LEN = 1024
    NUM_CLASSES = 10

    # 模型
    NUM_MODALS = 6  # AFBS 模态数
    GNN_HIDDEN = 32
    FILM_DIM = 32

    # 训练
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 100  # 减少到100个epoch以节省时间
    WARMUP_EPOCHS = 5
    WEIGHT_DECAY = 1e-4
    CLIP_NORM = 1.0

os.makedirs(Config.OUTPUT_BASE_DIR, exist_ok=True)