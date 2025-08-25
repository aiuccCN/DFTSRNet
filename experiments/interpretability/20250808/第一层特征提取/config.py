import torch, os

class Config:
    DATA_ROOT = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\数据集\西储大学实验数据集"
    OUTPUT_DIR = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\results"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_DATA_PATH = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\数据集\西储大学实验数据集"

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
    NUM_EPOCHS = 200
    WARMUP_EPOCHS = 5
    WEIGHT_DECAY = 1e-4
    CLIP_NORM = 1.0

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
