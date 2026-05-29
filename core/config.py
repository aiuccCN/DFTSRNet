import torch, os

# Resolve paths relative to this file so the repo is self-contained / portable.
_HERE = os.path.dirname(os.path.abspath(__file__))       # .../DFTSRNet/core
_REPO = os.path.dirname(_HERE)                            # .../DFTSRNet
_DATA = os.path.join(_REPO, "data", "dataset")           # bundled preprocessed npy (Dataset_A..F)

class Config:
    DATA_ROOT = os.environ.get("DFTSR_DATA_ROOT", _DATA)
    OUTPUT_DIR = os.environ.get("DFTSR_OUTPUT_DIR", os.path.join(_REPO, "results"))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_DATA_PATH = DATA_ROOT

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
