import numpy as np
import os

# 原始数据文件路径（请按实际路径修改）
data_path = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\数据集\西储大学实验数据集\Dataset_A\test_data_preprocessed.npy"
label_path = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\数据集\西储大学实验数据集\Dataset_A\test_labels.npy"

# 输出文件夹（自动创建）
out_dir = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\可解释性分析实验\data"
os.makedirs(out_dir, exist_ok=True)

# 每类采样数量
n_per_class = 1   # 你可自行修改

# 读取数据
X = np.load(data_path)        # shape: [N, 1024]
y = np.load(label_path)       # shape: [N,]

print("Original data shape:", X.shape)
print("Original label shape:", y.shape)
print("Label classes:", np.unique(y))

# 按类别采样
selected_idx = []
for cls in np.unique(y):
    idx = np.where(y == cls)[0]
    if len(idx) < n_per_class:
        print(f"Warning: Class {cls} only has {len(idx)} samples, less than {n_per_class}!")
        sel = idx
    else:
        sel = np.random.choice(idx, n_per_class, replace=False)
    selected_idx.extend(sel.tolist())

selected_idx = np.array(selected_idx)
X_out = X[selected_idx]
y_out = y[selected_idx]

print("Sampled data shape:", X_out.shape)
print("Sampled label shape:", y_out.shape)

# 保存结果
np.save(os.path.join(out_dir, "test_data.npy"), X_out)
np.save(os.path.join(out_dir, "test_labels.npy"), y_out)
print("Saved to:", os.path.join(out_dir, "test_data.npy"))

# 验证
for cls in np.unique(y_out):
    print(f"Class {cls}: {np.sum(y_out==cls)} samples")
