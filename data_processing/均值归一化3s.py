import numpy as np

def preprocess_signal(signal):
    """
    对单个TFR信号（1024点）进行预处理：
    1. 3σ准则处理异常值（用相邻点插值替换）
    2. 去均值
    3. 归一化到[-1, 1]
    """
    # 1. 3σ准则处理异常值
    mean = np.mean(signal)
    std = np.std(signal)
    upper_bound = mean + 3 * std  # 上界
    lower_bound = mean - 3 * std  # 下界
    
    # 找到异常值索引
    outliers = (signal < lower_bound) | (signal > upper_bound)
    
    # 对异常值进行相邻点插值替换（避免直接截断导致的突变）
    if np.any(outliers):
        # 遍历异常值，用前后非异常值的均值替换
        for i in np.where(outliers)[0]:
            # 找到前一个非异常值
            prev = i - 1
            while prev >= 0 and outliers[prev]:
                prev -= 1
            # 找到后一个非异常值
            next_idx = i + 1
            while next_idx < len(signal) and outliers[next_idx]:
                next_idx += 1
            
            # 替换异常值（若前后都有非异常值，用均值；否则用最近值）
            if prev >= 0 and next_idx < len(signal):
                signal[i] = (signal[prev] + signal[next_idx]) / 2
            elif prev >= 0:
                signal[i] = signal[prev]
            elif next_idx < len(signal):
                signal[i] = signal[next_idx]
    
    # 2. 去均值
    signal = signal - np.mean(signal)
    
    # 3. 归一化到[-1, 1]（除以最大绝对值）
    max_abs = np.max(np.abs(signal))
    if max_abs > 0:  # 避免除以0（理论上不会发生）
        signal = signal / max_abs
    
    return signal


def batch_preprocess(dataset_path):
    """
    批量预处理数据集（支持train/val/test数据）
    :param dataset_path: .npy文件路径（如train_data.npy）
    """
    # 加载数据（形状：[样本数, 1024]）
    data = np.load(dataset_path)
    print(f"开始预处理 {dataset_path}，样本数: {len(data)}")
    
    # 逐样本预处理
    preprocessed_data = np.zeros_like(data)
    for i in range(len(data)):
        preprocessed_data[i] = preprocess_signal(data[i])
    
    # 保存预处理后的数据（添加"_preprocessed"后缀）
    save_path = dataset_path.replace(".npy", "_preprocessed.npy")
    np.save(save_path, preprocessed_data)
    print(f"预处理完成，保存至: {save_path}")
    print(f"示例样本变化：\n原始：{data[0][:5]}\n处理后：{preprocessed_data[0][:5]}\n")


if __name__ == "__main__":
    # 批量处理所有数据集（可根据需要修改路径）
    dataset_dir = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\数据集\西储大学实验数据集"
    
    # 处理Dataset_A
    batch_preprocess(f"{dataset_dir}/Dataset_A/train_data.npy")
    batch_preprocess(f"{dataset_dir}/Dataset_A/val_data.npy")
    batch_preprocess(f"{dataset_dir}/Dataset_A/test_data.npy")
    
    # 处理Dataset_B
    batch_preprocess(f"{dataset_dir}/Dataset_B/train_data.npy")
    batch_preprocess(f"{dataset_dir}/Dataset_B/val_data.npy")
    batch_preprocess(f"{dataset_dir}/Dataset_B/test_data.npy")
    
    # 处理Dataset_C
    batch_preprocess(f"{dataset_dir}/Dataset_C/train_data.npy")
    batch_preprocess(f"{dataset_dir}/Dataset_C/val_data.npy")
    batch_preprocess(f"{dataset_dir}/Dataset_C/test_data.npy")
    
    print("所有数据集预处理完成！")