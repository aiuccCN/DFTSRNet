import numpy as np
import os
from pathlib import Path

def add_noise(signal, snr_db, seed=None):
    """
    向信号添加高斯白噪声
    
    Args:
        signal: 输入信号 (numpy array)
        snr_db: 信噪比 (dB)
        seed: 随机种子，确保可重复性
    
    Returns:
        带噪声的信号
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 计算信号功率
    signal_power = np.mean(signal ** 2)
    
    # 计算噪声功率
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # 生成噪声
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    
    # 添加噪声
    noisy_signal = signal + noise
    
    return noisy_signal

def generate_noisy_test_datasets():
    """
    为Dataset_B生成不同SNR级别的带噪声测试数据集
    """
    # 定义路径
    base_path = r"e:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\数据集\西储大学实验数据集\Dataset_B"
    output_path = r"e:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\不同噪声水平实验"
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # SNR级别
    snr_levels = [2, 0, -2, -4]
    
    # 加载原始测试数据和标签
    test_data = np.load(os.path.join(base_path, "test_data_preprocessed.npy"))
    test_labels = np.load(os.path.join(base_path, "test_labels.npy"))
    
    # 加载训练和验证数据（保持不变）
    train_data = np.load(os.path.join(base_path, "train_data_preprocessed.npy"))
    train_labels = np.load(os.path.join(base_path, "train_labels.npy"))
    val_data = np.load(os.path.join(base_path, "val_data_preprocessed.npy"))
    val_labels = np.load(os.path.join(base_path, "val_labels.npy"))
    
    print(f"原始测试数据形状: {test_data.shape}")
    print(f"原始测试标签形状: {test_labels.shape}")
    print(f"训练数据形状: {train_data.shape}")
    print(f"验证数据形状: {val_data.shape}")
    
    # 为每个SNR级别生成带噪声的数据集
    for snr in snr_levels:
        print(f"\n生成SNR {snr}dB的数据集...")
        
        # 创建SNR特定的目录
        snr_dir = os.path.join(output_path, f"Dataset_B_SNR_{snr}dB")
        os.makedirs(snr_dir, exist_ok=True)
        
        # 生成带噪声的测试数据
        noisy_test_data = np.zeros_like(test_data)
        
        for i in range(test_data.shape[0]):
            # 使用样本索引和SNR作为种子，确保可重复性
            seed = hash(f"sample_{i}_snr_{snr}") % (2**32)
            noisy_test_data[i] = add_noise(test_data[i], snr, seed=seed)
        
        # 保存带噪声的测试数据
        np.save(os.path.join(snr_dir, "test_data_preprocessed.npy"), noisy_test_data)
        np.save(os.path.join(snr_dir, "test_labels.npy"), test_labels)
        
        # 复制原始的训练和验证数据（不添加噪声）
        np.save(os.path.join(snr_dir, "train_data_preprocessed.npy"), train_data)
        np.save(os.path.join(snr_dir, "train_labels.npy"), train_labels)
        np.save(os.path.join(snr_dir, "val_data_preprocessed.npy"), val_data)
        np.save(os.path.join(snr_dir, "val_labels.npy"), val_labels)
        
        print(f"SNR {snr}dB数据集已保存到: {snr_dir}")
        
        # 验证噪声添加效果
        original_power = np.mean(test_data[0] ** 2)
        noisy_power = np.mean(noisy_test_data[0] ** 2)
        noise_power = np.mean((noisy_test_data[0] - test_data[0]) ** 2)
        actual_snr = 10 * np.log10(np.mean(test_data[0] ** 2) / noise_power)
        
        print(f"  原始信号功率: {original_power:.6f}")
        print(f"  带噪信号功率: {noisy_power:.6f}")
        print(f"  噪声功率: {noise_power:.6f}")
        print(f"  实际SNR: {actual_snr:.2f} dB (目标: {snr} dB)")
    
    # 创建数据集信息文件
    info_file = os.path.join(output_path, "dataset_info.txt")
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("Dataset_B 噪声数据集信息\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"原始数据路径: {base_path}\n")
        f.write(f"输出路径: {output_path}\n\n")
        f.write("数据集结构:\n")
        f.write(f"- 训练数据: {train_data.shape} (无噪声)\n")
        f.write(f"- 验证数据: {val_data.shape} (无噪声)\n")
        f.write(f"- 测试数据: {test_data.shape} (添加噪声)\n\n")
        f.write("SNR级别:\n")
        for snr in snr_levels:
            f.write(f"- {snr} dB\n")
        f.write("\n生成的数据集目录:\n")
        for snr in snr_levels:
            f.write(f"- Dataset_B_SNR_{snr}dB/\n")
            f.write(f"  ├── train_data_preprocessed.npy (无噪声)\n")
            f.write(f"  ├── train_labels.npy\n")
            f.write(f"  ├── val_data_preprocessed.npy (无噪声)\n")
            f.write(f"  ├── val_labels.npy\n")
            f.write(f"  ├── test_data_preprocessed.npy (SNR {snr}dB噪声)\n")
            f.write(f"  └── test_labels.npy\n")
    
    print(f"\n所有数据集生成完成！")
    print(f"数据集信息已保存到: {info_file}")
    print(f"\n生成的目录结构:")
    for snr in snr_levels:
        print(f"  Dataset_B_SNR_{snr}dB/")

if __name__ == "__main__":
    generate_noisy_test_datasets()