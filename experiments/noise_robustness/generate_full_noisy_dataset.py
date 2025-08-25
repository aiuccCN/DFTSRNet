import numpy as np
import os
from scipy import signal

def add_noise(data, snr_db, seed=None):
    """添加高斯白噪声"""
    if seed is not None:
        np.random.seed(seed)
    
    # 计算信号功率
    signal_power = np.mean(data**2)
    # 根据SNR计算噪声功率
    noise_power = signal_power / (10**(snr_db/10))
    # 生成高斯白噪声
    noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
    # 添加噪声
    noisy_data = data + noise
    return noisy_data

def create_full_noisy_dataset(original_dir, output_dir, snr_db, seed=42):
    """创建所有数据集都加噪声的版本"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置随机种子确保可重复性
    np.random.seed(seed)
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        # 加载原始数据
        data_path = os.path.join(original_dir, f'{split}_data_preprocessed.npy')
        label_path = os.path.join(original_dir, f'{split}_labels.npy')
        
        data = np.load(data_path)
        labels = np.load(label_path)
        
        # 为每个样本添加噪声（使用不同种子确保噪声独立）
        noisy_data = np.array([
            add_noise(sample, snr_db, seed=seed + i) 
            for i, sample in enumerate(data)
        ])
        
        # 保存带噪声的数据
        np.save(os.path.join(output_dir, f'{split}_data_preprocessed.npy'), noisy_data)
        np.save(os.path.join(output_dir, f'{split}_labels.npy'), labels)
        
        # 验证噪声效果
        original_power = np.mean(data**2)
        noise_power = np.mean((noisy_data - data)**2)
        actual_snr = 10 * np.log10(original_power / noise_power)
        
        print(f"{split}集 - 目标SNR: {snr_db}dB, 实际SNR: {actual_snr:.2f}dB")
    
    # 创建数据集信息文件
    info_content = f"""全噪声Dataset_B信息 (SNR: {snr_db}dB)
==================================================

原始数据路径: {original_dir}
输出路径: {output_dir}
目标SNR: {snr_db} dB
随机种子: {seed}

数据集结构:
- 训练数据: 添加{snr_db}dB噪声
- 验证数据: 添加{snr_db}dB噪声  
- 测试数据: 添加{snr_db}dB噪声

注意: 所有数据集都添加了相同水平的噪声
这种设置用于训练噪声鲁棒的模型
"""
    
    with open(os.path.join(output_dir, 'dataset_info.txt'), 'w', encoding='utf-8') as f:
        f.write(info_content)

def main():
    # 原始数据路径
    original_dir = r'e:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\数据集\西储大学实验数据集\Dataset_B'
    
    # SNR级别
    snr_levels = [2, 0, -2, -4]
    
    for snr in snr_levels:
        output_dir = rf'e:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\不同噪声水平实验\Dataset_B_Full_Noisy_{snr}dB'
        
        print(f"\n创建全噪声数据集 (SNR: {snr}dB)...")
        create_full_noisy_dataset(original_dir, output_dir, snr)
        print(f"完成: {output_dir}")

if __name__ == "__main__":
    main()