#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正版可解释性分析脚本 - 完全复刻VKCNN第五章
生成图14（特征映射热图）、表6（基尼指数与损失）、图15（注意力权重谱）、图16（Infogram分析）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import signal
from scipy.stats import entropy
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, hilbert, welch, get_window, sosfiltfilt
import math
import os
import sys
from pathlib import Path
import random
from torch.utils.data import DataLoader, TensorDataset
import json
import warnings
warnings.filterwarnings('ignore')

# 添加模型路径
sys.path.append(r'e:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\新版可解释实验')
from model import DFTSRNet
from models_lib import WDCNN, CNN1D

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# VKCNN风格配色方案
COLOR_SCHEME = {
    'light_blue': '#90C9E7',
    'cyan': '#219EBC', 
    'deep_cyan': '#136783',
    'navy': '#02304A',
    'gold': '#FEB705',
    'orange': '#F59E00'
}

class VKCNNExplainabilityAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        self.set_random_seed(42)
        
        # 创建输出目录
        self.output_dir = Path('结果')
        self.output_dir.mkdir(exist_ok=True)
        
        # 数据路径
        self.data_dir = Path('../data')
        self.models_dir = Path('../models')
        
    def set_random_seed(self, seed=42):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def load_data(self):
        """加载数据集"""
        print("\n加载数据集...")
        
        # 加载现有的测试数据
        test_data = np.load(self.data_dir / 'test_data.npy')
        test_labels = np.load(self.data_dir / 'test_labels.npy')
        
        # 添加通道维度以匹配模型输入要求 (N, L) -> (N, 1, L)
        if len(test_data.shape) == 2:
            test_data = np.expand_dims(test_data, axis=1)
        
        # 将测试数据分割为训练、验证和测试集
        n_samples = len(test_data)
        indices = np.random.permutation(n_samples)
        
        train_size = int(0.6 * n_samples)
        val_size = int(0.2 * n_samples)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_data = test_data[train_indices]
        train_labels = test_labels[train_indices]
        val_data = test_data[val_indices]
        val_labels = test_labels[val_indices]
        test_data = test_data[test_indices]
        test_labels = test_labels[test_indices]
        
        print(f"训练集: {train_data.shape}, 验证集: {val_data.shape}, 测试集: {test_data.shape}")
        
        return train_data, train_labels, val_data, val_labels, test_data, test_labels
    
    def create_model(self, model_name):
        """创建模型"""
        if model_name == 'DFTSRNet':
            model = DFTSRNet(num_classes=10)
        elif model_name == 'WDCNN':
            model = WDCNN(num_cls=10, input_len=1024)
        elif model_name == '1D-CNN':
            model = CNN1D(num_cls=10)
        else:
            raise ValueError(f"未知模型: {model_name}")
        
        return model.to(self.device)
    
    def train_model(self, model, train_data, train_labels, val_data, val_labels, epochs=50):
        """训练模型"""
        print(f"\n训练模型 {model.__class__.__name__}...")
        
        # 转换为张量
        train_dataset = TensorDataset(
            torch.FloatTensor(train_data),
            torch.LongTensor(train_labels)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(val_data),
            torch.LongTensor(val_labels)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
# 前向传播
            model_output = model(batch_data)
            # DFTSRNet返回元组，取第一个元素作为分类输出
            if isinstance(model_output, tuple):
                outputs = model_output[0]
            else:
                outputs = model_output
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    batch_data = batch_data.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    val_model_output = model(batch_data)
                    # DFTSRNet返回元组，取第一个元素作为分类输出
                    if isinstance(val_model_output, tuple):
                        val_outputs = val_model_output[0]
                    else:
                        val_outputs = val_model_output
                    loss = criterion(val_outputs, batch_labels)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        return model, train_losses, val_losses
    
    def gini_lorenz(self, energy_per_channel, eps=1e-12):
        """严格的基尼系数实现（Lorenz曲线定义）"""
        x = np.asarray(energy_per_channel, dtype=np.float64)
        x = np.abs(x) + eps
        x_sorted = np.sort(x)  # 升序
        n = x_sorted.size
        index = np.arange(1, n + 1, dtype=np.float64)
        g = (2.0 * np.sum(index * x_sorted) / (n * np.sum(x_sorted))) - (n + 1.0) / n
        return float(np.clip(g, 0.0, 1.0))
    
    def first_layer_feature_map_2d(self, x_first_layer, fs, nfft=1024):
        """将第一层输出转换为2D频谱图（用于图14）"""
        C, T = x_first_layer.shape
        
        # 确保窗口长度不超过信号长度
        nperseg = min(256, T // 4, T)
        if nperseg < 4:  # 如果信号太短，使用整个信号长度
            nperseg = T
        
        win = get_window('hann', nperseg)
        heat_list, f_list = [], None
        
        for c in range(C):
            if T >= nperseg:
                f, Pxx = welch(x_first_layer[c], fs=fs, window=win, 
                              nperseg=nperseg, noverlap=nperseg//2, nfft=nfft)
            else:
                # 信号太短，直接使用FFT
                fft_result = np.fft.fft(x_first_layer[c], n=nfft)
                Pxx = np.abs(fft_result[:nfft//2 + 1])**2
                f = np.fft.fftfreq(nfft, 1/fs)[:nfft//2 + 1]
            
            if f_list is None:
                f_list = f
            # 单通道归一化
            p = np.maximum(Pxx, 0.0)
            p = (p - p.min()) / (p.max() - p.min() + 1e-12)
            heat_list.append(p)
        
        heat = np.vstack(heat_list)  # (C, F)
        f_norm = f_list / fs  # 归一化频率 0~0.5
        return heat, f_norm
    
    def extract_first_layer_features(self, model, data, model_name):
        """提取第一层特征"""
        model.eval()
        features_list = []
        
        with torch.no_grad():
            for i in range(min(10, len(data))):
                sample = torch.FloatTensor(data[i]).unsqueeze(0).to(self.device)
                
                if model_name == 'DFTSRNet':
                    if hasattr(model, 'afbs'):
                        fused_signal, modals, attn_weights = model.afbs(sample)
                        # 取AFBS的各滤波器输出
                        features = modals.squeeze().cpu().numpy()  # (C, T)
                    else:
                        features = model.conv1(sample).squeeze().cpu().numpy()
                elif model_name == 'WDCNN':
                    # 取conv1的输出
                    features = model.conv1(sample).squeeze().cpu().numpy()
                elif model_name == '1D-CNN':
                    # 1D-CNN第一层卷积 (通过backbone[0]访问)
                    features = model.backbone[0](sample).squeeze().cpu().numpy()
                
                features_list.append(features)
        
        return np.array(features_list)
    
    def generate_figure14_feature_maps(self):
        """生成图14：第一层特征映射热图"""
        print("\n=== 生成图14：第一层特征映射热图 ===")
        
        # 加载数据
        train_data, train_labels, val_data, val_labels, test_data, test_labels = self.load_data()
        
        # 选择类别6的样本
        class_6_indices = np.where(test_labels == 6)[0]
        if len(class_6_indices) > 0:
            sample_data = test_data[class_6_indices[:10]]
            print(f"选择类别6的{len(sample_data)}个样本进行分析")
        else:
            sample_data = test_data[:10]
            print("未找到类别6样本，使用前10个测试样本")
        
        fs = 12000  # 采样频率
        
        # 为每个模型生成特征映射
        model_names = ['DFTSRNet', 'WDCNN', '1D-CNN']
        
        for model_name in model_names:
            print(f"\n处理模型: {model_name}")
            
            # 创建并训练模型
            model = self.create_model(model_name)
            trained_model, _, _ = self.train_model(model, train_data, train_labels, val_data, val_labels)
            
            # 提取第一层特征
            features = self.extract_first_layer_features(trained_model, sample_data, model_name)
            
            # 计算平均特征映射
            avg_features = np.mean(features, axis=0)  # (C, T)
            
            # 转换为2D频谱图
            heat_map, freq_norm = self.first_layer_feature_map_2d(avg_features, fs)
            
            # 绘制热图
            plt.figure(figsize=(12, 8))
            plt.imshow(heat_map, aspect='auto', cmap='viridis', 
                      extent=[0, 0.5, 0, heat_map.shape[0]], 
                      interpolation='bilinear', origin='lower')
            
            plt.colorbar(label='Normalized Power')
            plt.xlabel('Normalized Frequency')
            plt.ylabel('Kernel Index')
            plt.title(f'{model_name} - First Layer Feature Maps')
            
            # 保存图片
            save_path = self.output_dir / f'Figure_14_{model_name}_Feature_Maps.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"保存特征映射图: {save_path}")
    
    def compute_l1_l2_from_losses(self, train_losses, val_losses, last_k=10):
        """从训练历史计算L1/L2比值"""
        if len(train_losses) >= last_k and len(val_losses) >= last_k:
            L1 = float(np.mean(train_losses[-last_k:]))
            L2 = float(np.mean(val_losses[-last_k:]))
            return L1, L2, L1 / (L2 + 1e-12)
        else:
            return None, None, None
    
    def generate_table6_sparsity_loss(self):
        """生成表6：基尼指数与损失比"""
        print("\n=== 生成表6：基尼指数与损失比 ===")
        
        # 加载数据
        train_data, train_labels, val_data, val_labels, test_data, test_labels = self.load_data()
        
        results = []
        model_names = ['DFTSRNet', 'WDCNN', '1D-CNN']
        
        for model_name in model_names:
            print(f"\n分析模型: {model_name}")
            
            # 创建并训练模型
            model = self.create_model(model_name)
            trained_model, train_losses, val_losses = self.train_model(
                model, train_data, train_labels, val_data, val_labels)
            
            # 提取第一层特征计算基尼指数
            features = self.extract_first_layer_features(trained_model, test_data[:50], model_name)
            
            # 计算每个通道的能量
            channel_energies = []
            for sample_features in features:
                energies = np.sum(sample_features**2, axis=1)  # 每个通道的能量
                channel_energies.append(energies)
            
            # 取中位数作为代表
            median_energies = np.median(channel_energies, axis=0)
            gini = self.gini_lorenz(median_energies)
            
            # 计算L1/L2比值
            L1, L2, L1_over_L2 = self.compute_l1_l2_from_losses(train_losses, val_losses)
            
            results.append({
                'Model': model_name,
                'Gini': f"{gini:.4f}",
                'L1': f"{L1:.4f}" if L1 is not None else "N/A",
                'L2': f"{L2:.4f}" if L2 is not None else "N/A",
                'L1_over_L2': f"{L1_over_L2:.4f}" if L1_over_L2 is not None else "N/A"
            })
            
            print(f"{model_name} - Gini: {gini:.4f}, L1/L2: {L1_over_L2:.4f}")
        
        # 保存为CSV
        df = pd.DataFrame(results)
        csv_path = self.output_dir / 'Table_6_Sparsity_Loss.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n保存表格: {csv_path}")
        
        return df
    
    def project_band_weights_to_freq(self, weights, band_edges_hz, fs, n_bins=1024):
        """将频带权重投影到频率轴"""
        f = np.linspace(0, fs / 2, n_bins)  # Hz
        w_curve = np.zeros_like(f)
        
        for w, (fl, fh) in zip(weights, band_edges_hz):
            mask = (f >= fl) & (f <= fh)
            w_curve[mask] += w
        
        f_norm = f / fs  # 归一化到0~0.5
        return w_curve, f_norm
    
    def generate_figure15_attention_spectrum(self):
        """生成图15：注意力权重谱（训练前后对比）"""
        print("\n=== 生成图15：注意力权重谱 ===")
        
        # 加载数据
        train_data, train_labels, val_data, val_labels, test_data, test_labels = self.load_data()
        
        # 选择类别6样本
        class_6_indices = np.where(test_labels == 6)[0]
        if len(class_6_indices) > 0:
            sample_data = test_data[class_6_indices[0]]
        else:
            sample_data = test_data[0]
        
        fs = 12000
        
        # 创建未训练模型（训练前）
        untrained_model = self.create_model('DFTSRNet')
        
        # 获取训练前权重
        with torch.no_grad():
            data_tensor = torch.FloatTensor(sample_data).unsqueeze(0).to(self.device)
            if hasattr(untrained_model, 'afbs'):
                _, modals_before, _ = untrained_model.afbs(data_tensor)
                weights_before = torch.mean(torch.abs(modals_before), dim=2).squeeze().cpu().numpy()
            else:
                features_before = untrained_model.conv1(data_tensor)
                weights_before = torch.mean(torch.abs(features_before), dim=2).squeeze().cpu().numpy()
        
        # 训练模型
        trained_model, _, _ = self.train_model(untrained_model, train_data, train_labels, val_data, val_labels)
        
        # 获取训练后权重
        with torch.no_grad():
            if hasattr(trained_model, 'afbs'):
                _, modals_after, _ = trained_model.afbs(data_tensor)
                weights_after = torch.mean(torch.abs(modals_after), dim=2).squeeze().cpu().numpy()
            else:
                features_after = trained_model.conv1(data_tensor)
                weights_after = torch.mean(torch.abs(features_after), dim=2).squeeze().cpu().numpy()
        
        # 定义频带边界（简化版，实际应根据AFBS结构）
        n_bands = len(weights_before)
        band_edges = []
        for i in range(n_bands):
            fl = i * fs / (2 * n_bands)
            fh = (i + 1) * fs / (2 * n_bands)
            band_edges.append([fl, fh])
        band_edges = np.array(band_edges)
        
        # 投影到频率轴
        w_before, f_norm = self.project_band_weights_to_freq(weights_before, band_edges, fs)
        w_after, _ = self.project_band_weights_to_freq(weights_after, band_edges, fs)
        
        # 绘制对比图
        plt.figure(figsize=(12, 8))
        plt.plot(f_norm, w_before, label='Before Training', 
                color=COLOR_SCHEME['light_blue'], linewidth=2)
        plt.plot(f_norm, w_after, label='After Training', 
                color=COLOR_SCHEME['deep_cyan'], linewidth=2)
        
        plt.xlabel('Normalized Frequency')
        plt.ylabel('Attention Weight')
        plt.title('Attention Weight Spectrum: Before vs After Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        save_path = self.output_dir / 'Figure_15_Attention_Spectrum.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"保存注意力权重谱图: {save_path}")
    
    def bandpass_sos(self, fc_hz, bw_hz, fs, order=4):
        """设计带通滤波器"""
        f1 = max(1.0, fc_hz - bw_hz / 2.0)
        f2 = min(fs/2 - 10.0, fc_hz + bw_hz / 2.0)
        wn = [f1 / (fs/2), f2 / (fs/2)]
        sos = butter(order, wn, btype='bandpass', output='sos')
        return sos
    
    def infogram_information(self, x, fs, fc_grid, bw_list, nfft=4096):
        """计算Infogram信息量"""
        I_map = np.zeros((len(bw_list), len(fc_grid)), dtype=np.float64)
        
        for i, bw in enumerate(bw_list):
            for j, fc in enumerate(fc_grid):
                try:
                    sos = self.bandpass_sos(fc, bw, fs, order=4)
                    x_bp = sosfiltfilt(sos, x)
                    env = np.abs(hilbert(x_bp))
                    
                    # 包络谱
                    Se = np.abs(np.fft.rfft(env, n=nfft))**2
                    Se = Se / (Se.sum() + 1e-12)
                    
                    # Shannon熵
                    H = -np.sum(Se * np.log(Se + 1e-12))
                    I = np.log(len(Se)) - H  # 信息量
                    I_map[i, j] = I
                except:
                    I_map[i, j] = 0
        
        return I_map, fc_grid, np.asarray(bw_list, dtype=np.float64)
    
    def load_bearing_params(self):
        """加载轴承参数"""
        try:
            with open('bearing_params.json', 'r', encoding='utf-8') as f:
                params = json.load(f)
            return params
        except:
            # 默认参数
            return {
                "rpm": 1797,
                "ball_diameter": 7.94,
                "pitch_diameter": 39.04,
                "contact_angle": 0,
                "num_balls": 9
            }
    
    def calculate_bearing_fault_frequencies(self, bearing_params=None):
        """计算轴承故障特征频率"""
        if bearing_params is None:
            bearing_params = self.load_bearing_params()
        
        rpm = bearing_params['rpm']
        Db = bearing_params['ball_diameter']
        Dp = bearing_params['pitch_diameter']
        alpha = math.radians(bearing_params['contact_angle'])
        Nb = bearing_params['num_balls']
        
        fr = rpm / 60  # 转频
        
        # 计算故障特征频率
        BPFO = (Nb / 2) * fr * (1 - (Db / Dp) * math.cos(alpha))  # 外圈故障频率
        BPFI = (Nb / 2) * fr * (1 + (Db / Dp) * math.cos(alpha))  # 内圈故障频率
        BSF = (Dp / (2 * Db)) * fr * (1 - ((Db / Dp) * math.cos(alpha))**2)  # 滚动体故障频率
        FTF = (fr / 2) * (1 - (Db / Dp) * math.cos(alpha))  # 保持架故障频率
        
        return {
            'fr': fr,
            'BPFO': BPFO,
            'BPFI': BPFI, 
            'BSF': BSF,
            'FTF': FTF
        }
    
    def generate_figure16_infogram(self):
        """生成图16：Infogram与包络谱"""
        print("\n=== 生成图16：Infogram与包络谱 ===")
        
        # 加载数据
        _, _, _, _, test_data, test_labels = self.load_data()
        
        # 选择类别6样本
        class_6_indices = np.where(test_labels == 6)[0]
        if len(class_6_indices) > 0:
            sample_data = test_data[class_6_indices[0]]
        else:
            sample_data = test_data[0]
        
        fs = 12000
        
        # 定义Infogram参数
        fc_grid = np.arange(500, 6000, 250)  # 中心频率网格
        bw_list = [600, 900, 1200, 1500, 1800]  # 带宽列表
        
        # 计算Infogram
        I_map, fc_grid, bw_array = self.infogram_information(sample_data, fs, fc_grid, bw_list)
        
        # 找到最优参数
        max_idx = np.unravel_index(np.argmax(I_map), I_map.shape)
        optimal_bw = bw_array[max_idx[0]]
        optimal_fc = fc_grid[max_idx[1]]
        
        print(f"最优共振频段: Fc={optimal_fc:.0f}Hz, Bw={optimal_bw:.0f}Hz")
        
        # 绘制Infogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图：Infogram热图
        im1 = ax1.imshow(I_map, aspect='auto', cmap='viridis', 
                        extent=[fc_grid[0], fc_grid[-1], bw_array[0], bw_array[-1]],
                        origin='lower')
        ax1.plot(optimal_fc, optimal_bw, 'ro', markersize=10, markerfacecolor='red')
        ax1.set_xlabel('Center Frequency (Hz)')
        ax1.set_ylabel('Bandwidth (Hz)')
        ax1.set_title(f'Infogram: Optimal Bw={optimal_bw:.0f}Hz, Fc={optimal_fc:.0f}Hz')
        plt.colorbar(im1, ax=ax1, label='Information Content')
        
        # 右图：包络谱
        sos = self.bandpass_sos(optimal_fc, optimal_bw, fs)
        x_filtered = sosfiltfilt(sos, sample_data)
        envelope = np.abs(hilbert(x_filtered))
        
        # 计算包络谱
        freqs = np.fft.rfftfreq(len(envelope), 1/fs)
        envelope_spectrum = np.abs(np.fft.rfft(envelope))
        
        ax2.plot(freqs, envelope_spectrum, color=COLOR_SCHEME['deep_cyan'], linewidth=1.5)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('Envelope Spectrum')
        ax2.grid(True, alpha=0.3)
        
        # 标注特征频率
        fault_freqs = self.calculate_bearing_fault_frequencies()
        colors = [COLOR_SCHEME['gold'], COLOR_SCHEME['orange'], COLOR_SCHEME['cyan']]
        labels = ['BPFO', 'BPFI', 'BSF']
        
        for i, (label, freq) in enumerate(zip(labels, [fault_freqs['BPFO'], fault_freqs['BPFI'], fault_freqs['BSF']])):
            if freq < freqs[-1]:
                ax2.axvline(freq, color=colors[i % len(colors)], linestyle='--', alpha=0.7, label=f'{label}: {freq:.1f}Hz')
                ax2.axvline(2*freq, color=colors[i % len(colors)], linestyle=':', alpha=0.5, label=f'2×{label}: {2*freq:.1f}Hz')
        
        ax2.legend()
        ax2.set_xlim(0, 1000)  # 限制显示范围
        
        plt.tight_layout()
        
        # 保存图片
        save_path = self.output_dir / 'Figure_16_Infogram_Analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"保存Infogram分析图: {save_path}")
    
    def run_complete_analysis(self):
        """运行完整的可解释性分析"""
        print("\n" + "="*60)
        print("开始VKCNN第五章可解释性分析复刻")
        print("="*60)
        
        try:
            # 生成图14：特征映射热图
            self.generate_figure14_feature_maps()
            
            # 生成表6：基尼指数与损失比
            self.generate_table6_sparsity_loss()
            
            # 生成图15：注意力权重谱
            self.generate_figure15_attention_spectrum()
            
            # 生成图16：Infogram分析
            self.generate_figure16_infogram()
            
            print("\n" + "="*60)
            print("可解释性分析完成！")
            print(f"所有结果已保存到: {self.output_dir}")
            print("生成的文件:")
            for file in self.output_dir.glob('*'):
                print(f"  - {file.name}")
            print("="*60)
            
        except Exception as e:
            print(f"\n分析过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    analyzer = VKCNNExplainabilityAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()