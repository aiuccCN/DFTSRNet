# -*- coding: utf-8 -*-
"""
corrected_explainability_analysis.py

VKCNN风格的5.1节可解释性实验（完整可运行版本）
-------------------------------------------------------------------
该脚本一次性生成以下产物（与文中图表对应）：
  - 图14：不同核函数（DFTSRNet/WDCNN/1D‑CNN）第一层特征映射（Normalized frequency × Kernel）
  - 表6：基尼指数 + 损失比 l1/l2（可选从日志读入）
  - 图15：训练前/后频带注意力权重谱（需要模型暴露 band_w）
  - 图16：Infogram（共振频段） + 包络谱

【与原版相比的重要修正】
1) 基尼指数（Gini）严格采用 Lorenz 曲线定义的解析式、对 0 值与缩放不敏感，并支持逐核/逐样本聚合。
   G(x) = (2 * sum_{i=1..n} i * x_i) / (n * sum_x) - (n + 1) / n   , x 已按非降序排列且非负。
2) Infogram 采用 “带通滤波 -> Hilbert 包络 -> 包络谱 -> Shannon 熵 H -> 归一化负熵 J = 1 - H/logN”
   在 (中心频率 fc, 带宽 bw) 的二维网格上评估，J 越大表明该带产生更加尖锐的调制峰，符合启发式定义。

【使用说明】
python corrected_explainability_analysis.py \
    --data_dir data \
    --fs 12000 \
    --models DFTSRNet:models/DFTSRNet.pth,WDCNN:models/WDCNN.pth,CNN1D:models/CNN1D.pth \
    --log_train results/logs/train_loss.csv --log_val results/logs/val_loss.csv \
    --out_dir results/vis

注：
- 若无法直接从模型取到第一层输出，可事先将第一层特征图保存为 data/featmaps_{tag}.npy，
  则本脚本会自动读取并跳过前向提取阶段。
- 注意力权重谱需要模型在前向过程暴露 self.band_w；若没有将跳过图15但不影响其他产物。
"""

import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, hilbert

# ---------------------------
# 配置与小工具
# ---------------------------

PALETTE = {
    "blue1": (2/255, 72/255, 111/255),
    "blue2": (3/255, 119/255, 191/255),
    "teal":  (19/255, 103/255, 131/255),
    "cyan":  (144/255, 201/255, 231/255),
    "gold":  (254/255, 183/255, 5/255),
    "orange":(250/255, 134/255, 0/255),
}

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

# ---------------------------
# 基尼指数：严谨实现
# ---------------------------

def gini_1d(x: np.ndarray) -> float:
    """
    计算一维向量的基尼指数（0~1），x可以包含0/负数，内部取绝对值。
    参考解析式：G = (2 * sum(i * x_i)) / (n * sum_x) - (n + 1)/n
    注意：要求 x_i 非负且按非降序排列；若 sum_x == 0 则返回 0。
    """
    x = np.abs(np.asarray(x, dtype=np.float64).ravel())
    s = x.sum()
    if s <= 0:
        return 0.0
    xs = np.sort(x)  # 非降序
    n = xs.size
    # i 从 1 开始
    i = np.arange(1, n + 1, dtype=np.float64)
    g = (2.0 * np.sum(i * xs) / (n * s)) - (n + 1.0) / n
    # 数值噪声修正到[0,1]
    return float(np.clip(g, 0.0, 1.0))

def gini_feature_map(feat: np.ndarray, use_fft: bool = True) -> float:
    """
    对第一层特征映射计算“平均基尼指数”
    feat: (B, C, L) 或 (C, L)
    - 若 use_fft=True，则在 L 维做 rFFT 后的谱幅度作为分布（推荐，与图14一致）
    - 否则直接在时间维做稀疏度评估
    """
    arr = np.asarray(feat)
    if arr.ndim == 2:
        arr = arr[None, ...]  # -> (1,C,L)
    B, C, L = arr.shape
    if use_fft:
        spec = np.abs(np.fft.rfft(arr, axis=-1))  # (B,C,F)
        spec = spec / (spec.sum(axis=-1, keepdims=True) + 1e-12)
        rows = spec.reshape(-1, spec.shape[-1])   # (B*C,F)
    else:
        pos = np.abs(arr)
        pos = pos / (pos.sum(axis=-1, keepdims=True) + 1e-12)
        rows = pos.reshape(-1, L)                 # (B*C,L)

    g_list = [gini_1d(r) for r in rows]
    return float(np.mean(g_list))

# ---------------------------
# Overlap Ratio（可选）
# ---------------------------

def overlap_ratio(spec: np.ndarray, threshold_db: float = -20.0) -> float:
    """
    计算两两通道谱的“重叠率”（交并比的平均）。
    spec: (C,F) 非负谱，内部将按阈值二值化。
    threshold_db: 相对最大幅度的阈值（dB）。默认 -20 dB。
    """
    C, F = spec.shape
    if C < 2:
        return 0.0
    spec = spec / (spec.max(axis=1, keepdims=True) + 1e-12)
    # 转 dB
    db = 20 * np.log10(spec + 1e-12)
    mask = (db >= threshold_db).astype(np.uint8)  # (C,F)
    inter = 0.0
    union = 0.0
    from itertools import combinations
    for i, j in combinations(range(C), 2):
        inter += np.logical_and(mask[i], mask[j]).sum()
        union += np.logical_or(mask[i], mask[j]).sum()
    return float(inter / max(union, 1.0))

# ---------------------------
# Infogram：带通 + 包络谱 + 归一化负熵
# ---------------------------

def _butter_bandpass(low, high, fs, order=4):
    ny = 0.5 * fs
    low = max(low, 1.0) / ny
    high = min(high, ny - 1.0) / ny
    if not (0 < low < high < 1):
        return None, None
    b, a = butter(order, [low, high], btype='band')
    return b, a

def envelope_spectrum(x: np.ndarray, fs: float, nfft: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 Hilbert 包络谱（幅度谱）
    返回: (freq, amp)
    """
    x = np.asarray(x, dtype=np.float64)
    env = np.abs(hilbert(x))
    amp = np.abs(np.fft.rfft(env, n=nfft))
    freq = np.fft.rfftfreq(nfft, d=1.0/fs)
    return freq, amp

def infogram(x: np.ndarray,
             fs: float,
             centers: np.ndarray,
             bandwidths: np.ndarray,
             nfft: int = 4096) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算 Infogram（二维网格上每个 (bw, fc) 的归一化负熵）
    定义：对带通后的 Hilbert 包络谱做 Shannon 熵 H，
         J = 1 - H / log(N)，其中 N 为谱点数（去掉直流后的有效点数）。
    返回：heat(bw,fc), centers(Hz), bandwidths(Hz)
    """
    x = np.asarray(x, dtype=np.float64)
    heat = np.full((len(bandwidths), len(centers)), np.nan, dtype=np.float64)

    for bi, bw in enumerate(bandwidths):
        half = bw / 2.0
        for ci, fc in enumerate(centers):
            b, a = _butter_bandpass(fc - half, fc + half, fs, order=4)
            if b is None:
                continue
            y = filtfilt(b, a, x)  # 稳定零相位滤波
            _, amp = envelope_spectrum(y, fs, nfft=nfft)
            amp[0] = 0.0  # 去直流
            p = amp / (amp.sum() + 1e-12)
            # Shannon 熵（只在 p>0 上计算）
            p_pos = p[p > 0]
            H = -np.sum(p_pos * np.log(p_pos))
            H_max = math.log(len(p_pos))
            J = 0.0 if H_max <= 0 else (1.0 - H / H_max)
            heat[bi, ci] = J
    return heat, centers, bandwidths

# ---------------------------
# 绘图函数
# ---------------------------

def plot_feature_map_spectrum(feat: np.ndarray,
                              fs: float,
                              out_png: str,
                              title: str = 'Feature maps',
                              vmax: Optional[float] = None):
    """
    将第一层特征 (C,L) 或 (B,C,L) 画为 “Kernel × Normalized frequency”的图（与 Fig14 一致）
    """
    arr = np.asarray(feat)
    if arr.ndim == 3:
        arr = arr[0]  # 取一条样本 (C,L)
    assert arr.ndim == 2, "feat must be (C,L) or (B,C,L)"
    spec = np.abs(np.fft.rfft(arr, axis=-1))
    spec = spec / (spec.max() + 1e-12)
    F = spec.shape[-1]
    f = np.linspace(0, 0.5, F)  # 归一化到[0,0.5]（Nyquist）

    plt.figure(figsize=(5.2, 3.2))
    im = plt.imshow(spec, aspect='auto', origin='lower',
                    extent=[f[0], f[-1], 0, spec.shape[0]],
                    cmap='viridis', vmin=0, vmax=vmax)
    plt.xlabel('Normalized frequency')
    plt.ylabel('Kernel')
    cbar = plt.colorbar(im)
    cbar.set_label('Amplitude (norm.)')
    plt.title(title)
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=300)
    plt.close()

def plot_infogram(heat: np.ndarray, centers: np.ndarray, bws: np.ndarray, out_png: str, fs: float):
    plt.figure(figsize=(6.0, 3.4))
    extent = [centers.min(), centers.max(), 1, len(bws)]
    im = plt.imshow(heat, aspect='auto', origin='lower',
                    extent=[centers.min(), centers.max(), 1, len(bws)],
                    cmap='viridis')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Level')  # 可将 bw -> Level 映射解释在图注
    cbar = plt.colorbar(im)
    cbar.set_label('Normalized negentropy')
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=300)
    plt.close()

def plot_envelope_spectrum(x: np.ndarray, fs: float, out_png: str, xlim: Optional[Tuple[float,float]] = None):
    f, amp = envelope_spectrum(np.asarray(x, dtype=np.float64), fs, nfft=4096)
    amp = amp / (amp.max() + 1e-12)
    plt.figure(figsize=(6.0, 3.0))
    plt.plot(f, amp, linewidth=1.2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (norm.)')
    if xlim:
        plt.xlim(*xlim)
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=300)
    plt.close()

# ---------------------------
# 训练/验证损失比（可选）
# ---------------------------

def compute_loss_ratio(train_csv: str, val_csv: str, tail: int = 10) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    从日志文件中读取最后 tail 个epoch的loss均值，计算 l1/l2。
    返回 (l1, l2, ratio)。若文件不存在，返回 (None,None,None)
    """
    if (not train_csv) or (not val_csv) or (not os.path.exists(train_csv)) or (not os.path.exists(val_csv)):
        return None, None, None
    dt = pd.read_csv(train_csv)
    dv = pd.read_csv(val_csv)
    def tail_mean(df):
        col = [c for c in df.columns if 'loss' in c.lower()]
        if not col:
            col = df.columns[-1:]
        return float(df[col[0]].tail(tail).mean())
    l1 = tail_mean(dt); l2 = tail_mean(dv)
    return l1, l2, (l1 / l2 if l2 and l2>0 else None)

# ---------------------------
# 主流程
# ---------------------------

@dataclass
class Args:
    data_dir: str = 'data'
    fs: float = 12000.0
    models: str = ''  # "DFTSRNet:models/DFTSRNet.pth,WDCNN:models/WDCNN.pth,CNN1D:models/CNN1D.pth"
    out_dir: str = 'results/vis'
    # infogram grid
    fmin: float = 1000.0
    fmax: float = 6000.0
    fstep: float = 250.0
    bw_list: str = '500,1000,1500,2000'
    sample_index: int = 0
    # optional logs
    log_train: str = ''
    log_val: str = ''
    # feature map fallback
    use_fft_for_gini: bool = True

def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='data')
    p.add_argument('--fs', type=float, default=12000.0)
    p.add_argument('--models', type=str, default='')
    p.add_argument('--out_dir', type=str, default='results/vis')
    p.add_argument('--fmin', type=float, default=1000.0)
    p.add_argument('--fmax', type=float, default=6000.0)
    p.add_argument('--fstep', type=float, default=250.0)
    p.add_argument('--bw_list', type=str, default='500,1000,1500,2000')
    p.add_argument('--sample_index', type=int, default=0)
    p.add_argument('--log_train', type=str, default='')
    p.add_argument('--log_val', type=str, default='')
    p.add_argument('--use_fft_for_gini', action='store_true', default=True)
    ns = p.parse_args()
    return Args(**vars(ns))

# ---------------------------
# 与模型的接口：尽量宽容
# ---------------------------

def try_load_model_and_first_layer(tag: str, pth: str, xbatch: np.ndarray) -> Optional[np.ndarray]:
    """
    尝试加载模型并取第一层特征 (B,C,L)。
    约定：
      - DFTSRNet 前向后在实例上有属性 .afbs_raw
      - WDCNN / CNN1D 在实例上有 .conv1_raw 或 .first_raw
    若失败则返回 None
    """
    try:
        import torch
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 用户可根据自己工程替换下述 import
        from src.model import DFTSRNet, WDCNN, CNN1D  # 仅示例
        name = tag.upper()
        if 'DFT' in name:
            mdl = DFTSRNet().to(dev).eval()
        elif 'WDC' in name:
            mdl = WDCNN().to(dev).eval()
        else:
            mdl = CNN1D().to(dev).eval()
        ckpt = torch.load(pth, map_location=dev)
        if 'model_state_dict' in ckpt:
            mdl.load_state_dict(ckpt['model_state_dict'])
        else:
            mdl.load_state_dict(ckpt)
        xb = torch.from_numpy(xbatch).float().to(dev)
        with torch.no_grad():
            _ = mdl(xb)
        for attr in ['afbs_raw', 'conv1_raw', 'first_raw']:
            if hasattr(mdl, attr):
                feat = getattr(mdl, attr)
                feat = to_numpy(feat)  # (B,C,L)
                return feat
    except Exception as e:
        print(f'[WARN] {tag} first-layer extraction failed: {e}')
    return None

# ---------------------------
# 运行
# ---------------------------

def main():
    args = parse_args()
    fs = float(args.fs)
    ensure_dir(args.out_dir)

    # 1) 读取一个原始样本（用于 Fig16）
    #    尝试 data/raw_signal.npy；否则从 test_data.npy 取一条。
    raw_sig_fp = os.path.join(args.data_dir, 'raw_signal.npy')
    if os.path.exists(raw_sig_fp):
        x_raw = np.load(raw_sig_fp).astype(np.float64)
    else:
        test_data_fp = os.path.join(args.data_dir, 'test_data.npy')
        if os.path.exists(test_data_fp):
            X = np.load(test_data_fp)
            x_raw = X[args.sample_index].astype(np.float64)
        else:
            raise FileNotFoundError('缺少 data/raw_signal.npy 或 data/test_data.npy')

    # 2) 第一层特征：优先读取预存 npy；否则尝试从模型提取
    tags_and_paths = []
    if args.models:
        for item in args.models.split(','):
            tag, pth = item.split(':', 1)
            tags_and_paths.append((tag.strip(), pth.strip()))
    else:
        # 没指定则仅用预存 npy
        pass

    featmaps: Dict[str, np.ndarray] = {}
    for tag, pth in tags_and_paths:
        # 小批量构造：取 8 条数据
        test_data_fp = os.path.join(args.data_dir, 'test_data.npy')
        if not os.path.exists(test_data_fp):
            print('[WARN] 未找到 test_data.npy，无法从模型提取第一层特征。')
            break
        X = np.load(test_data_fp).astype(np.float32)
        xbatch = X[:8] if X.ndim == 2 else X[:8, None, :]
        feat = try_load_model_and_first_layer(tag, pth, xbatch)
        if feat is not None:
            featmaps[tag] = feat
            np.save(os.path.join(args.data_dir, f'featmaps_{tag}.npy'), feat)

    # 同时读取任何已经存在的 featmaps_*.npy
    for tag in ['DFTSRNet','WDCNN','CNN1D','dfts','wdcnn','cnn']:
        fp = os.path.join(args.data_dir, f'featmaps_{tag}.npy')
        if os.path.exists(fp):
            featmaps[tag] = np.load(fp)

    if not featmaps:
        raise RuntimeError('未能获得任何第一层特征映射。请先提供 data/featmaps_{tag}.npy 或 --models 进行提取。')

    # 3) 图14：每个模型画一张
    for tag, feat in featmaps.items():
        out_png = os.path.join(args.out_dir, f'Fig14_{tag}.png')
        plot_feature_map_spectrum(feat, fs, out_png, title=f'{tag} - Feature maps')

    # 4) 表6：基尼 + （可选）损失比
    rows = []
    for tag, feat in featmaps.items():
        # 采用 rFFT 幅度做稀疏性：与图14一致
        g = gini_feature_map(feat, use_fft=args.use_fft_for_gini)
        # 用整个特征做 L1/L2（幅度）作为“强度对比”参考；论文中的 l1/l2 是训练/测试损失比，
        # 若提供日志则覆盖
        arr = np.abs(to_numpy(feat))
        l1 = float(np.mean(np.abs(arr)))
        l2 = float(np.sqrt(np.mean(arr ** 2)))
        ratio = l1 / l2 if l2 > 0 else np.nan
        rows.append([tag, g, l1, l2, ratio])
    df = pd.DataFrame(rows, columns=['核', '基尼指数', 'l1', 'l2', 'l1/l2(幅度)'])

    # 覆盖为训练/验证损失比（可选）
    t_l1, v_l2, r = compute_loss_ratio(args.log_train, args.log_val)
    if r is not None:
        df.loc[df['核'].idxmax(), 'l1'] = t_l1
        df.loc[df['核'].idxmax(), 'l2'] = v_l2
        df.loc[df['核'].idxmax(), 'l1/l2(幅度)'] = r  # 列名不改，避免破坏表头

    df_out = os.path.join(args.out_dir, 'Table6_sparsity.csv')
    df.to_csv(df_out, index=False)
    print('[INFO] 保存：', df_out)
    print(df)

    # 5) 图15：注意力权重谱（如果提供 attn_pre/post）
    pre_fp = os.path.join(args.data_dir, 'attn_pre.npy')
    post_fp= os.path.join(args.data_dir, 'attn_post.npy')
    meta_fp= os.path.join(args.data_dir, 'band_meta.json')  # [{'fc':Hz,'bw':Hz,'fs':fs}, ...]
    if os.path.exists(pre_fp) and os.path.exists(post_fp) and os.path.exists(meta_fp):
        pre = np.load(pre_fp).mean(axis=0)
        post= np.load(post_fp).mean(axis=0)
        meta = json.load(open(meta_fp, 'r', encoding='utf-8'))
        fc = np.array([m['fc'] for m in meta], dtype=np.float64)
        bw = np.array([m.get('bw', 0.0) for m in meta], dtype=np.float64)
        normf = fc / (0.5 * meta[0].get('fs', fs))
        plt.figure(figsize=(6.0, 3.2))
        plt.plot(normf, pre, label='Before training', linewidth=1.2)
        plt.plot(normf, post, label='After training', linewidth=1.6)
        plt.xlabel('Normalized frequency')
        plt.ylabel('Attention weight')
        plt.legend()
        plt.tight_layout()
        out15 = os.path.join(args.out_dir, 'Fig15_AttnSpectrum.png')
        plt.savefig(out15, dpi=300); plt.close()
        print('[INFO] 保存：', out15)
    else:
        print('[WARN] 缺少 attn_pre/attn_post/band_meta，跳过图15。')

    # 6) 图16：Infogram + 包络谱
    centers = np.arange(args.fmin, args.fmax + 1e-9, args.fstep, dtype=np.float64)
    bws = np.array([float(x) for x in args.bw_list.split(',')], dtype=np.float64)
    heat, cen, bw = infogram(x_raw, fs, centers, bws, nfft=4096)
    out16a = os.path.join(args.out_dir, 'Fig16a_Infogram.png')
    plot_infogram(heat, cen, bw, out16a, fs)
    out16b = os.path.join(args.out_dir, 'Fig16b_EnvelopeSpectrum.png')
    plot_envelope_spectrum(x_raw, fs, out16b, xlim=(0, 800))
    print('[INFO] 保存：', out16a, ' 和 ', out16b)

if __name__ == '__main__':
    main()
