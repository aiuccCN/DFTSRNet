# -*- coding: utf-8 -*-
"""
Autogram (MODWPT-like via Wavelet Packet reconstruction) for class6.npy
- Replicates the MATLAB script structure you provided:
  * Wavelet packet (db12), recompose each node to full-length subband
  * Squared-envelope -> unbiased ACF -> kurtosis-like metric (KUR)
  * Brick-tiling across frequency per level to form Autogram
  * Pick (level, node) with max KUR and report fc, Bw
- Outputs:
  * PNG/SVG of the Autogram
  * (optional) SES of the max-KUR node
"""

import os
import numpy as np
import pywt
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.signal.windows import hann

# ---------------- Paths & constants ----------------
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.environ.get("AUTOGRAM_DATA", os.path.join(_HERE, "class6.npy"))
FS = 12000  # Hz
LEVEL = 5   # same default as MATLAB脚本里的示例，可改 5~8
WAVELET = "db12"
OUT_DIR = os.environ.get("AUTOGRAM_OUT", _HERE)
os.makedirs(OUT_DIR, exist_ok=True)

# ----- 版式（Times New Roman、细长）-----
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,  # 改为10号字体
    "axes.labelsize": 10,  # 坐标轴标题10号字体

    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

# ---------------- Utils ----------------
def unbiased_autocorr(x: np.ndarray) -> np.ndarray:
    """Unbiased autocorrelation (full) of a 1D array."""
    x = np.asarray(x, dtype=float)
    n = x.size
    # full correlation
    r = np.correlate(x, x, mode="full")
    # unbiased normalization: divide by (n - |lag|)
    lags = np.arange(-n + 1, n, dtype=int)
    bias = n - np.abs(lags)
    r = r / np.maximum(bias, 1)
    return r

def movmean(x, win):
    """Simple moving average with 'same' length."""
    w = np.ones(int(win), dtype=float) / float(win)
    return np.convolve(x, w, mode="same")

def kurt_metric(ac_seg: np.ndarray, mode_q: int = 0) -> float:
    """
    MATLAB Kurt() 的 Python 复刻：
      - q==0: x -= min(x)
      - q==+1: 去均值滑窗、保留正半轴
      - q==-1: 去均值滑窗、保留负半轴
      - 最后 E = mean(x^2); K = mean(x^4)/E^2
    """
    x = np.array(ac_seg, dtype=float).copy()
    if mode_q == 1:
        x = x - movmean(x, 4000)
        x = np.where(x > 0, x, 0.0)
    elif mode_q == -1:
        x = x - movmean(x, 4000)
        x = np.where(x < 0, x, 0.0)
    # 与 MATLAB 同步：去掉最小值，避免负值拖拽能量
    x = x - np.min(x)
    E = np.mean(x ** 2) + 1e-12
    K = np.mean(x ** 4) / (E ** 2)
    return float(K)

def waveletpacket_subbands(x, wavelet: str, level: int):
    """
    用 pywt.WaveletPacket 取各层 2^lev 个节点的重构信号（与原长相同）。
    返回: list(level_rows), 其中每个 level_rows 是 [subband_i ...] 的列表，
         level=0 时返回 [x] 本身以对齐 MATLAB 的做法。
    """
    wp = pywt.WaveletPacket(data=x, wavelet=wavelet, mode="symmetric", maxlevel=level)
    out = []
    # level 0：直接用原信号
    out.append([x.copy()])
    for lev in range(1, level + 1):
        nodes = wp.get_level(lev, order="natural")  # 自然顺序：等价二进分段频带
        rows = [node.reconstruct(False) for node in nodes]
        out.append(rows)
    return out

def compute_autogram(x_raw: np.ndarray, fs: float, level: int, wavelet: str, lmp: int = 0):
    """
    完整计算 Autogram：
      * 生成各层子带（重构为原长）
      * 各子带做平方包络 -> 无偏自相关 -> kurt_metric
      * 砖块化铺频率
      * 挑最大 KUR 节点并计算 fc, Bw（与 MATLAB 公式一致）
    返回：
      color: (level+1, 2^level) 的砖块矩阵
      x_edges: 频率边界 (2^level + 1,)
      best_info: dict(level, node, K, fc_Hz, Bw_Hz)
    """
    x = np.asarray(x_raw, dtype=float)
    x -= np.mean(x)

    # MATLAB 的两个“去边/起始”参数
    N = len(x)
    ss0 = int(0.005 * N)     # 去掉滤波瞬态
    AAA = 1.55               # 截取 ACF 的上界系数
    cor = 0.010              # 去掉 ACF 的头部（AC）少量样本
    # 重要：AC 段起点，按 N 比例
    # （MATLAB 代码里对 env^2 的 ACF 用 length(c) 决定下标）
    def ac_segment(z):
        L = len(z)
        ac = unbiased_autocorr(z)
        zero = L - 1  # 零时延在 full ACF 的中点
        ss = int(cor * L) + 1
        hi = int(AAA * L)
        return ac[zero + ss: hi]

    # 1) 子带
    bands = waveletpacket_subbands(x, wavelet, level)

    # 2) 每个子带做 KUR
    KUR = []
    for lev in range(0, level + 1):
        row = []
        for sub in bands[lev]:
            # 与 MATLAB 对齐：去瞬态，再取包络
            c = sub[ss0: -ss0] if (2 * ss0 < len(sub)) else sub.copy()
            env = np.abs(hilbert(c))
            seg = ac_segment(env ** 2)
            row.append(kurt_metric(seg, lmp))
        KUR.append(row)

    # 3) 砖块化：最顶层 2^level 列，低层复制成大砖块
    T = 2 ** level
    color = np.zeros((level + 1, T), dtype=float)
    for lev in range(0, level + 1):
        vals = np.asarray(KUR[lev], dtype=float)
        TE = T // (2 ** lev)
        color[lev, :] = np.repeat(vals, TE)[:T]

    # 4) 找最大 KUR 的层与节点
    best_K = -np.inf
    best_lev = 0
    best_idx = 0
    for lev in range(0, level + 1):
        vals = np.asarray(KUR[lev])
        idx = int(np.argmax(vals))
        if vals[idx] > best_K:
            best_K = float(vals[idx])
            best_lev = lev
            best_idx = idx  # 0-based

    # 5) fc, Bw 公式（与 MATLAB 标题相同）
    # LM1 = best_lev, JM1 = best_idx+1（MATLAB 从 1 开始记节点）
    LM1 = best_lev
    JM1 = best_idx + 1
    # Bw = Fs / 2^(LM1+1)
    Bw = fs / (2 ** (LM1 + 1))
    # fc = j/2^LM1 * Fs*0.5 - Fs/2^(LM1+2)
    fc = JM1 / (2 ** LM1) * (fs * 0.5) - fs / (2 ** (LM1 + 2))

    # x 轴边界：0 … Fs/2，等分为 2^level 个砖块
    x_edges = np.linspace(0.0, fs * 0.5, T + 1)
    best = dict(level=LM1, node=JM1, K=best_K, fc_Hz=fc, Bw_Hz=Bw)
    return color, x_edges, best

def plot_autogram(color, x_edges, level, fs, out_path, best_info=None):
    """
    画出与 MATLAB 类似的"砖块化" Autogram：
      - 横轴 Hz
      - 纵轴 Level（从 0 顶部往下增大；若想 1..N，从 1 开始标）
    """
    # 归一化显示（仅用于 colormap），真实 KUR 数值在打印里看
    Z = color.copy()
    vmin, vmax = Z.min(), Z.max()
    if vmax > vmin:
        Z = (Z - vmin) / (vmax - vmin)

    # pcolormesh 需要 (M+1,N+1) 的边界；我们让纵轴每层等高：
    y_edges = np.arange(0, level + 1 + 1, 1.0)  # 0..level+1
    X, Y = np.meshgrid(x_edges, y_edges)

    # 9cm宽度转换为英寸 (1英寸 = 2.54cm)
    fig_width_inch = 9 / 2.54
    fig, ax = plt.subplots(figsize=(fig_width_inch, 4.6), constrained_layout=True)
    pcm = ax.pcolormesh(X, Y, Z[::-1, :], shading="auto", cmap="viridis")  # 反转让 level=0 在顶部
    ax.set_xlim(0, fs * 0.5)
    ax.set_ylim(0, level + 1)
    # ytick 显示 0..level（若要 1..level，把 +1、去 0）
    ax.set_yticks(np.arange(0.5, level + 1.5, 1.0))
    ax.set_yticklabels([str(i) for i in range(level, -1, -1)])  # 顶部 0，向下增
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Level")

    # 不添加colorbar标注，但添加显示fc和Bw的标题
    if best_info:
        ax.set_title(f"fc = {best_info['fc_Hz']:.2f} Hz, Bw = {best_info['Bw_Hz']:.2f} Hz")

    fig.savefig(out_path + ".png", dpi=300)
    fig.savefig(out_path + ".svg")
    plt.close(fig)

def plot_ses_of_best(x, best, bands, fs, out_path):
    """
    可选：把最佳节点的平方包络谱（SES）画出来，便于核对解调带。
    """
    LM1 = best["level"]
    JM1 = best["node"]
    # 取对应子带（level 0 用原始）
    sub = bands[LM1][JM1] if LM1 > 0 else bands[0][0]
    sub = np.asarray(sub, dtype=float)
    N = sub.size
    ss0 = int(0.005 * N)
    c = sub[ss0:-ss0] if (2 * ss0 < N) else sub
    env = np.abs(hilbert(c))
    # SES: FFT of (env^2) with Hann window
    nfft = int(2 * np.ceil(len(env) / 2.0))
    w = hann(len(env), sym=False)
    SES = np.abs(np.fft.rfft((env ** 2) * w, n=nfft))
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)

    # 9cm宽度转换为英寸
    fig_width_inch = 9 / 2.54
    fig, ax = plt.subplots(figsize=(fig_width_inch, 3.0), constrained_layout=True)
    ax.plot(f, SES, lw=1.2)
    ax.set_xlim(0, 2000)  # 横坐标改为2000
    ax.set_ylim(0, 8000)  # 纵坐标改为8000
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("SES amplitude")
    # 去掉图标题
    fig.savefig(out_path + "_SES.png", dpi=300)
    plt.close(fig)

def plot_combined_autogram_ses(color, x_edges, level, fs, x, best, bands, out_path):
    """
    合并绘制Autogram热力图（上）和SES包络图（下）
    """
    # 15cm宽度转换为英寸 (1英寸 = 2.54cm)
    fig_width_inch = 15 / 2.54
    
    # 创建子图：上下布局
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width_inch, 6.0), 
                                   constrained_layout=True, height_ratios=[1, 0.8])
    
    # ========== 上图：Autogram热力图 ==========
    # 归一化显示（仅用于 colormap），真实 KUR 数值在打印里看
    Z = color.copy()
    vmin, vmax = Z.min(), Z.max()
    if vmax > vmin:
        Z = (Z - vmin) / (vmax - vmin)

    # pcolormesh 需要 (M+1,N+1) 的边界；我们让纵轴每层等高：
    y_edges = np.arange(0, level + 1 + 1, 1.0)  # 0..level+1
    X, Y = np.meshgrid(x_edges, y_edges)

    pcm = ax1.pcolormesh(X, Y, Z[::-1, :], shading="auto", cmap="viridis")  # 反转让 level=0 在顶部
    ax1.set_xlim(0, fs * 0.5)
    ax1.set_ylim(0, level + 1)
    # ytick 显示 0..level
    ax1.set_yticks(np.arange(0.5, level + 1.5, 1.0))
    ax1.set_yticklabels([str(i) for i in range(level, -1, -1)])  # 顶部 0，向下增
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Level")
    
    # 移除fc和Bw标题
    
    # 添加(a)标注到左上角
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='left')
    
    # ========== 下图：SES包络图 ==========
    LM1 = best["level"]
    JM1 = best["node"]
    # 取对应子带（level 0 用原始）
    sub = bands[LM1][JM1] if LM1 > 0 else bands[0][0]
    sub = np.asarray(sub, dtype=float)
    N = sub.size
    ss0 = int(0.005 * N)
    c = sub[ss0:-ss0] if (2 * ss0 < N) else sub
    env = np.abs(hilbert(c))
    # SES: FFT of (env^2) with Hann window
    nfft = int(2 * np.ceil(len(env) / 2.0))
    w = hann(len(env), sym=False)
    SES = np.abs(np.fft.rfft((env ** 2) * w, n=nfft))
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)

    ax2.plot(f, SES, lw=1.2)
    ax2.set_xlim(0, 2000)  # 横坐标改为2000
    ax2.set_ylim(0, 8000)  # 纵坐标改为8000
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("SES amplitude")
    
    # 添加(b)标注到左上角
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='left')
    
    # 保存图片
    fig.savefig(out_path + "_combined.png", dpi=300)
    fig.savefig(out_path + "_combined.svg")
    plt.close(fig)

# 修改main函数
def main():
    # 读信号
    x = np.load(DATA_PATH).astype(float).squeeze()
    print(f"[Data] class6 loaded: length={len(x)}, Fs={FS} Hz")

    # 计算 Autogram
    color, x_edges, best = compute_autogram(x, FS, LEVEL, WAVELET, lmp=0)
    print(f"[Best] level={best['level']}, node={best['node']}, "
          f"K={best['K']:.3f}, fc={best['fc_Hz']:.2f} Hz, Bw={best['Bw_Hz']:.2f} Hz")

    # 计算子带用于SES绘制
    bands = waveletpacket_subbands(x - np.mean(x), WAVELET, LEVEL)
    
    # 出图（合并的Autogram + SES）
    out_base = os.path.join(OUT_DIR, f"Autogram_class6_L{LEVEL}")
    plot_combined_autogram_ses(color, x_edges, LEVEL, FS, x, best, bands, out_base)
    print(f"[Save] {out_base}_combined.png / .svg")
    print("[Done]")

if __name__ == "__main__":
    main()
