# -*- coding: utf-8 -*-
"""
DFTSRNet_AttentionSpectrum:
- 选取Dataset_A中标签==6的一个样本，画其幅度谱；
- “训练前”：使用与已训权重同形状的随机卷积核，估计频段注意力；
- “训练后”：从 DFTSRNet.pth 提取第一层卷积核，估计频段注意力；
- 用红蓝渐变在谱线上编码注意力强弱（红=高，蓝=低）；
- 输出：DFTSRNet_AttentionSpectrum.svg（矢量） + .png（预览）。
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 防止 OpenMP 冲突

import json
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# ========================= 固定版式（8 cm 宽，6.2 cm 高） =========================
CM = 1/2.54
FIG_W = 8 * CM
FIG_H = 6.2 * CM   # 统一尺寸
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

# ========================= 路径硬编码 =========================
DATA_DIR   = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\数据集\西储大学实验数据集\Dataset_A"
MODELS_DIR = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\新版可解释实验\models"
OUT_DIR    = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\新版可解释实验\20250808\注意力加权频谱"
os.makedirs(OUT_DIR, exist_ok=True)

# 模型配置
MODEL_CONFIGS = {
    "WDCNN": {
        "ckpt_path": os.path.join(MODELS_DIR, "WDCNN.pth"),
        "prefer_keys": ["conv1.weight", "backbone.0.weight", "fusion.conv1.weight"]
    },
    "DFTSRNet": {
        "ckpt_path": os.path.join(MODELS_DIR, "DFTSRNet.pth"),
        "prefer_keys": ["fusion.conv1.weight", "conv1.weight", "backbone.0.weight"]
    }
}

FS        = 12000  # 采样率 Hz（按你的数据设定）
LABEL_TGT = 6      # 只画标签=6

# ========================= 工具函数 =========================
def load_checkpoint_state_dict(ckpt_path: str):
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        for k in ["model_state_dict","state_dict","net","model","ema_state_dict"]:
            if k in obj and isinstance(obj[k], (dict, torch.nn.Module)):
                sd = obj[k]
                return sd.state_dict() if isinstance(sd, torch.nn.Module) else sd
        # 退路：字典里大多是 tensor，就当 state_dict
        if len(obj)>0 and sum(isinstance(v, torch.Tensor) for v in obj.values())/len(obj) > 0.5:
            return obj
    if isinstance(obj, torch.nn.Module):
        return obj.state_dict()
    raise RuntimeError("Unrecognized checkpoint format")

def pick_first_conv_weight(state_dict: dict, prefer=("fusion.conv1.weight","conv1.weight","backbone.0.weight")):
    for k in prefer:
        if k in state_dict and isinstance(state_dict[k], torch.Tensor) and state_dict[k].ndim==3:
            return k, state_dict[k]
    # 扫描一个3D权重
    for k,v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.ndim==3 and v.shape[1]>=1:
            return k, v
    raise RuntimeError("No Conv1d weight (3D) found in checkpoint.")

def generate_combined_attention_plot(model_data, x, f_norm, amp):
    """
    生成2×2子图布局的注意力频谱图
    model_data: {model_name: {"A_before": A_before, "A_after": A_after}}
    """
    # 将归一化频率转换为实际频率 (Hz)
    f_hz = f_norm * FS / 2  # f_norm范围[0,1] -> [0, FS/2]
    
    # 创建2×2子图布局
    fig, axes = plt.subplots(2, 2, figsize=(2*FIG_W, 2*FIG_H))
    fig.patch.set_facecolor('white')
    
    # 子图标注
    labels = ['(a)', '(b)', '(c)', '(d)']
    model_names = list(model_data.keys())
    
    # 第一行：训练前 (Before Training)
    for i, model_name in enumerate(model_names):
        ax = axes[0, i]
        ax.set_facecolor("white")
        lc = colored_line(ax, f_hz, amp, model_data[model_name]["A_before"], cmap="coolwarm", lw=1.0)
        cbar = fig.colorbar(lc, ax=ax)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude (g)")
        ax.set_xlim(0, 6000)  # 限制显示范围为0-6000Hz
        ax.grid(True, alpha=0.3)
        # 添加左上角标注
        ax.text(0.02, 0.98, labels[i], transform=ax.transAxes, fontsize=12, 
                va='top', ha='left')
    
    # 第二行：训练后 (After Training)
    for i, model_name in enumerate(model_names):
        ax = axes[1, i]
        ax.set_facecolor("white")
        lc = colored_line(ax, f_hz, amp, model_data[model_name]["A_after"], cmap="coolwarm", lw=1.0)
        cbar = fig.colorbar(lc, ax=ax)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude (g)")
        ax.set_xlim(0, 6000)  # 限制显示范围为0-6000Hz
        ax.grid(True, alpha=0.3)
        # 添加左上角标注
        ax.text(0.02, 0.98, labels[i+2], transform=ax.transAxes, fontsize=12, 
                va='top', ha='left')
    
    plt.tight_layout()
    
    # 保存图片
    base_path = os.path.join(OUT_DIR, "Combined_AttentionSpectrum")
    fig.savefig(base_path + ".svg", bbox_inches="tight")
    fig.savefig(base_path + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    return base_path

def fft_mag_1d(x, fs, nfft=None):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if nfft is None:
        nfft = max(4096, 1<<int(np.ceil(np.log2(n))))
    X = np.fft.rfft(x * np.hanning(n), n=nfft)
    f = np.fft.rfftfreq(nfft, d=1.0/fs)          # 0..fs/2 (Hz)
    amp = np.abs(X) / n                           # 简单幅值标定
    return f, amp

def kernels_attention_curve(w, nfft=4096, use_gate=None):
    """
    w: (K, C, L) 第一层卷积核；返回按频率聚合后的注意力 A(f) in [0,1]
    思路：对每个kernel做rFFT取幅值 |H_k(f)|，对k累加并归一化 -> A(f)。
    若 use_gate 提供大小(K,)的门控向量，则做加权求和。
    """
    W = w.detach().cpu().numpy()
    K, C, L = W.shape
    # rFFT 频点数
    F = nfft//2 + 1
    A = np.zeros(F, dtype=np.float64)
    for k in range(K):
        # 多通道功率合成
        pwr = np.zeros(F, dtype=np.float64)
        for c in range(C):
            H = np.fft.rfft(W[k, c, :], n=nfft)
            pwr += (np.abs(H)**2)
        mag = np.sqrt(pwr)
        if use_gate is not None:
            A += use_gate[k] * mag
        else:
            A += mag
    # min-max 归一到 [0,1]
    A -= A.min()
    A /= (A.max() + 1e-12)
    return A

def colored_line(ax, f_hz, amp, weight01, cmap="coolwarm", lw=1.0):
    """把一条曲线按点权重weight01(0..1)着色。"""
    # 为了实现渐变，拆成很多短线段
    points = np.array([f_hz, amp]).T.reshape(-1,1,2)   # (N,1,2)
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)  # (N-1,2,2)
    # 线段用前一点的权重着色
    cvals  = weight01[:-1]
    lc = LineCollection(segs, array=cvals, cmap=cmap, linewidths=lw)
    ax.add_collection(lc)
    ax.set_xlim(f_hz[0], f_hz[-1])
    ax.set_ylim(0, max(amp)*1.05 if amp.max()>0 else 1.0)
    return lc

# ========================= 主过程 =========================
def main():
    # 1) 取一个标签==6的样本
    Xtr = np.load(os.path.join(DATA_DIR, "train_data_preprocessed.npy"))  # (N, L) 或 (N,1,L)
    ytr = np.load(os.path.join(DATA_DIR, "train_labels.npy"))
    idxs = np.where(ytr == LABEL_TGT)[0]
    if len(idxs) == 0:
        raise RuntimeError(f"找不到标签=={LABEL_TGT}的样本。")
    x = Xtr[idxs[0]]
    if x.ndim == 2:        # (1,L) -> (L,)
        x = x[0]
    x = x.astype(np.float64)

    # 2) 样本幅度谱与归一化频率
    f_hz, amp = fft_mag_1d(x, FS, nfft=None)        # 0..fs/2
    f_norm = f_hz / (FS/2)                          # 0..0.5 归一化至Nyquist

    # 3) 处理每个模型
    model_data = {}  # 存储所有模型的数据
    
    for model_name, config in MODEL_CONFIGS.items():
        print(f"\n===== {model_name} =====")
        try:
            # 加载模型权重
            sd = load_checkpoint_state_dict(config["ckpt_path"])
            key, w_after = pick_first_conv_weight(sd, prefer=config["prefer_keys"])
            print(f"[Info] Using first conv: {key}, weight shape={tuple(w_after.shape)}")
            K, C, L = tuple(w_after.shape)

            # 可选：门控/注意力向量（若checkpoint里有，且大小匹配K）
            gate = None
            for k,v in sd.items():
                if isinstance(v, torch.Tensor) and v.ndim==1 and v.numel()==K:
                    gate = v.detach().cpu().numpy().astype(np.float64)
                    # 归一化到[0,1]
                    gate -= gate.min()
                    gate /= (gate.max()+1e-12)
                    break

            # "训练后"的频段注意力
            A_after = kernels_attention_curve(w_after, nfft= max(4096, 1<<int(np.ceil(np.log2(len(x))))) , use_gate=gate)

            # "训练前"：与训练后同形状的随机卷积核
            torch.manual_seed(0)  # 为可复现
            w_before = torch.randn_like(w_after) * 0.1
            A_before = kernels_attention_curve(w_before, nfft= max(4096, 1<<int(np.ceil(np.log2(len(x))))) , use_gate=None)

            # 存储数据用于生成合并图表
            model_data[model_name] = {
                "A_before": A_before,
                "A_after": A_after
            }

            stats = {
                "Model": model_name,
                "Kernels": int(K),
                "Gate_used": gate is not None,
                "MeanAttn_before": float(np.mean(A_before)),
                "MeanAttn_after": float(np.mean(A_after))
            }
            print("[Stats]", json.dumps(stats, indent=2))
            
        except Exception as e:
            print(f"[Error] Failed to process {model_name}: {e}")
            continue
            
    # 生成合并的图表
    if model_data:
        try:
            base_path = generate_combined_attention_plot(model_data, x, f_norm, amp)
            print(f"\n[Success] Generated combined plot: {base_path}.svg/.png")
        except Exception as e:
            print(f"[Error] Failed to generate combined plot: {e}")

if __name__ == "__main__":
    main()
