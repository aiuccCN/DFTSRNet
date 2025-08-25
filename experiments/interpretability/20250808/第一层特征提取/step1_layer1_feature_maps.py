# -*- coding: utf-8 -*-
"""
Step 1: Layer-1 frequency response heatmaps (VKCNN Fig.14 style)
- Reads trained checkpoints and computes the frequency response of the FIRST Conv1d layer.
- Saves vector figures (.svg/.eps) with journal layout: width=9 cm, font=Times New Roman, fontsize=10.
- Outputs to: E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\新版可解释实验\20250808\第一层特征提取\结果
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # 防止 OpenMP 冲突报错

import sys
import math
import json
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ------------------ 固定版式（8 cm 宽、6.2 cm 高、Times New Roman、字号 12） ------------------
CM = 1/2.54
FIG_W = 8 * CM  # 宽度8cm
FIG_H = 6.2 * CM  # 高度6.2cm
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 16,
    "svg.fonttype": "none",      # 保持文本为矢量字体
    "pdf.fonttype": 42,
})

# ------------------ 你的目录（硬编码） ------------------
MODELS_DIR = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\新版可解释实验\models"
OUT_DIR     = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\新版可解释实验\20250808\第一层特征提取\结果"
os.makedirs(OUT_DIR, exist_ok=True)
print("[Info] Using device: cuda" if torch.cuda.is_available() else "[Info] Using device: cpu")
print("[Out ]", OUT_DIR)

# ------------------ 要处理的模型与优先键名 ------------------
TARGETS = [
    dict(
        name="DFTSRNet",
        ckpt=os.path.join(MODELS_DIR, "DFTSRNet.pth"),
        prefer_keys=["fusion.conv1.weight", "conv1.weight"],  # DFTSRNet 常见第一层
    ),
    dict(
        name="WDCNN",
        ckpt=os.path.join(MODELS_DIR, "WDCNN.pth"),
        prefer_keys=["conv1.weight", "features.0.weight"],
    ),
    dict(
        name="Simple1DCNN",
        ckpt=os.path.join(MODELS_DIR, "1D-CNN.pth"),
        prefer_keys=["backbone.0.weight", "conv1.weight", "features.0.weight"],
    ),
]


# ------------------ 工具函数 ------------------
def _load_state_dict(ckpt_path):
    """尽可能鲁棒地从 checkpoint 里取出 state_dict。返回 OrderedDict。"""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"[Debug] Loading checkpoint from: {ckpt_path}")

    obj = torch.load(ckpt_path, map_location="cpu")
    print(f"[Debug] Loaded object type: {type(obj)}")

    # 常见几种格式
    if isinstance(obj, dict):
        print(f"[Debug] Dictionary keys: {list(obj.keys())[:10]}")
        # 1) 常见 key
        for k in ["state_dict", "model_state_dict", "net", "model", "ema_state_dict"]:
            if k in obj and isinstance(obj[k], (dict, torch.nn.modules.module.Module)):
                sd = obj[k]
                if isinstance(sd, torch.nn.modules.module.Module):
                    sd = sd.state_dict()
                print(f"[Debug] Found key '{k}' with type: {type(sd)}")
                return sd
        # 2) 直接是 state_dict
        # 字典里 value 多为 tensor 的话也当作 state_dict
        tensors = [isinstance(v, torch.Tensor) for v in obj.values()]
        if len(tensors) > 0 and sum(tensors) / len(tensors) > 0.5:
            print("[Debug] Dict looks like raw state_dict.")
            return obj
    elif isinstance(obj, torch.nn.modules.module.Module):
        print("[Debug] Checkpoint is a torch.nn.Module, using state_dict().")
        return obj.state_dict()

    raise RuntimeError("Unrecognized checkpoint format; add more branches if needed.")


def _find_first_conv_weight(state_dict, prefer_keys):
    """在 state_dict 里找第一层 Conv1d 的权重张量 (out_ch, in_ch, kL)。"""
    # 先按优先表找
    for k in prefer_keys:
        if k in state_dict and state_dict[k].ndim == 3:
            w = state_dict[k]
            return k, w
    # 否则扫描所有 3D 权重，选一个“最像第一层”的
    candidates = []
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.ndim == 3 and v.shape[1] >= 1:
            candidates.append((k, v))
    if not candidates:
        raise RuntimeError("No 3D Conv1d-like weight found in state_dict.")
    # 简单规则：按名字长度与字典序优先，常能拿到靠前的层
    candidates.sort(key=lambda kv: (len(kv[0]), kv[0]))
    return candidates[0]


def _freq_response_from_weight(w: torch.Tensor, nfft: int = 4096):
    """
    从 (out_ch, in_ch, kL) 的卷积核权重计算幅频响应 |H(f)|
    返回：freq(0..0.5), spec(out_ch, nfft//2+1)
    """
    w = w.detach().cpu().numpy()  # (O, I, L)
    O, I, L = w.shape
    spec = np.zeros((O, nfft // 2 + 1), dtype=np.float64)

    for o in range(O):
        pwr = np.zeros(nfft // 2 + 1, dtype=np.float64)
        for i in range(I):
            fft = np.fft.rfft(w[o, i, :], n=nfft)
            pwr += (np.abs(fft) ** 2)  # 功率叠加
        mag = np.sqrt(pwr)
        spec[o] = mag

    # 全局归一化（避免不同模型色标不一致）
    spec /= (np.max(spec) + 1e-12)

    # 按峰值频率排序（可读性更强）
    peak_idx = np.argmax(spec, axis=1)
    order = np.argsort(peak_idx)
    spec = spec[order, :]

    freq = np.linspace(0.0, 0.5, spec.shape[1])  # 归一化到 Nyquist=0.5
    return freq, spec


def _merge_kernels(spec, model_name, merge_factor=4):
    """
    对指定模型的kernel进行合并
    spec: (K, F) 频谱矩阵
    model_name: 模型名称
    merge_factor: 合并因子，每几个kernel合并为1个
    """
    if model_name != "DFTSRNet":
        return spec
    
    K, F = spec.shape
    # 计算合并后的kernel数量
    merged_K = (K + merge_factor - 1) // merge_factor  # 向上取整
    merged_spec = np.zeros((merged_K, F))
    
    for i in range(merged_K):
        start_idx = i * merge_factor
        end_idx = min((i + 1) * merge_factor, K)
        # 对每组kernel取最高值（最大值）
        merged_spec[i] = np.max(spec[start_idx:end_idx], axis=0)
    
    print(f"[Info] {model_name}: 原始kernel数量 {K} -> 合并后 {merged_K} (每{merge_factor}个取最高值)")
    return merged_spec


def _plot_combined_and_save(results, out_dir):
    """
    将三个模型的热图合并到一张图中，采用1x3子图布局
    results: list of (model_name, freq, spec) tuples
    """
    # 创建合并图：1行3列
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W * 3, FIG_H), constrained_layout=True)
    
    labels = ['(a)', '(b)', '(c)']
    
    for i, (model_name, freq, spec) in enumerate(results):
        ax = axes[i]
        
        # 对DFTSRNet进行kernel合并
        spec = _merge_kernels(spec, model_name, merge_factor=4)
        
        # 避免 1 行时 y 轴 singular 的警告：给 imshow 明确 extent
        K, F = spec.shape
        extent = [freq[0], freq[-1], 0.5, K + 0.5]

        im = ax.imshow(
            spec,
            aspect="auto",
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap="viridis",
            vmin=0.0, vmax=1.0
        )
        ax.set_xlabel("Normalized frequency")
        ax.set_ylabel("Kernel")
        
        # 添加标注到图表左上角外侧
        ax.text(-0.1, 1.05, labels[i], transform=ax.transAxes, fontsize=12,
                verticalalignment='bottom', horizontalalignment='left', color='black')

        # 纵轴刻度（最多 8 个）
        if K <= 8:
            yticks = np.arange(1, K + 1)
        else:
            step = max(1, K // 8)
            yticks = np.arange(1, K + 1, step)
        ax.set_yticks(yticks)

        # 只在最右边的子图添加colorbar
        if i == len(results) - 1:
            cbar = fig.colorbar(im, ax=ax)
    
    # 保存合并图
    base = os.path.join(out_dir, "Combined_Layer1_FeatureMaps")
    fig.savefig(base + ".svg", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Save] {os.path.basename(base)}.svg, {os.path.basename(base)}.png")


def main():
    results = []  # 收集所有模型的结果
    
    for cfg in TARGETS:
        name = cfg["name"]
        ckpt = cfg["ckpt"]
        prefs = cfg["prefer_keys"]

        print(f"\n===== {name} =====")
        try:
            sd = _load_state_dict(ckpt)
        except Exception as e:
            print(f"[Error] Failed to load state_dict for {name}: {e}")
            continue

        try:
            key, w = _find_first_conv_weight(sd, prefs)
            print(f"[Pick] use weight: {key} (shape={tuple(w.shape)})")
            freq, spec = _freq_response_from_weight(w, nfft=4096)
            print(f"[Spec] Heatmap shape: {spec.shape}, freq in [{freq[0]:.3f},{freq[-1]:.3f}]")
            results.append((name, freq, spec))
        except Exception as e:
            print(f"[Error] Failed to process {name}: {e}")
    
    # 生成合并图
    if len(results) == 3:
        _plot_combined_and_save(results, OUT_DIR)
        print("\n[Done] Step 1 complete. Combined figure saved to:", OUT_DIR)
    else:
        print(f"\n[Warning] Expected 3 models but got {len(results)}. No combined figure generated.")


if __name__ == "__main__":
    main()
