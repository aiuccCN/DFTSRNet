# -*- coding: utf-8 -*-
"""
Step 2: Kernel Gini Index (VKCNN Table-6 style)
- 从 checkpoint 中读取第一层 Conv1d 的权重，计算各 kernel 的 |H(f)| 幅频响应；
- 在频率轴上对每个 kernel 的归一化谱计算 Gini 指数；最后对所有 kernel 取均值；
- 可与现有的 L1/L2 结果合并，导出为一张表。
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 防止 OpenMP 冲突

import json
import csv
import math
import torch
import numpy as np
from collections import OrderedDict

# ------------------ 硬编码路径（按需修改） ------------------
MODELS_DIR = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\新版可解释实验\models"
OUT_DIR    = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\新版可解释实验\20250808\基尼"
os.makedirs(OUT_DIR, exist_ok=True)

# 如果你已有一个 L1/L2 的 CSV，按下行填写；没有就置为 None
EXISTING_L1L2_CSV = None  # e.g., r"E:\...\L1L2_results.csv"

TARGETS = [
    dict(
        name="DFTSRNet",
        ckpt=os.path.join(MODELS_DIR, "DFTSRNet.pth"),
        prefer_keys=["fusion.conv1.weight", "conv1.weight", "features.0.weight"],
    ),
    dict(
        name="WDCNN",
        ckpt=os.path.join(MODELS_DIR, "WDCNN.pth"),
        prefer_keys=["conv1.weight", "features.0.weight"],
    ),
    dict(
        name="1D-CNN",
        ckpt=os.path.join(MODELS_DIR, "1D-CNN.pth"),
        prefer_keys=["backbone.0.weight", "conv1.weight", "features.0.weight"],
    ),
]

def load_state_dict_robust(ckpt_path):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu")
    # 常见结构
    if isinstance(obj, dict):
        for k in ["model_state_dict", "state_dict", "net", "model", "ema_state_dict"]:
            if k in obj:
                sd = obj[k]
                if isinstance(sd, torch.nn.Module):
                    sd = sd.state_dict()
                return sd
        # 直接是 state_dict
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    if isinstance(obj, torch.nn.Module):
        return obj.state_dict()
    raise RuntimeError(f"Unrecognized checkpoint format: {ckpt_path}")

def find_first_conv_weight(sd: OrderedDict, prefer_keys):
    # 优先按名单找
    for k in prefer_keys:
        if k in sd and isinstance(sd[k], torch.Tensor) and sd[k].ndim == 3:
            return k, sd[k]
    # 否则找第一个 3D 权重
    cands = [(k, v) for k, v in sd.items() if isinstance(v, torch.Tensor) and v.ndim == 3]
    if not cands:
        raise RuntimeError("No 3D Conv1d-like weight in state_dict.")
    # 简单排序挑一个“最像第一层”的
    cands.sort(key=lambda kv: (len(kv[0]), kv[0]))
    return cands[0]

def rfft_magnitude(t: np.ndarray, nfft=4096):
    # 返回正频率幅度谱 |X(f)|, 长度 nfft//2+1
    spec = np.abs(np.fft.rfft(t, n=nfft))
    return spec

def gini_from_prob(p: np.ndarray):
    """
    计算频谱基尼指数，带小样本修正
    p: 已经归一化的概率分布 (sum=1)
    返回: 修正后的Gini指数 [0,1]
    """
    p = np.asarray(p, dtype=np.float64).clip(min=0)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    n = p.size
    p_sorted = np.sort(p)  # 升序
    
    # 经典离散Gini公式：1 - 2*Σ((n-i+0.5)/n * p_sorted[i])
    i = np.arange(1, n + 1, dtype=np.float64)
    g = 1.0 - 2.0 * np.sum((n - i + 0.5) / n * p_sorted)
    
    # 小样本修正（让不同频率bins数更可比）
    if n > 1:
        g = g * (n / (n - 1.0))
    
    return float(max(0.0, min(1.0, g)))

def spectral_gini_for_layer(w: torch.Tensor, nfft=4096, drop_dc_bins=1):
    """
    计算每个卷积核的频谱基尼指数，然后取平均值和标准差
    输入 w: (out_ch, in_ch, kL)
    - 对每个输出通道计算综合频谱幅度
    - 对每个核的频谱分布单独计算基尼指数
    - 返回: (平均Gini, 标准差Gini, 每个kernel的Gini列表)
    """
    W = w.detach().cpu().numpy()  # (O, I, L)
    O, I, L = W.shape
    ginis = []
    
    for o in range(O):
        # 计算该输出通道的综合频谱功率
        pwr = np.zeros(nfft // 2 + 1, dtype=np.float64)
        for i in range(I):
            fft = np.fft.rfft(W[o, i, :], n=nfft)
            pwr += (np.abs(fft) ** 2)
        
        # 转换为幅度谱
        mag = np.sqrt(pwr)
        
        # 去掉直流分量
        if drop_dc_bins > 0:
            mag = mag[drop_dc_bins:]
        
        # 归一化为概率分布并计算基尼指数
        if mag.sum() <= 0:
            ginis.append(0.0)
        else:
            p = mag / mag.sum()
            ginis.append(gini_from_prob(p))
    
    ginis = np.asarray(ginis)
    mean_gini = float(ginis.mean())
    std_gini = float(ginis.std())
    
    return mean_gini, std_gini, ginis.tolist()

def main():
    rows = []
    for cfg in TARGETS:
        name = cfg["name"]
        ckpt = cfg["ckpt"]
        try:
            sd = load_state_dict_robust(ckpt)
            key, w = find_first_conv_weight(sd, cfg["prefer_keys"])
            g_mean, g_std, g_each = spectral_gini_for_layer(
                w, nfft=4096, drop_dc_bins=1
            )
            print(f"[{name}] first-layer: {key} -> Spectral Gini = {g_mean:.4f} ± {g_std:.4f} "
                  f"(K={len(g_each)})")
            rows.append(dict(Model=name, Gini=g_mean, Gini_Std=g_std))
        except Exception as e:
            print(f"[{name}] ERROR: {e}")
            rows.append(dict(Model=name, Gini=np.nan))

    # 写出 Gini 独立 CSV
    gini_csv = os.path.join(OUT_DIR, "Table_Step2_KernelGini.csv")
    with open(gini_csv, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["Model", "Spectral Gini (Mean)", "Spectral Gini (Std)"])
        for r in rows:
            mean_val = f"{r['Gini']:.4f}" if np.isfinite(r.get('Gini', np.nan)) else "nan"
            std_val = f"{r.get('Gini_Std', 0):.4f}" if np.isfinite(r.get('Gini_Std', np.nan)) else "nan"
            wcsv.writerow([r["Model"], mean_val, std_val])
    print("[Save]", gini_csv)

    # 若你有 L1/L2 的 CSV，这里可合并为表6
    if EXISTING_L1L2_CSV and os.path.isfile(EXISTING_L1L2_CSV):
        # 读 L1/L2
        m2l = {}
        with open(EXISTING_L1L2_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 你的列名若不同，改这里
                model = row.get("Method") or row.get("Model")
                if not model: 
                    continue
                m2l[model] = row

        # 合并
        out_csv = os.path.join(OUT_DIR, "Table_Step2_Gini_and_L1L2.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["Method", "Spectral Gini (Mean±Std)", "l1", "l2", "l1/l2"])
            for r in rows:
                m = r["Model"]
                g = r["Gini"]
                g_std = r.get("Gini_Std", 0)
                gini_str = f"{g:.4f}±{g_std:.4f}" if np.isfinite(g) and np.isfinite(g_std) else "nan"
                if m in m2l:
                    l1 = m2l[m].get("l1", "")
                    l2 = m2l[m].get("l2", "")
                    ratio = m2l[m].get("l1/l2", "")
                else:
                    l1 = l2 = ratio = ""
                wcsv.writerow([m, gini_str, l1, l2, ratio])
        print("[Save]", out_csv)

if __name__ == "__main__":
    main()
