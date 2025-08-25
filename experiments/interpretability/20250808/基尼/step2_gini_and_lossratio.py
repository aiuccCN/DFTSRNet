# -*- coding: utf-8 -*-
import os, json, math
import numpy as np
import torch
import torch.nn.functional as F

# ======= 你的路径（自行确认）=======
DATA_DIR   = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\数据集\西储大学实验数据集\Dataset_A"
MODELS_DIR = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\新版可解释实验\models"
OUT_DIR    = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\新版可解释实验\20250808\基尼"
os.makedirs(OUT_DIR, exist_ok=True)
N_PER_CLASS = 20            # 每类取多少条做统计
FS = 12000                  # 采样率，用于频轴注释
NFFT = 1024                 # Welch NFFT
SEG = 256                   # Welch segment length
HOP = 128                   # Welch hop

TARGETS = [
    dict(name="DFTSRNet", ckpt=os.path.join(MODELS_DIR, "DFTSRNet.pth"),
         prefer=["fusion.conv1.weight","conv1.weight","backbone.0.weight"]),
    dict(name="WDCNN",    ckpt=os.path.join(MODELS_DIR, "WDCNN.pth"),
         prefer=["conv1.weight","features.0.weight","backbone.0.weight"]),
    dict(name="1D-CNN",   ckpt=os.path.join(MODELS_DIR, "1D-CNN.pth"),
         prefer=["backbone.0.weight","conv1.weight","features.0.weight"]),
]

def load_np(path): 
    x = np.load(path)
    return x

def balanced_subset(X, y, per_cls):
    cls = np.unique(y)
    idx = []
    for c in cls:
        I = np.where(y==c)[0]
        I = I[:per_cls] if len(I)>=per_cls else I
        idx.extend(I.tolist())
    idx = np.array(idx, dtype=int)
    return X[idx], y[idx]

def robust_load_state_dict(p):
    obj = torch.load(p, map_location="cpu")
    if isinstance(obj, dict):
        for k in ["state_dict","model_state_dict","net","model","ema_state_dict"]:
            if k in obj and isinstance(obj[k], (dict,)):
                return obj[k], obj
        # raw state_dict?
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj, {}
    raise RuntimeError("unsupported ckpt format: "+str(type(obj)))

def find_first_conv(state_dict, prefer):
    for k in prefer:
        if k in state_dict and state_dict[k].ndim==3:
            b = state_dict.get(k.replace("weight","bias"), None)
            return k, state_dict[k], b
    # fallback: first 3D weight
    for k,v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.ndim==3 and v.shape[1]>=1:
            b = state_dict.get(k.replace("weight","bias"), None)
            return k, v, b
    raise RuntimeError("no Conv1d-like weight found")

def conv1d_same(x, w, b=None):
    # x: (N,1,L), w:(K,1,ks), b:(K,)
    ks = w.shape[-1]
    pad = ks//2
    y = F.conv1d(x, w, bias=b, stride=1, padding=pad)
    return y   # (N,K,L)

def welch_psd(sig, nfft=1024, seg=256, hop=128):
    # sig: (L,) numpy
    L = len(sig)
    if L < seg: 
        seg = L
        hop = max(1, seg//2)
    win = np.hanning(seg)
    denom = (win**2).sum()
    acc = None; nseg=0
    for st in range(0, L-seg+1, hop):
        s = sig[st:st+seg]*win
        S = np.fft.rfft(s, n=nfft)
        P = (np.abs(S)**2)/denom
        acc = P if acc is None else acc+P
        nseg += 1
    if nseg==0:
        S = np.fft.rfft(sig*np.hanning(L), n=nfft)
        P = (np.abs(S)**2)/max(1,(np.hanning(L)**2).sum())
        acc = P; nseg=1
    return acc / nseg  # (nfft//2+1,)

def gini_from_prob(p):
    # p >=0, sum p = 1
    p = np.asarray(p, dtype=np.float64)
    if p.sum() <= 0: 
        return 0.0
    p = p / p.sum()
    n = p.size
    ps = np.sort(p)
    # Gini = 1 - 2 * sum_{i=1}^n (n+1-i)*p_(i) / n
    ranks = np.arange(1, n+1)
    G = 1.0 - 2.0*np.sum((n+1-ranks)*ps)/n
    return max(0.0, min(1.0, float(G)))

def spectrum_gini_from_activ(Y):
    # Y: (N,K,L) torch tensor on cpu
    Y = Y.detach().cpu().numpy()
    N,K,L = Y.shape
    g = []
    for k in range(K):
        # 聚合所有样本得到“代表性”PSD
        acc = None
        for n in range(N):
            P = welch_psd(Y[n,k,:], nfft=NFFT, seg=SEG, hop=HOP)
            acc = P if acc is None else acc+P
        Pk = acc / N
        # 频率归一化成概率
        g.append(gini_from_prob(Pk))
    g = np.array(g)
    return g.mean(), g.std(), g  # mean±std + per-kernel list

def try_load_history(model_name):
    # 可扩展：history_{name}.json / npz / txt
    cand = [os.path.join(MODELS_DIR, f"history_{model_name}.json"),
            os.path.join(MODELS_DIR, f"{model_name}_history.json"),
            os.path.join(MODELS_DIR, f"history_{model_name}.npz")]
    for p in cand:
        if os.path.isfile(p) and p.endswith(".json"):
            with open(p,"r",encoding="utf-8") as f:
                H=json.load(f); 
            return H
        if os.path.isfile(p) and p.endswith(".npz"):
            H=np.load(p, allow_pickle=True)
            return {k:H[k].tolist() for k in H.files}
    return None

def last10_ratio(train_losses, val_losses):
    t = np.mean(train_losses[-10:]) if len(train_losses)>0 else np.nan
    v = np.mean(val_losses[-10:])   if len(val_losses)>0 else np.nan
    # 两种口径都给
    sym = max(t,v)/max(1e-12,min(t,v))
    val_over_train = v/max(1e-12,t)
    return t, v, sym, val_over_train

def main():
    # 读数据
    X = load_np(os.path.join(DATA_DIR,"test_data_preprocessed.npy"))
    y = load_np(os.path.join(DATA_DIR,"test_labels.npy"))
    # 取平衡子集
    Xb, yb = balanced_subset(X, y, N_PER_CLASS)  # (M,1024)
    Xb = torch.from_numpy(Xb).float().unsqueeze(1)  # (M,1,L)
    device = torch.device("cpu")  # 只做第一层，CPU就够了
    Xb = Xb.to(device)

    rows = []
    for cfg in TARGETS:
        name = cfg["name"]; ckpt = cfg["ckpt"]
        try:
            sd, meta = robust_load_state_dict(ckpt)
            key,w,b = find_first_conv(sd, cfg["prefer"])
            if b is not None: b = b.to(device)
            w = w.to(device)
            # SAME 卷积得到第一层输出
            Y = conv1d_same(Xb, w, b)  # (M,K,L)
            # 频谱基尼（特征图口径）
            g_mean, g_std, g_list = spectrum_gini_from_activ(Y)
            # 训练曲线（可选）
            H = try_load_history(name)
            if H is not None and ("train_loss" in H and "val_loss" in H):
                l1, l2, lr_sym, lr_vt = last10_ratio(np.array(H["train_loss"],dtype=float),
                                                     np.array(H["val_loss"],dtype=float))
            else:
                # 如果没有历史，就从 ckpt meta 里尝试（你也可以手填）
                l1 = float(meta.get("Last10_Train_Loss", np.nan))
                l2 = float(meta.get("Last10_Val_Loss",   np.nan))
                if np.isfinite(l1) and np.isfinite(l2):
                    _,_,lr_sym,lr_vt = last10_ratio(np.array([l1]), np.array([l2]))
                else:
                    lr_sym = np.nan; lr_vt = np.nan

            rows.append(dict(
                Method=name,
                Gini_mean=float(g_mean),
                Gini_std=float(g_std),
                L1=float(l1), L2=float(l2),
                L1overL2_sym=float(lr_sym),
                ValOverTrain=float(lr_vt),
                Kernels=int(Y.shape[1])
            ))
            print(f"[{name}] Gini={g_mean:.4f}±{g_std:.4f}, "
                  f"L1={l1:.4f}, L2={l2:.4f}, "
                  f"ratio_sym={lr_sym:.4f}, val/train={lr_vt:.4f}, K={Y.shape[1]}")
        except Exception as e:
            print(f"[{name}] ERROR: {e}")

    # 保存表格
    import csv
    out_csv = os.path.join(OUT_DIR, "Table_Step2_Gini_LossRatio.csv")
    with open(out_csv,"w",newline="",encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)
    print("[Save]", out_csv)

if __name__=="__main__":
    main()
