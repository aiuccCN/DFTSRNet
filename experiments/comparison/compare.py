# compare.py  (overwrite existing file)
# ----------------------------------------------------------
# Compare 6 models (without ResNet50) performance under seeds 42 / 97 / 20
# Generate acc_summary_seed*.csv and acc_summary_mean_std.csv
# Save each trained model to specified directory
# Add training curve plotting functionality
# ----------------------------------------------------------
import os, csv, warnings, random
warnings.filterwarnings("ignore")
import numpy as np, matplotlib.pyplot as plt, torch, pywt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from dataset_loader import get_dataloaders
from train_utils    import train_epoch, eval_epoch
from models_lib     import CNN1D, WDCNN, InceptionTime1D, VMD_CNN
from model          import DFTSRNet
from config         import Config
device = Config.DEVICE

BASE_PATH = Config.BASE_DATA_PATH   # 根目录由 config.py 提供
DATASETS = {d:os.path.join(BASE_PATH,f'Dataset_{d}') for d in ['A','B','C']}

# Output directory
OUTPUT_DIR = r'e:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\对比实验\对比试验结果'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model save directory
MODEL_DIR = os.path.join(OUTPUT_DIR, 'saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- Set random seed ----------------
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------- SVM + Wavelet ----------------
def wp_feat(sig, wav='db4', lvl=3):
    wp = pywt.WaveletPacket(sig, wav, 'symmetric', maxlevel=lvl)
    return np.array([np.sqrt(np.sum(n.data**2)) for n in wp.get_level(lvl,'freq')])

def run_svm(root, model_name, dataset_name, seed):
    tr_x = np.load(os.path.join(root,'train_data_preprocessed.npy'))
    va_x = np.load(os.path.join(root,'val_data_preprocessed.npy'))
    te_x = np.load(os.path.join(root,'test_data_preprocessed.npy'))
    tr_y = np.load(os.path.join(root,'train_labels.npy'))
    va_y = np.load(os.path.join(root,'val_labels.npy'))
    te_y = np.load(os.path.join(root,'test_labels.npy'))
    tr_f = np.array([wp_feat(x) for x in tr_x])
    va_f = np.array([wp_feat(x) for x in va_x])
    te_f = np.array([wp_feat(x) for x in te_x])
    best_c = max([0.1,1,10], key=lambda c:accuracy_score(va_y,SVC(C=c,gamma='auto').fit(tr_f,tr_y).predict(va_f)))
    clf = SVC(C=best_c,gamma='auto').fit(np.vstack([tr_f,va_f]), np.hstack([tr_y,va_y]))
    
    # 保存SVM模型
    import joblib
    model_path = os.path.join(MODEL_DIR, f'{model_name}_{dataset_name}_seed{seed}.pkl')
    joblib.dump(clf, model_path)
    print(f"  -> 保存SVM模型: {model_path}")
    
    return accuracy_score(te_y, clf.predict(te_f))

# ---------------- 深度模型 ----------------
def run_deep(model, loaders, epochs=200, model_name=None, dataset_name=None, seed=None, return_curves=False):
    tr_loader, va_loader, te_loader = loaders
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val, best_state = 0, None
    
    # 用于记录训练曲线
    acc_history = []
    loss_history = []
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, tr_loader, opt, device)
        _, val_acc = eval_epoch(model, va_loader, device)
        
        # 记录训练曲线数据
        if return_curves:
            acc_history.append(val_acc)
            loss_history.append(train_loss)
        
        if val_acc > best_val:
            best_val, best_state = val_acc, model.state_dict()
    
    model.load_state_dict(best_state)
    _, test_acc = eval_epoch(model, te_loader, device)
    
    # 保存深度学习模型
    if model_name and dataset_name and seed is not None:
        model_path = os.path.join(MODEL_DIR, f'{model_name}_{dataset_name}_seed{seed}.pth')
        torch.save({
            'model_state_dict': best_state,
            'model_name': model_name,
            'dataset': dataset_name,
            'seed': seed,
            'test_accuracy': test_acc,
            'val_accuracy': best_val
        }, model_path)
        print(f"  -> 保存模型: {model_path}")
    
    if return_curves:
        return test_acc, acc_history, loss_history
    return test_acc

# ---------------- 模型列表 ----------------
MODELS = {
    'SVM+Wavelet': None,
    '1D-CNN'     : CNN1D,
    'WDCNN'      : WDCNN,
    'Inception'  : InceptionTime1D,
    'VMD+CNN'    : VMD_CNN,
    'DFTSRNet'   : lambda: DFTSRNet(num_classes=10)
}

# ---------------- 主循环 ----------------
all_runs = {}           # {seed:{dataset:{model:acc}}}
all_curves = {}         # {dataset:{model:[acc_curves, loss_curves]}}

for seed in [42,97,20]:
    print(f"\n==========  RANDOM SEED {seed}  ==========")
    set_seed(seed)
    acc_res = {ds:{} for ds in DATASETS.keys()}
    
    # 只在第一个种子时收集训练曲线数据（用于绘图）
    collect_curves = (seed == 42)
    if collect_curves:
        all_curves = {ds:{} for ds in DATASETS.keys()}
    
    for ds, root in DATASETS.items():
        loaders = get_dataloaders(root, batch_size=128)
        for name, ctor in MODELS.items():
            print(f"[Seed {seed}] Dataset {ds} | {name}")
            if name=='SVM+Wavelet':
                acc = run_svm(root, name, ds, seed)
                if collect_curves:
                    # SVM没有训练曲线，用固定值填充
                    all_curves[ds][name] = ([acc] * 200, [0.1] * 200)
            else:
                if collect_curves:
                    acc, acc_hist, loss_hist = run_deep(ctor(), loaders, epochs=200, 
                                                       model_name=name, dataset_name=ds, 
                                                       seed=seed, return_curves=True)
                    all_curves[ds][name] = (acc_hist, loss_hist)
                else:
                    acc = run_deep(ctor(), loaders, epochs=200, model_name=name, 
                                 dataset_name=ds, seed=seed)
            acc_res[ds][name] = acc
            print(f"  Test Acc = {acc:.4f}")
    
    all_runs[seed] = acc_res
    # -------- 保存单次 CSV --------
    out_csv = os.path.join(OUTPUT_DIR, f'acc_summary_seed{seed}.csv')
    with open(out_csv,'w',newline='') as f:
        w = csv.writer(f); w.writerow(['Dataset']+list(MODELS.keys()))
        for ds in ['A','B','C']:
            w.writerow([ds]+[f"{acc_res[ds][m]:.4f}" for m in MODELS])
    print(f"  -> Saved {out_csv}")

# -------- 绘制训练曲线图 --------
for ds in ['A', 'B', 'C']:
    if ds in all_curves and all_curves[ds]:
        acc_plot = {m: curves[0] for m, curves in all_curves[ds].items()}
        loss_plot = {m: curves[1] for m, curves in all_curves[ds].items()}
        
        xs = np.arange(1, 201)
        palette = {  # User template RGB
            'SVM+Wavelet': (144/255, 201/255, 231/255),
            '1D-CNN': (33/255, 158/255, 188/255),
            'WDCNN': (19/255, 103/255, 131/255),
            'Inception': (2/255, 48/255, 74/255),
            'VMD+CNN': (254/255, 183/255, 5/255),
            'DFTSRNet': 'red'
        }
        
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for m, c in acc_plot.items():
            style = {'lw': 4, 'color': palette[m]} if m == 'DFTSRNet' else {'lw': 1.5, 'color': palette[m]}
            ax[0].plot(xs, c, label=m, **style)
        ax[0].set_ylabel('Accuracy')
        ax[0].legend(loc='lower right')
        
        for m, l in loss_plot.items():
            style = {'lw': 4, 'color': palette[m]} if m == 'DFTSRNet' else {'lw': 1.5, 'color': palette[m]}
            ax[1].plot(xs, l, **style)
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Loss')
        
        plt.tight_layout()
        curve_path = os.path.join(OUTPUT_DIR, f'curves_Dataset{ds}_ALL.png')
        plt.savefig(curve_path, dpi=300)
        plt.close()
        print(f"  -> Saved training curve plot: {curve_path}")

# -------- 计算均值和标准差 --------
mean_std = {ds:{m:[] for m in MODELS} for ds in DATASETS}
for seed,res in all_runs.items():
    for ds in DATASETS:
        for m in MODELS:
            mean_std[ds][m].append(res[ds][m])

# -------- 保存汇总 CSV --------
final_csv = os.path.join(OUTPUT_DIR, 'acc_summary_mean_std.csv')
with open(final_csv,'w',newline='') as f:
    w = csv.writer(f); w.writerow(['Dataset']+[f"{m}_mean" for m in MODELS]+[f"{m}_std" for m in MODELS])
    for ds in ['A','B','C']:
        means=[np.mean(mean_std[ds][m]) for m in MODELS]
        stds =[np.std(mean_std[ds][m])  for m in MODELS]
        w.writerow([ds]+[f"{v:.4f}" for v in means]+[f"{v:.4f}" for v in stds])
print(f"\nAll completed: Generated single-seed result files and summary files")
print(f"Results saved in: {OUTPUT_DIR}")
print(f"Models saved in: {MODEL_DIR}")

# -------- 生成模型保存信息汇总 --------
model_info_path = os.path.join(OUTPUT_DIR, 'saved_models_info.txt')
with open(model_info_path, 'w', encoding='utf-8') as f:
    f.write("Saved Model Files Information\n")
    f.write("=" * 50 + "\n\n")
    f.write("File naming convention: {model_name}_{dataset}_{seed}.{extension}\n")
    f.write("- Deep learning models: .pth format\n")
    f.write("- SVM models: .pkl format\n\n")

    f.write("List of saved models:\n")
    for seed in [42, 97, 20]:
        f.write(f"\nSeed {seed}:\n")
        for ds in ['A', 'B', 'C']:
            f.write(f"  Dataset {ds}:\n")
            for model_name in MODELS.keys():
                if model_name == 'SVM+Wavelet':
                    f.write(f"    - {model_name}_{ds}_seed{seed}.pkl\n")
                else:
                    f.write(f"    - {model_name}_{ds}_seed{seed}.pth\n")

print(f"Model information summary saved in: {model_info_path}")