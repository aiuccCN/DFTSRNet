# -*- coding: utf-8 -*-
"""
Model Comparison Experiments under Noisy Environments
Testing multiple model performance on full noisy datasets at different SNR levels
"""
import os, csv, warnings, random, time
warnings.filterwarnings("ignore")
import numpy as np, matplotlib.pyplot as plt, torch, pywt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

from dataset import get_dataloaders
from train_utils import train_epoch, eval_epoch
from models_lib import CNN1D, WDCNN, InceptionTime1D, VMD_CNN
from model import DFTSRNet
from config import Config

device = Config.DEVICE

# Noisy dataset paths
BASE_PATH = r"e:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\不同噪声水平实验"
NOISY_DATASETS = {
    'SNR_2dB': os.path.join(BASE_PATH, 'Dataset_B_Full_Noisy_2dB'),
    'SNR_0dB': os.path.join(BASE_PATH, 'Dataset_B_Full_Noisy_0dB'),
    'SNR_-2dB': os.path.join(BASE_PATH, 'Dataset_B_Full_Noisy_-2dB'),
    'SNR_-4dB': os.path.join(BASE_PATH, 'Dataset_B_Full_Noisy_-4dB')
}

# Output directory
OUTPUT_DIR = os.path.join(BASE_PATH, 'noisy_comparison_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model save directory
MODEL_DIR = os.path.join(OUTPUT_DIR, 'saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Visualization directory
VIS_DIR = os.path.join(OUTPUT_DIR, 'visualizations')
os.makedirs(VIS_DIR, exist_ok=True)

# Set random seed
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# SVM + Wavelet implementation
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
    
    print(f"    Extracting wavelet features...")
    tr_f = np.array([wp_feat(x) for x in tr_x])
    va_f = np.array([wp_feat(x) for x in va_x])
    te_f = np.array([wp_feat(x) for x in te_x])
    
    print(f"    Training SVM...")
    best_c = max([0.1,1,10], key=lambda c:accuracy_score(va_y,SVC(C=c,gamma='auto').fit(tr_f,tr_y).predict(va_f)))
    clf = SVC(C=best_c,gamma='auto').fit(np.vstack([tr_f,va_f]), np.hstack([tr_y,va_y]))
    
    # 保存SVM模型
    import joblib
    model_path = os.path.join(MODEL_DIR, f'{model_name}_{dataset_name}_seed{seed}.pkl')
    joblib.dump(clf, model_path)
    
    test_pred = clf.predict(te_f)
    test_acc = accuracy_score(te_y, test_pred)
    
    return test_acc, te_y, test_pred

# Deep model training
def run_deep(model, loaders, epochs=100, model_name=None, dataset_name=None, seed=None):
    tr_loader, va_loader, te_loader = loaders
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val, best_state = 0, None
    
    print(f"    Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, tr_loader, opt, device)
        val_loss, val_acc = eval_epoch(model, va_loader, device)
        
        if val_acc > best_val:
            best_val, best_state = val_acc, model.state_dict()
        
        if (epoch + 1) % 20 == 0:
            print(f"      Epoch {epoch+1}/{epochs}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
    
    model.load_state_dict(best_state)
    test_loss, test_acc = eval_epoch(model, te_loader, device)
    
    # 获取测试集预测结果
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in te_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
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
    
    return test_acc, np.array(all_labels), np.array(all_preds)

# Model definitions
MODELS = {
    'SVM+Wavelet': None,
    '1D-CNN': CNN1D,
    'WDCNN': WDCNN,
    'Inception': InceptionTime1D,
    
    'DFTSRNet': lambda: DFTSRNet(num_classes=10)
}

# Add VMD_CNN to model list if available
if VMD_CNN is not None:
    MODELS['VMD+CNN'] = VMD_CNN

print(f"Available models: {list(MODELS.keys())}")
print(f"Noisy datasets: {list(NOISY_DATASETS.keys())}")

# Main experiment loop
all_runs = {}  # {seed: {dataset: {model: acc}}}
all_predictions = {}  # {seed: {dataset: {model: (y_true, y_pred)}}}

for seed in [42]:
    print(f"\n{'='*60}")
    print(f"Random seed: {seed}")
    print(f"{'='*60}")
    
    set_seed(seed)
    acc_res = {ds: {} for ds in NOISY_DATASETS.keys()}
    pred_res = {ds: {} for ds in NOISY_DATASETS.keys()}
    
    for ds_name, root in NOISY_DATASETS.items():
        print(f"\nDataset: {ds_name}")
        print(f"Path: {root}")
        
        # Check if dataset exists
        if not os.path.exists(root):
            print(f"  Warning: Dataset path does not exist, skipping {ds_name}")
            continue
            
        loaders = get_dataloaders(root, batch_size=64)
        
        for model_name, model_ctor in MODELS.items():
            print(f"\n  Model: {model_name}")
            start_time = time.time()
            
            try:
                if model_name == 'SVM+Wavelet':
                    acc, y_true, y_pred = run_svm(root, model_name, ds_name, seed)
                else:
                    model = model_ctor()
                    acc, y_true, y_pred = run_deep(model, loaders, epochs=100, 
                                                  model_name=model_name, dataset_name=ds_name, seed=seed)
                
                acc_res[ds_name][model_name] = acc
                pred_res[ds_name][model_name] = (y_true, y_pred)
                
                elapsed = time.time() - start_time
                print(f"    Test accuracy: {acc:.4f} (Time: {elapsed:.1f}s)")
                
            except Exception as e:
                print(f"    Error: {str(e)}")
                acc_res[ds_name][model_name] = 0.0
                pred_res[ds_name][model_name] = (np.array([]), np.array([]))
    
    all_runs[seed] = acc_res
    all_predictions[seed] = pred_res
    
    # Save single seed results
    out_csv = os.path.join(OUTPUT_DIR, f'noisy_acc_summary_seed{seed}.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Dataset'] + list(MODELS.keys()))
        for ds in NOISY_DATASETS.keys():
            if ds in acc_res:
                w.writerow([ds] + [f"{acc_res[ds].get(m, 0.0):.4f}" for m in MODELS])
    print(f"\nSaved results: {out_csv}")

# Calculate statistical results
print(f"\n{'='*60}")
print("Calculating statistical results...")
print(f"{'='*60}")

mean_std = {ds: {m: [] for m in MODELS} for ds in NOISY_DATASETS}
for seed, res in all_runs.items():
    for ds in NOISY_DATASETS:
        if ds in res:
            for m in MODELS:
                if m in res[ds]:
                    mean_std[ds][m].append(res[ds][m])

# Save summary statistics
final_csv = os.path.join(OUTPUT_DIR, 'noisy_acc_summary_mean_std.csv')
with open(final_csv, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['Dataset'] + [f"{m}_mean" for m in MODELS] + [f"{m}_std" for m in MODELS])
    for ds in NOISY_DATASETS.keys():
        if ds in mean_std:
            means = [np.mean(mean_std[ds][m]) if mean_std[ds][m] else 0.0 for m in MODELS]
            stds = [np.std(mean_std[ds][m]) if mean_std[ds][m] else 0.0 for m in MODELS]
            w.writerow([ds] + [f"{v:.4f}" for v in means] + [f"{v:.4f}" for v in stds])

print(f"Saved summary results: {final_csv}")

# Generate visualizations
print("\nGenerating visualization charts...")

# Set matplotlib parameters
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

# Define color scheme
colors = {
    'SVM+Wavelet': '#1f77b4',  # Blue
    '1D-CNN': '#ff7f0e',       # Orange
    'WDCNN': '#2ca02c',        # Green
    'Inception': '#d62728',     # Red
    'VMD+CNN': '#9467bd',      # Purple
    'DFTSRNet': '#e377c2'      # Pink
}

# 1. Accuracy comparison bar chart
fig, axes = plt.subplots(2, 2, figsize=(15/2.54, 12/2.54))  # 15cm width
axes = axes.flatten()

for i, ds in enumerate(NOISY_DATASETS.keys()):
    if ds in mean_std:
        models = list(MODELS.keys())
        means = [np.mean(mean_std[ds][m]) if mean_std[ds][m] else 0.0 for m in models]
        stds = [np.std(mean_std[ds][m]) if mean_std[ds][m] else 0.0 for m in models]
        
        x_pos = np.arange(len(models))
        bars = axes[i].bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8,
                          color=[colors.get(m, '#1f77b4') for m in models])
        
        axes[i].set_ylabel('Accuracy')
        axes[i].set_xticks(x_pos)
        axes[i].set_xticklabels(models, rotation=45, ha='right')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'noisy_models_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Noise robustness trend chart
fig, ax = plt.subplots(figsize=(15/2.54, 10/2.54))  # 15cm width

snr_levels = ['SNR_2dB', 'SNR_0dB', 'SNR_-2dB', 'SNR_-4dB']
snr_values = [2, 0, -2, -4]

for model in MODELS.keys():
    model_means = []
    model_stds = []
    for ds in snr_levels:
        if ds in mean_std and model in mean_std[ds] and mean_std[ds][model]:
            model_means.append(np.mean(mean_std[ds][model]))
            model_stds.append(np.std(mean_std[ds][model]))
        else:
            model_means.append(0.0)
            model_stds.append(0.0)
    
    ax.errorbar(snr_values, model_means, yerr=model_stds, 
               label=model, marker='o', linewidth=2, markersize=6, 
               color=colors.get(model, '#1f77b4'))

ax.set_xlabel('SNR (dB)')
ax.set_ylabel('Test Accuracy')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'noise_robustness_trends.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Generate confusion matrices (using seed 42 results)
if 42 in all_predictions:
    for ds_name in NOISY_DATASETS.keys():
        if ds_name in all_predictions[42]:
            fig, axes = plt.subplots(2, 3, figsize=(15/2.54, 10/2.54))  # 15cm width
            axes = axes.flatten()
            
            for i, (model_name, (y_true, y_pred)) in enumerate(all_predictions[42][ds_name].items()):
                if len(y_true) > 0 and i < len(axes):
                    cm = confusion_matrix(y_true, y_pred)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], 
                               cbar_kws={'shrink': 0.8})
                    axes[i].set_xlabel('Predicted Label')
                    axes[i].set_ylabel('True Label')
            
            # Hide extra subplots
            for j in range(len(all_predictions[42][ds_name]), len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(VIS_DIR, f'confusion_matrices_{ds_name}.png'), dpi=300, bbox_inches='tight')
            plt.close()

print(f"\n{'='*60}")
print("Experiment completed!")
print(f"{'='*60}")
print(f"Results saved in: {OUTPUT_DIR}")
print(f"Models saved in: {MODEL_DIR}")
print(f"Visualizations saved in: {VIS_DIR}")
print(f"\nMain result files:")
print(f"- Summary statistics: {final_csv}")
print(f"- Performance comparison chart: {os.path.join(VIS_DIR, 'noisy_models_comparison.png')}")
print(f"- Robustness trend chart: {os.path.join(VIS_DIR, 'noise_robustness_trends.png')}")