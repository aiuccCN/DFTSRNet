# DFTSRNet: Dynamic Frequency-Temporal Structural Reasoning Network for Interpretable Bearing Fault Diagnosis

**Authors:** Dingyu Cao¹, Rongxi Wang¹*, Jing Yuan¹, Jianmin Gao¹
**Affiliation:** ¹State Key Laboratory for Manufacturing Systems Engineering, Xi'an Jiaotong University, Xi'an 710049, China
**Corresponding Author:** Rongxi Wang (rongxiwang@163.com)

## Abstract

Accurate and interpretable diagnosis of rolling bearings is important for the reliable operation of rotating machinery, yet many deep models remain black boxes and cannot adaptively select fault-informative frequency bands under varying loads and noise. DFTSRNet couples data-driven band selection with structural reasoning through three tightly integrated stages. Adaptive Frequency-Band Selection (AFBS) uses a learnable multi-scale one-dimensional filter bank followed by band-level spectral attention to emphasize informative sub-bands. Modal Graph Reasoning (MGR) represents the sub-bands as a sample-specific complete graph and infers data-driven inter-band relations via graph attention. A Multi-Scale Structure-aware Fusion CNN (MSF-CNN) injects the graph-derived summary into the temporal feature extractor through feature-wise linear modulation (FiLM). The band-attention weights and a post-hoc inter-band envelope-affinity analysis provide intrinsic, auditable diagnostic evidence; these are interpreted as data-driven statistical descriptors rather than as proof of mechanical causality.

This repository contains the complete implementation of DFTSRNet, including the core model, baseline pipelines, ablation variants, and analysis scripts used to reproduce the figures and tables in the paper.

## Project Structure

```
DFTSRNet/
├── core/                           # Core model implementation
│   ├── config.py                   # Configuration settings (paths are relative / env-overridable)
│   ├── dataset.py                  # Dataset loading and preprocessing
│   ├── focal_loss.py               # Focal loss implementation
│   ├── main.py                     # Main training script
│   ├── model.py                    # DFTSRNet model architecture (AFBS + MGR + MSF-CNN)
│   └── train_eval.py               # Training and evaluation utilities
├── data_processing/                # Data preprocessing and split scripts
├── experiments/                    # Experimental scripts and results
│   ├── comparison/                 # Baseline comparison (incl. TFN, KAN, RF, XGBoost)
│   ├── noise_robustness/           # Noise experiments (noise-aware training)
│   ├── ablation/                   # Ablation studies (AFBS / MGR / FiLM)
│   └── interpretability/           # Interpretability analysis (Autogram, SES, Gini, attention)
└── README.md                       # This file
```

> Note: the raw vibration datasets are **not** redistributed in this repository. Use the scripts in `data_processing/` to prepare the data and generate the exact train/validation/test splits.

## Datasets

Two rolling-bearing datasets are used:

- **Case 1 — CWRU benchmark (subsets A / B / C):** Case Western Reserve University drive-end data, **10 classes** (normal + ball, inner-race, and outer-race faults at 0.007 / 0.014 / 0.021 in), motor loads 0 / 1 / 2 hp, sampled at **12 kHz**.
- **Case 2 — Laboratory platform (subsets D / E / F):** N205 and NU205 bearings, **4 classes** (normal, outer-race, rolling-element, inner-race), rotational speeds **600 / 800 / 1000 rpm**, sampled at **50 kHz**.

Each input sample is a **1,024-point** segment after segmentation and normalization.

## Quick Start

### 1. Environment Setup

```bash
pip install torch numpy pandas matplotlib scikit-learn torch_geometric thop
```

### 2. Data Preprocessing

```bash
# Normalize and split the dataset (see data_processing/ for the exact scripts)
python data_processing/dataset_split.py
```

### 3. Training DFTSRNet

```bash
cd core
python main.py
```

### 4. Running Experiments

```bash
# Baseline comparison
cd experiments/comparison && python compare.py

# Noise experiments (noise-aware training)
cd experiments/noise_robustness && python train_all_datasets.py

# Ablation studies
cd experiments/ablation && python run_ablation.py
```

#### Interpretability Analysis

```bash
cd experiments/interpretability
# Step 1: first-layer feature/frequency maps
python step1_layer1_feature_maps/step1_layer1_feature_maps.py
# Step 2: spectral Gini index and loss ratio
python "Step 2 Gini importance analysis"/step2_gini_and_lossratio.py
# Step 3: attention-weighted spectrum
python "Step 3  Attention-weighted spectrum"/step3_attention_spectrum.py
# Step 4: Autogram resonance-band analysis (post-hoc expert cross-check)
python "Step 4 Infogram analysis"/step4_infogram_class6_ave.py
```

## Model Architecture

DFTSRNet integrates three coordinated components:

1. **Adaptive Frequency-Band Selection (AFBS):** a learnable one-dimensional filter bank of **six kernels** (lengths 3/5/7/9/11/13) produces multi-scale sub-bands; **band-level spectral attention** (a temperature softmax over per-band rFFT-magnitude descriptors) reweights and fuses them into a single signal. No center frequency or bandwidth is preset — the kernels are randomly initialized (Kaiming) and learned end-to-end.

2. **Modal Graph Reasoning (MGR):** the sub-bands form a sample-specific complete graph with self-loops; multi-head graph attention infers an inter-band **coupling matrix**, interpreted as a **data-driven affinity pattern rather than a causal or mechanical relation**.

3. **Multi-Scale Structure-aware Fusion CNN (MSF-CNN):** the graph summary conditions the convolutional features via **feature-wise linear modulation (FiLM)**, followed by multi-scale residual convolutions and a linear classifier.

## Key Features

- **Adaptive band selection** with a compact learnable filter bank (six kernels; 48 first-layer parameters).
- **Intrinsic interpretability:** band-attention weights (produced in the forward pass) together with a **post-hoc inter-band envelope-affinity** analysis indicate which bands and relations support each decision. These are reported as **statistical, data-driven** descriptors, not causal proof; they are cross-checked against Autogram and squared-envelope-spectrum (SES) analysis.
- **Compact model:** about **71K parameters**.
- **Robustness:** tolerant to impulsive disturbances; with **noise-aware training** it retains high accuracy down to −4 dB SNR. Strong narrowband harmonic interference remains challenging.
- **End-to-end learning:** trained directly from segmented, normalized vibration without hand-crafted features.

## Results

- **Datasets:** CWRU benchmark (subsets A–C) and a laboratory platform with N205/NU205 bearings at 600/800/1000 rpm (subsets D–F).
- **Accuracy:** competitive with recent strong baselines — comparable to the strongest interpretable time-frequency baseline (TFN) while using about **2.6× fewer parameters** — with low run-to-run variance and a statistically significant gain over the strongest deep baseline on CWRU (paired t-test, p < 0.05).
- **Robustness:** retained under impulsive disturbance and, with noise-aware training, at low SNR.
- **Interpretability:** the learned bands align with fault-characteristic regions (independently cross-checked by Autogram and SES analysis); inter-band relations are reported as statistical affinities.
- **Ablation studies:** confirm the complementarity of AFBS, MGR, and FiLM.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{cao2026dftsrnet,
  title   = {Dynamic Frequency-Temporal Structural Reasoning Network for Interpretable Bearing Fault Diagnosis},
  author  = {Cao, Dingyu and Wang, Rongxi and Yuan, Jing and Gao, Jianmin},
  journal = {Measurement},
  year    = {2026},
  note    = {State Key Laboratory for Manufacturing Systems Engineering, Xi'an Jiaotong University}
}
```

## License

This project is licensed under the MIT License — see the LICENSE file for details.

## Contact

For questions and support, please contact:
- **Corresponding Author**: Rongxi Wang (rongxiwang@163.com)
- **Address**: 28 Xianning West Road, Xi'an Jiaotong University, Xi'an 710049, China
