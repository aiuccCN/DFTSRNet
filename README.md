# DFTSRNet: Dynamic Frequency-Time Structural Reasoning Network for Explainable Bearing Fault Diagnosis

**Authors:** Dingyu Cao¹, Rongxi Wang¹*, Jing Yuan¹, Jianmin Gao¹  
**Affiliation:** ¹State Key Laboratory for Manufacturing Systems Engineering, Xi'an Jiaotong University, Xi'an 710049, China  
**Corresponding Author:** Rongxi Wang (rongxiwang@163.com)

## Abstract

Accurate and interpretable diagnosis of rolling bearings is crucial for reliable operation of rotating machinery. Many deep models remain black boxes and cannot adaptively select fault-informative frequency bands under changing loads and noise. This paper presents DFTSRNet, a Dynamic Frequency-Time Structural Reasoning Network that couples data-driven band selection with structural reasoning for robust and transparent diagnosis. The network first performs adaptive frequency-band selection with a learnable one-dimensional filter bank and assigns physically meaningful spectral attention to salient bands. It then models sub-bands as a fully connected graph and infers inter-band couplings through graph attention, producing structured embeddings. Finally, a modal-structural-fusion convolutional network injects graph-derived context by feature-wise linear modulation and completes classification with multi-scale residual convolutions.

This repository contains the complete implementation of DFTSRNet for bearing fault diagnosis, including core model code, experimental scripts, and analysis tools.

## Project Structure

```
DFTSRNet/
├── core/                           # Core model implementation
│   ├── config.py                   # Configuration settings
│   ├── dataset.py                  # Dataset loading and preprocessing
│   ├── focal_loss.py               # Focal loss implementation
│   ├── main.py                     # Main training script
│   ├── model.py                    # DFTSRNet model architecture
│   └── train_eval.py               # Training and evaluation utilities
├── data_processing/                # Data preprocessing scripts
│   ├── mean_normalization_3s.py    # Mean normalization (3 seconds)
│   ├── dataset_split.py             # Dataset splitting
│   └── dataset_split_ABC.py         # Dataset splitting for ABC datasets
├── data/                           # Dataset directory
│   └── CWRU_Dataset/               # Case Western Reserve University dataset
│       ├── A/                      # Dataset A
│       ├── B/                      # Dataset B
│       ├── C/                      # Dataset C
│       ├── D/                      # Dataset D
│       ├── E/                      # Dataset E
│       └── F/                      # Dataset F
├── experiments/                    # Experimental scripts and results
│   ├── comparison/                 # Comparative experiments
│   ├── noise_robustness/           # Noise robustness experiments
│   ├── ablation/                   # Ablation studies
│   └── interpretability/           # Interpretability analysis
└── README.md                       # This file
```

## Dataset

The experiments are conducted on the Case Western Reserve University (CWRU) bearing dataset, which includes six subsets (A-F) with different operating conditions:

- **Dataset A-F**: Different motor loads and fault severities
- **Fault Types**: Normal, Inner race fault, Outer race fault, Ball fault
- **Sampling Frequency**: 12 kHz
- **Signal Length**: 3 seconds (36,000 data points)

## Quick Start

### 1. Environment Setup

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

### 2. Data Preprocessing

```bash
# Normalize the dataset
python data_processing/mean_normalization_3s.py

# Split the dataset
python data_processing/dataset_split.py
```

### 3. Training DFTSRNet

```bash
cd core
python main.py
```

### 4. Running Experiments

#### Comparative Experiments
```bash
cd experiments/comparison
python compare.py
```

#### Noise Robustness Experiments
```bash
cd experiments/noise_robustness
python train_all_datasets.py
```

#### Ablation Studies
```bash
cd experiments/ablation
python run_ablation.py
```

#### Interpretability Analysis
```bash
cd experiments/interpretability
# Step 1: Feature map visualization
python step1_layer1_feature_maps/step1_layer1_feature_maps.py

# Step 2: Gini importance analysis
python "Step 2 Gini importance analysis"/step2_gini_and_lossratio.py

# Step 3: Attention-weighted spectrum
python "Step 3  Attention-weighted spectrum"/step3_attention_spectrum.py

# Step 4: Infogram analysis
python "Step 4 Infogram analysis"/step4_infogram_class6_ave.py
```

## Model Architecture

DFTSRNet integrates three coordinated components:

1. **Adaptive Frequency-Band Selection**: Uses a learnable one-dimensional filter bank to perform adaptive frequency-band selection and assigns physically meaningful spectral attention to salient bands

2. **Structural Reasoning Module**: Models sub-bands as a fully connected graph and infers inter-band couplings through graph attention, producing structured embeddings and a coupling matrix for interpretation

3. **Modal-Structural-Fusion Network**: Injects graph-derived context by feature-wise linear modulation (FiLM) and completes classification with multi-scale residual convolutions

## Key Features

1. **Adaptive Band Selection**: Learnable filter bank that adaptively selects fault-informative frequency bands under changing conditions
2. **Structural Reasoning**: Graph attention networks model inter-band couplings and provide sample-specific structural insights
3. **Intrinsic Interpretability**: Attention weights and coupling matrix reveal which bands and interactions drive predictions
4. **Robust Performance**: State-of-the-art accuracy with low variance and strong robustness to low signal-to-noise ratios
5. **End-to-End Learning**: Trained without hand-crafted preprocessing, naturally extends to multi-sensor inputs
6. **Lightweight Design**: Efficient parameter usage with feature-wise linear modulation

## Results

- **Datasets**: Evaluated on Case Western Reserve University (CWRU) benchmark and laboratory platform with N205 and NU205 bearings at 600, 800, and 1000 rpm
- **Performance**: State-of-the-art accuracy with low variance across different operating conditions
- **Robustness**: Strong performance maintained at low signal-to-noise ratios
- **Interpretability**: Visual analyses confirm that selected bands and learned couplings align with bearing physics
- **Ablation Studies**: Confirm the complementarity of all three network modules

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{dftsrnet2024,
  title={DFTSRNet: Dynamic Frequency-Time Structural Reasoning Network for Explainable Bearing Fault Diagnosis},
  author={Cao, Dingyu and Wang, Rongxi and Yuan, Jing and Gao, Jianmin},
  journal={[Journal Name]},
  year={2024},
  note={State Key Laboratory for Manufacturing Systems Engineering, Xi'an Jiaotong University}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and support, please contact:
- **Corresponding Author**: Rongxi Wang (rongxiwang@163.com)
- **Address**: 28 XianNing Road, Xi'an Jiaotong University, Xi'an 710049, China
- **Tel**: +86-29-82663702
- **Fax**: +86-29-82669137