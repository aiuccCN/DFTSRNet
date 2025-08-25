# DFTSRNet: Deep Fourier Transform Spectral Residual Network

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
│   ├── 均值归一化3s.py              # Mean normalization (3 seconds)
│   ├── 数据集划分.PY                # Dataset splitting
│   └── 数据集划分ABC.py             # Dataset splitting for ABC datasets
├── data/                           # Dataset directory
│   └── 西储大学实验数据集/           # Case Western Reserve University dataset
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
python data_processing/均值归一化3s.py

# Split the dataset
python data_processing/数据集划分.PY
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
python 20250808/第一层特征提取/step1_layer1_feature_maps.py

# Step 2: Gini importance analysis
python 20250808/基尼/step2_gini_and_lossratio.py

# Step 3: Attention-weighted spectrum
python 20250808/注意力加权频谱/step3_attention_spectrum.py

# Step 4: Infogram analysis
python 20250810/step4_infogram_class6_ave.py
```

## Model Architecture

DFTSRNet combines:
- **DFT-based Feature Extraction**: Captures frequency domain characteristics
- **Spectral Residual Learning**: Enhances feature representation
- **Attention Mechanism**: Focuses on discriminative frequency components
- **Multi-scale Processing**: Handles signals at different scales

## Key Features

1. **Frequency Domain Processing**: Utilizes DFT for robust feature extraction
2. **Residual Learning**: Improves gradient flow and feature learning
3. **Attention Mechanism**: Automatically focuses on important frequency bands
4. **Multi-dataset Validation**: Tested on six different datasets (A-F)
5. **Noise Robustness**: Evaluated under various noise conditions
6. **Interpretability**: Comprehensive analysis of model decisions

## Results

- **Accuracy**: Achieves state-of-the-art performance on CWRU dataset
- **Robustness**: Maintains high accuracy under noisy conditions
- **Efficiency**: Competitive computational complexity
- **Interpretability**: Clear visualization of learned features and attention patterns

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{dftsrnet2024,
  title={DFTSRNet: Deep Fourier Transform Spectral Residual Network for Bearing Fault Diagnosis},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and support, please contact [your-email@example.com].