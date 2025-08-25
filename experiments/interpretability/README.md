# Interpretability Analysis

This directory contains scripts and tools for analyzing the interpretability of DFTSRNet model. The analysis is divided into four main steps to provide comprehensive insights into the model's decision-making process.

## Directory Structure

```
interpretability/
├── README.md                           # This file
├── config.py                           # Configuration settings
├── model.py                            # Model definitions
├── models_lib.py                       # Model library
├── feature_explain_main.py             # Main feature explanation script
├── bearing_params.json                 # Bearing parameters configuration
├── 提取时序和标签.py                    # Extract time series and labels
├── models/                             # Pre-trained models
│   ├── 1D-CNN.pth                      # 1D-CNN model
│   ├── DFTSRNet.pth                    # DFTSRNet model
│   └── WDCNN.pth                       # WDCNN model
├── step1_layer1_feature_maps/          # Step 1: Layer 1 feature maps analysis
│   ├── step1_layer1_feature_maps.py    # Feature map visualization script
│   ├── config.py                       # Configuration for step 1
│   ├── model.py                        # Model definitions for step 1
│   ├── models_lib.py                   # Model library for step 1
│   └── 结果/                           # Results directory
├── Step 2 Gini importance analysis/    # Step 2: Gini importance analysis
│   ├── step2_gini_and_lossratio.py     # Gini coefficient and loss ratio analysis
│   ├── step2_gini_kernel_only.py       # Kernel-only Gini analysis
│   ├── check_data_shape.py             # Data shape verification
│   ├── config.py                       # Configuration for step 2
│   ├── model.py                        # Model definitions for step 2
│   ├── models_lib.py                   # Model library for step 2
│   └── 结果/                           # Results directory
├── Step 3  Attention-weighted spectrum/ # Step 3: Attention-weighted spectrum analysis
│   ├── step3_attention_spectrum.py     # Attention spectrum visualization
│   ├── config.py                       # Configuration for step 3
│   ├── model.py                        # Model definitions for step 3
│   └── models_lib.py                   # Model library for step 3
└── Step 4 Infogram analysis/           # Step 4: Infogram analysis
    ├── step4_infogram_class6_ave.py    # Infogram analysis for 6 classes
    └── Autogram_class6_L5_combined.png # Combined infogram results
```

## Analysis Steps

### Step 1: Layer 1 Feature Maps Analysis

**Purpose**: Visualize and analyze the feature maps extracted by the first convolutional layer of different models.

**Script**: `step1_layer1_feature_maps/step1_layer1_feature_maps.py`

**Output**: 
- Feature map visualizations for DFTSRNet, 1D-CNN, and WDCNN models
- Combined comparison plots
- SVG and PNG format results

### Step 2: Gini Importance Analysis

**Purpose**: Analyze the importance of different frequency components using Gini coefficient and loss ratio metrics.

**Scripts**: 
- `Step 2 Gini importance analysis/step2_gini_and_lossratio.py`
- `Step 2 Gini importance analysis/step2_gini_kernel_only.py`

**Output**:
- Gini coefficient analysis results
- Loss ratio analysis
- Importance ranking tables
- Bar charts showing feature importance

### Step 3: Attention-Weighted Spectrum Analysis

**Purpose**: Visualize the attention weights applied to different frequency bands and analyze the model's focus areas.

**Script**: `Step 3  Attention-weighted spectrum/step3_attention_spectrum.py`

**Output**:
- Attention-weighted spectrum visualizations
- Combined spectrum analysis plots
- Frequency band importance analysis

### Step 4: Infogram Analysis

**Purpose**: Generate infograms (information diagrams) to show the information content and discriminative power of different signal components.

**Script**: `Step 4 Infogram analysis/step4_infogram_class6_ave.py`

**Output**:
- Infogram visualizations for all 6 bearing fault classes
- Combined infogram analysis
- Information content quantification

## Usage

### Prerequisites

Ensure you have the required dependencies installed:
```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn seaborn
```

### Running the Analysis

1. **Prepare Models**: Ensure pre-trained models are available in the `models/` directory

2. **Run Individual Steps**:
   ```bash
   # Step 1: Feature maps
   cd step1_layer1_feature_maps
   python step1_layer1_feature_maps.py
   
   # Step 2: Gini analysis
   cd "Step 2 Gini importance analysis"
   python step2_gini_and_lossratio.py
   
   # Step 3: Attention spectrum
   cd "Step 3  Attention-weighted spectrum"
   python step3_attention_spectrum.py
   
   # Step 4: Infogram analysis
   cd "Step 4 Infogram analysis"
   python step4_infogram_class6_ave.py
   ```

3. **Run Complete Analysis**:
   ```bash
   python feature_explain_main.py
   ```

## Configuration

- **config.py**: Main configuration file containing model parameters, data paths, and analysis settings
- **bearing_params.json**: Bearing-specific parameters for physical interpretation
- Each step has its own configuration file for step-specific settings

## Results Interpretation

### Feature Maps (Step 1)
- Shows how different models extract features from raw signals
- Reveals the learned filters and their responses to different fault patterns
- Helps understand the model's initial feature extraction capabilities

### Gini Analysis (Step 2)
- Quantifies the importance of different frequency components
- Higher Gini coefficients indicate more discriminative features
- Loss ratio analysis shows the impact of removing specific components

### Attention Spectrum (Step 3)
- Visualizes which frequency bands the model focuses on
- Shows adaptive frequency selection capabilities
- Reveals fault-specific frequency patterns

### Infogram Analysis (Step 4)
- Provides information-theoretic view of signal components
- Shows the discriminative power of different signal parts
- Helps validate the model's attention mechanisms

## Notes

- All analysis scripts are designed to work with the DFTSRNet architecture
- Results are saved in both PNG and SVG formats for publication quality
- The analysis supports comparison between different model architectures
- Configuration files allow easy adaptation to different datasets and model variants

## Citation

If you use these interpretability analysis tools in your research, please cite the DFTSRNet paper:

```bibtex
@article{dftsrnet2024,
  title={DFTSRNet: Dynamic Frequency-Time Structural Reasoning Network for Explainable Bearing Fault Diagnosis},
  author={Cao, Dingyu and Wang, Rongxi and Yuan, Jing and Gao, Jianmin},
  journal={[Journal Name]},
  year={2024}
}
```