# 修正版可解释性分析使用说明

## 概述

本脚本实现了基于**Lorenz曲线解析式的标准基尼指数**和**标准Infogram**的可解释性分析，修正了原版代码中的计算偏差问题。

## 主要修正内容

### 1. 基尼指数（Gini Index）修正

- **原问题**: 数据未排序、含零值处理不当、数值放缩敏感
- **修正方案**: 采用Lorenz曲线解析式标准实现
- **公式**: `G = (2 * Σ(i * x_i)) / (n * Σx_i) - (n+1)/n`
- **特点**: 
  - 数据自动排序（非降序）
  - 零值安全处理
  - 数值稳定性好
  - 结果限制在[0,1]区间

### 2. Infogram修正

- **原问题**: 简化为时间包络熵或谱峰计数
- **修正方案**: 标准文献实现流程
- **流程**: 带通滤波 → Hilbert包络 → 包络谱 → Shannon熵 → 归一化负熵
- **公式**: `J = 1 - H/log(N)`，其中H为Shannon熵，N为有效谱点数
- **特点**:
  - 使用零相位filtfilt滤波
  - 去除直流分量
  - 归一化负熵计算
  - 二维网格热图显示

## 运行要求

### 数据准备

1. **训练/测试数据**（必需）:
   - `数据集/西储大学实验数据集/Dataset_A/train_data_preprocessed.npy`
   - `数据集/西储大学实验数据集/Dataset_A/train_labels.npy`
   - `数据集/西储大学实验数据集/Dataset_A/val_data_preprocessed.npy`
   - `数据集/西储大学实验数据集/Dataset_A/val_labels.npy`
   - `数据集/西储大学实验数据集/Dataset_A/test_data_preprocessed.npy`
   - `数据集/西储大学实验数据集/Dataset_A/test_labels.npy`

2. **模型文件**（可选）:
   - 如果需要从预训练模型提取特征，请准备相应的.pth文件
   - 脚本会自动训练模型进行分析

### 环境要求

```python
numpy
torch
matplotlib
seaborn
pandas
scipy
```

## 使用方法

### 直接运行

```bash
cd "e:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\新版可解释实验\20250807"
python corrected_explainability_analysis.py
```

### 输出文件

运行完成后，将在`结果`文件夹中生成：

1. **Figure_1_Feature_Maps_Comparison_Class4.png**
   - 对应论文图14
   - 三个模型（DFTSRNet、WDCNN、1D-CNN）的第一层特征映射对比
   - 显示归一化频率 × 核函数的热图

2. **Table_2_Quantitative_Metrics.csv**
   - 对应论文表6
   - 包含基尼指数、L1损失、L2损失、损失比等定量指标
   - 基尼指数基于第一层特征的频域稀疏度计算

3. **Figure_3_Infogram_Analysis.png**
   - 对应论文图16
   - 上图：Infogram热图（频率×带宽级别的归一化负熵）
   - 下图：包络谱，标注轴承故障特征频率

## 参数说明

### Infogram参数

- **中心频率范围**: 1000-6000 Hz，步长250 Hz
- **带宽选择**: [500, 1000, 1500, 2000] Hz
- **采样频率**: 12000 Hz
- **FFT点数**: 4096

### 基尼指数参数

- **计算对象**: 第一层特征的rFFT幅度谱
- **归一化**: 每个通道独立归一化
- **聚合方式**: 逐核计算后平均

## 技术细节

### 基尼指数计算流程

1. 提取第一层特征 (B, C, L)
2. 沿长度维做rFFT得到幅度谱 (B, C, F)
3. 每个通道归一化: `spec = spec / (spec.sum(axis=-1) + 1e-12)`
4. 重塑为 (B*C, F) 并逐行计算基尼指数
5. 平均得到最终结果

### Infogram计算流程

1. 对每个(中心频率, 带宽)组合设计带通滤波器
2. 使用filtfilt进行零相位滤波
3. Hilbert变换得到解析信号的包络
4. 对包络做FFT得到包络谱
5. 去除直流分量并归一化
6. 计算Shannon熵: `H = -Σ(p * log(p))`
7. 归一化负熵: `J = 1 - H/log(N)`

## 故障排除

### 常见问题

1. **数据文件未找到**
   - 检查数据集路径是否正确
   - 确保所有必需的.npy文件存在

2. **模型训练失败**
   - 检查GPU内存是否足够
   - 尝试减小batch_size

3. **Infogram计算异常**
   - 检查信号长度是否足够
   - 确保采样频率设置正确

### 调试建议

- 运行前检查所有路径设置
- 如遇到内存不足，可减小数据批次大小
- 可以单独运行各个分析函数进行调试

## 与论文对应关系

| 输出文件 | 论文图表 | 说明 |
|---------|---------|------|
| Figure_1_Feature_Maps_Comparison_Class4.png | 图14 | 第一层特征映射对比 |
| Table_2_Quantitative_Metrics.csv | 表6 | 基尼指数和损失比 |
| Figure_3_Infogram_Analysis.png | 图16 | Infogram和包络谱 |

## 参考文献

- Lorenz曲线基尼系数计算方法
- Infogram标准实现（Antoni, 2007）
- 轴承故障诊断中的循环平稳分析