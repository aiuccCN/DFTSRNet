# DFTSRNet 中文介绍

## 项目简介

DFTSRNet（Dynamic Frequency-Temporal Structural Reasoning Network，动态频率-时间结构推理网络）是一种面向滚动轴承故障诊断的深度学习方法。本项目提供模型实现、基线对比、消融实验、抗噪实验和可解释性分析代码，旨在从振动信号中学习具有判别力的频带特征，并分析频带之间的关系。

模型主要由以下部分构成：

- **自适应频带选择（AFBS）**：通过可学习滤波器组提取多尺度频带，并学习不同频带的重要性。
- **模态图推理（MGR）**：将各频带表示为图节点，建模频带之间的关系。
- **多尺度结构感知融合 CNN（MSF-CNN）**：融合图推理得到的结构信息与卷积特征，完成故障分类。

项目包含 CWRU 轴承数据集及实验室平台数据的处理和实验脚本。原始振动数据不随仓库分发；请根据 `data_processing/` 中的说明自行准备数据。

## 论文

**论文题目：** Dynamic Frequency-Temporal Structural Reasoning Network for Interpretable Bearing Fault Diagnosis  
**作者：** Dingyu Cao、Rongxi Wang、Jing Yuan、Jianmin Gao  
**期刊：** *Measurement*，2026  
**DOI：** [10.1016/j.measurement.2026.123438](https://doi.org/10.1016/j.measurement.2026.123438)

```bibtex
@article{cao2026dftsrnet,
  title   = {Dynamic Frequency-Temporal Structural Reasoning Network for Interpretable Bearing Fault Diagnosis},
  author  = {Cao, Dingyu and Wang, Rongxi and Yuan, Jing and Gao, Jianmin},
  journal = {Measurement},
  year    = {2026},
  doi     = {10.1016/j.measurement.2026.123438},
  url     = {https://doi.org/10.1016/j.measurement.2026.123438}
}
```

## 代码与运行

请参阅[英文项目主页](README.md)，了解环境配置、数据预处理、模型训练和实验运行方法。
