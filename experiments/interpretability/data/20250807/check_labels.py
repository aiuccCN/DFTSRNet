import numpy as np

# 加载测试集标签
test_labels = np.load(r'e:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\数据集\西储大学实验数据集\Dataset_A\test_labels.npy')

print('测试集标签分布:')
unique, counts = np.unique(test_labels, return_counts=True)
for label, count in zip(unique, counts):
    print(f'标签 {label}: {count} 个样本')

# 查找故障样本
fault_indices = np.where(test_labels != 0)[0]
print(f'\n故障样本索引前10个: {fault_indices[:10]}')
print(f'故障样本总数: {len(fault_indices)}')

if len(fault_indices) > 0:
    print(f'第一个故障样本的索引: {fault_indices[0]}')
    print(f'第一个故障样本的标签: {test_labels[fault_indices[0]]}')
else:
    print('测试集中没有故障样本')

# 显示前20个样本的标签
print(f'\n前20个样本的标签: {test_labels[:20]}')