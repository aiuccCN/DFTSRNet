import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_training_curves_from_csv():
    """
    从CSV文件读取训练数据并绘制训练曲线图
    使用2×1子图布局：上图显示准确率曲线，下图显示损失曲线
    """
    
    # 设置matplotlib参数
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['font.size'] = 10
    
    # 定义模型颜色（与compare.py完全一致）
    palette = {
        '1D-CNN': (33/255, 158/255, 188/255),
        'WDCNN': (19/255, 103/255, 131/255),
        'Inception': (2/255, 48/255, 74/255),
        'VMD+CNN': (254/255, 183/255, 5/255),
        'DFTSRNet': 'red'
    }
    
    # CSV文件路径
    DATA_DIR = r'e:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\对比实验\对比试验结果\训练曲线'
    acc_csv_path = os.path.join(DATA_DIR, 'training_accuracy_DatasetA.csv')
    loss_csv_path = os.path.join(DATA_DIR, 'training_loss_DatasetA.csv')
    
    # 检查文件是否存在
    if not os.path.exists(acc_csv_path):
        print(f"错误：找不到准确率数据文件 {acc_csv_path}")
        print("请先运行 train_and_save_data.py 生成训练数据")
        return
    
    if not os.path.exists(loss_csv_path):
        print(f"错误：找不到损失数据文件 {loss_csv_path}")
        print("请先运行 train_and_save_data.py 生成训练数据")
        return
    
    # 读取CSV数据
    print("正在从CSV文件读取训练数据...")
    acc_df = pd.read_csv(acc_csv_path)
    loss_df = pd.read_csv(loss_csv_path)
    
    # 获取epoch数据
    epochs = acc_df['Epoch'].values
    
    # 创建2x1子图布局：上图显示准确率曲线，下图显示损失曲线
    # 将英寸转换为厘米：13cm = 5.12inch, 10.32cm = 4.06inch
    fig, ax = plt.subplots(2, 1, figsize=(5.12, 4.06), sharex=True)
    
    # 绘制准确率曲线
    print("正在绘制准确率曲线...")
    for model_name in ['1D-CNN', 'WDCNN', 'Inception', 'VMD+CNN', 'DFTSRNet']:
        if model_name in acc_df.columns:
            acc_values = acc_df[model_name].values
            style = {'lw': 4, 'color': palette[model_name]} if model_name == 'DFTSRNet' else {'lw': 1.5, 'color': palette[model_name]}
            ax[0].plot(epochs, acc_values, label=model_name, **style)
    
    ax[0].set_ylabel('Accuracy', fontsize=10, fontweight='normal', fontfamily='Times New Roman')
    ax[0].legend(loc='lower right', fontsize=10, prop={'family': 'Times New Roman', 'weight': 'normal'})
    ax[0].grid(True, alpha=0.3)
    ax[0].tick_params(axis='both', which='major', labelsize=10)
    
    # 绘制损失曲线
    print("正在绘制损失曲线...")
    for model_name in ['1D-CNN', 'WDCNN', 'Inception', 'VMD+CNN', 'DFTSRNet']:
        if model_name in loss_df.columns:
            loss_values = loss_df[model_name].values
            
            # 处理可能的元组格式数据
            processed_loss = []
            for val in loss_values:
                if isinstance(val, str) and val.startswith('('):
                    # 解析元组字符串，提取第一个数值（损失值）
                    try:
                        import ast
                        tuple_val = ast.literal_eval(val)
                        processed_loss.append(tuple_val[0] if isinstance(tuple_val, tuple) else tuple_val)
                    except:
                        processed_loss.append(float(val.split(',')[0].strip('(')))
                else:
                    processed_loss.append(float(val))
            
            style = {'lw': 4, 'color': palette[model_name]} if model_name == 'DFTSRNet' else {'lw': 1.5, 'color': palette[model_name]}
            ax[1].plot(epochs, processed_loss, **style)
    
    ax[1].set_xlabel('Epoch', fontsize=10, fontweight='normal', fontfamily='Times New Roman')
    ax[1].set_ylabel('Loss', fontsize=10, fontweight='normal', fontfamily='Times New Roman')
    ax[1].legend(['1D-CNN', 'WDCNN', 'Inception', 'VMD+CNN', 'DFTSRNet'], 
                loc='upper right', fontsize=10, prop={'family': 'Times New Roman', 'weight': 'normal'})
    ax[1].grid(True, alpha=0.3)
    ax[1].tick_params(axis='both', which='major', labelsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    OUTPUT_DIR = r'e:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\对比实验\对比试验结果\训练曲线'
    curve_path = os.path.join(OUTPUT_DIR, 'training_curves_DatasetA_from_csv.png')
    
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线图已保存到: {curve_path}")
    print("\n训练曲线绘制完成！")
    
    # 显示数据统计信息
    print("\n=== 数据统计信息 ===")
    print(f"训练轮数: {len(epochs)} epochs")
    print(f"包含模型: {list(acc_df.columns[1:])}")
    
    # 显示最终准确率
    print("\n=== 最终准确率 ===")
    for model_name in ['1D-CNN', 'WDCNN', 'Inception', 'VMD+CNN', 'DFTSRNet']:
        if model_name in acc_df.columns:
            final_acc = acc_df[model_name].iloc[-1]
            print(f"{model_name}: {final_acc:.4f}")

def show_data_info():
    """
    显示CSV文件中的数据信息
    """
    DATA_DIR = r'e:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\GPT修改版\对比实验\对比试验结果\训练曲线'
    acc_csv_path = os.path.join(DATA_DIR, 'training_accuracy_DatasetA.csv')
    loss_csv_path = os.path.join(DATA_DIR, 'training_loss_DatasetA.csv')
    
    if os.path.exists(acc_csv_path) and os.path.exists(loss_csv_path):
        acc_df = pd.read_csv(acc_csv_path)
        loss_df = pd.read_csv(loss_csv_path)
        
        print("=== 准确率数据信息 ===")
        print(f"数据形状: {acc_df.shape}")
        print(f"列名: {list(acc_df.columns)}")
        print(f"前5行数据:")
        print(acc_df.head())
        
        print("\n=== 损失数据信息 ===")
        print(f"数据形状: {loss_df.shape}")
        print(f"列名: {list(loss_df.columns)}")
        print(f"前5行数据:")
        print(loss_df.head())
    else:
        print("CSV数据文件不存在，请先运行 train_and_save_data.py")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--info':
        show_data_info()
    else:
        plot_training_curves_from_csv()