#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并所有混淆矩阵TXT文件的脚本
"""

import os
import glob
from datetime import datetime

def merge_confusion_matrices():
    """
    合并所有混淆矩阵TXT文件
    """
    # 定义路径
    confusion_dir = r"e:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\results\confusion_matrices"
    output_file = os.path.join(confusion_dir, "merged_confusion_matrices.txt")
    
    # 获取所有TXT文件
    txt_files = glob.glob(os.path.join(confusion_dir, "*.txt"))
    txt_files = [f for f in txt_files if not f.endswith("merged_confusion_matrices.txt")]  # 排除输出文件
    
    # 按文件名排序
    txt_files.sort()
    
    print(f"找到 {len(txt_files)} 个TXT文件")
    
    # 合并文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 写入标题
        outfile.write("="*80 + "\n")
        outfile.write("消融实验 - 所有混淆矩阵合并报告\n")
        outfile.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        outfile.write(f"包含文件数量: {len(txt_files)}\n")
        outfile.write("="*80 + "\n\n")
        
        # 逐个读取并写入文件内容
        for i, txt_file in enumerate(txt_files, 1):
            filename = os.path.basename(txt_file)
            print(f"处理文件 {i}/{len(txt_files)}: {filename}")
            
            outfile.write(f"\n{'='*60}\n")
            outfile.write(f"文件 {i}: {filename}\n")
            outfile.write(f"{'='*60}\n")
            
            try:
                with open(txt_file, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)
                    outfile.write("\n")
            except Exception as e:
                outfile.write(f"读取文件时出错: {str(e)}\n")
                print(f"读取文件 {filename} 时出错: {str(e)}")
        
        # 写入结尾
        outfile.write("\n" + "="*80 + "\n")
        outfile.write("合并完成\n")
        outfile.write("="*80 + "\n")
    
    print(f"\n合并完成！输出文件: {output_file}")
    print(f"合并了 {len(txt_files)} 个文件")
    
    # 显示文件大小
    file_size = os.path.getsize(output_file)
    print(f"输出文件大小: {file_size:,} 字节")
    
    return output_file

def create_summary_table():
    """
    创建模型性能汇总表
    """
    confusion_dir = r"e:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\results\confusion_matrices"
    summary_file = os.path.join(confusion_dir, "performance_summary.txt")
    
    # 模型列表
    models = ['Base_CNN', '+AFBS', '+MGR', '+FiLM', 'Full', 'w_o_AFBS', 'w_o_FiLM', 'w_o_MGR']
    seeds = [20, 42, 97]
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("消融实验性能汇总表\n")
        f.write("="*80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("模型性能概览:\n")
        f.write("-"*60 + "\n")
        f.write(f"{'模型名称':<15} {'种子20':<10} {'种子42':<10} {'种子97':<10} {'平均值':<10}\n")
        f.write("-"*60 + "\n")
        
        for model in models:
            accuracies = []
            row = f"{model:<15}"
            
            for seed in seeds:
                filename = f"confusion_matrix_{model}_seed{seed}.txt"
                filepath = os.path.join(confusion_dir, filename)
                
                if os.path.exists(filepath):
                    # 这里可以解析文件获取准确率，暂时用占位符
                    row += f"{'N/A':<10}"
                else:
                    row += f"{'缺失':<10}"
            
            row += f"{'N/A':<10}\n"
            f.write(row)
        
        f.write("-"*60 + "\n")
        f.write("\n注意: 具体准确率需要从详细结果文件中获取\n")
    
    print(f"性能汇总表已创建: {summary_file}")
    return summary_file

if __name__ == "__main__":
    print("开始合并混淆矩阵TXT文件...")
    
    # 合并所有TXT文件
    merged_file = merge_confusion_matrices()
    
    # 创建汇总表
    summary_file = create_summary_table()
    
    print("\n处理完成！")
    print(f"合并文件: {merged_file}")
    print(f"汇总表: {summary_file}")