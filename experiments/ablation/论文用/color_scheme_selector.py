#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配色方案选择器
用于预览和选择不同的图表配色方案
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# 定义所有可用的配色方案
COLOR_SCHEMES = {
    'scheme1': {
        'colors': ['#1f77b4', '#ff7f0e', '#2ca02c'],
        'name': '学术蓝橙配色（推荐）',
        'description': '经典的学术论文配色，蓝色代表稳定，橙色代表活力，绿色代表成长'
    },
    'scheme2': {
        'colors': ['#2E4057', '#048A81', '#F18F01'],
        'name': '深色专业配色',
        'description': '深沉专业的配色方案，适合技术类论文'
    },
    'scheme3': {
        'colors': ['#003f5c', '#58508d', '#bc5090'],
        'name': '科研期刊常用配色',
        'description': '顶级期刊常用的渐变配色，从深蓝到紫到粉'
    },
    'scheme4': {
        'colors': ['#d62728', '#2ca02c', '#1f77b4'],
        'name': '经典红绿蓝配色',
        'description': '传统的红绿蓝三色搭配，对比鲜明'
    },
    'scheme5': {
        'colors': ['#8c564b', '#e377c2', '#7f7f7f'],
        'name': '温和中性配色',
        'description': '柔和的棕色、粉色、灰色搭配，适合商务场合'
    },
    'scheme6': {
        'colors': ['#17becf', '#bcbd22', '#ff7f0e'],
        'name': '活力青春配色',
        'description': '青色、黄绿、橙色的活力组合'
    },
    'scheme7': {
        'colors': ['#9467bd', '#8c564b', '#e377c2'],
        'name': '优雅紫棕配色',
        'description': '紫色、棕色、粉色的优雅组合'
    },
    'scheme8': {
        'colors': ['#2E8B57', '#4682B4', '#DC143C'],
        'name': '自然海洋配色',
        'description': '海绿、钢蓝、深红的自然配色'
    }
}

def preview_color_schemes():
    """预览所有配色方案"""
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    fig.suptitle('配色方案预览', fontsize=16, fontweight='bold')
    
    schemes = list(COLOR_SCHEMES.items())
    
    for idx, (scheme_key, scheme_data) in enumerate(schemes):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # 绘制示例曲线
        x = np.linspace(0, 10, 100)
        for i, color in enumerate(scheme_data['colors']):
            y = np.sin(x + i * np.pi/3) + i * 0.5
            ax.plot(x, y, color=color, linewidth=3, label=f'Line {i+1}', alpha=0.8)
        
        ax.set_title(f"{scheme_key.upper()}: {scheme_data['name']}", fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 在图下方显示颜色条
        for i, color in enumerate(scheme_data['colors']):
            rect = Rectangle((i*2, -2.5), 1.8, 0.3, facecolor=color, alpha=0.8)
            ax.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig('color_schemes_preview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印配色方案信息
    print("\n" + "="*60)
    print("配色方案详细信息")
    print("="*60)
    
    for scheme_key, scheme_data in COLOR_SCHEMES.items():
        print(f"\n{scheme_key.upper()}: {scheme_data['name']}")
        print(f"描述: {scheme_data['description']}")
        print(f"颜色代码: {', '.join(scheme_data['colors'])}")
        print("-" * 40)

def apply_color_scheme(scheme_key):
    """应用指定的配色方案到生成脚本"""
    if scheme_key not in COLOR_SCHEMES:
        print(f"错误: 配色方案 '{scheme_key}' 不存在")
        print(f"可用方案: {', '.join(COLOR_SCHEMES.keys())}")
        return False
    
    # 读取当前的生成脚本
    script_path = 'generate_paper_figures.py'
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找并替换配色方案
        scheme_data = COLOR_SCHEMES[scheme_key]
        new_scheme_line = f"CURRENT_SCHEME = {{'colors': {scheme_data['colors']}, 'name': '{scheme_data['name']}'}}"
        
        # 替换CURRENT_SCHEME行
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('CURRENT_SCHEME ='):
                lines[i] = new_scheme_line
                break
        
        # 写回文件
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"✓ 成功应用配色方案: {scheme_data['name']}")
        print(f"颜色: {', '.join(scheme_data['colors'])}")
        return True
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {script_path}")
        return False
    except Exception as e:
        print(f"错误: {e}")
        return False

def show_current_scheme():
    """显示当前使用的配色方案"""
    script_path = 'generate_paper_figures.py'
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找当前配色方案
        for line in content.split('\n'):
            if line.strip().startswith('CURRENT_SCHEME ='):
                print(f"当前配色方案: {line.strip()}")
                return
        
        print("未找到当前配色方案设置")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {script_path}")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    print("配色方案选择器")
    print("=" * 30)
    print("1. 预览所有配色方案")
    print("2. 应用指定配色方案")
    print("3. 查看当前配色方案")
    print("4. 退出")
    
    while True:
        choice = input("\n请选择操作 (1-4): ").strip()
        
        if choice == '1':
            preview_color_schemes()
        elif choice == '2':
            print("\n可用配色方案:")
            for key, data in COLOR_SCHEMES.items():
                print(f"  {key}: {data['name']}")
            
            scheme = input("\n请输入配色方案代码 (如 scheme1): ").strip()
            apply_color_scheme(scheme)
        elif choice == '3':
            show_current_scheme()
        elif choice == '4':
            print("再见!")
            break
        else:
            print("无效选择，请输入 1-4")