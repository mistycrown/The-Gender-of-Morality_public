
"""
================================================================================
道德词典静态性别散点图 (Word Scatter Plot)
================================================================================

【功能描述】
绘制全量道德词汇的散点图：
- X轴：性别投影得分 (Gender Projection Score)
- Y轴：道德类别 (Category)
- 颜色 (Color)：道德极性 (Polarity: Virtue vs Vice)
- 标注 (Label)：自动标注典型词汇

【输入】
- moral_bias_mvp_static_full.json

【输出】
- viz/moral_static_full_word_scatter.png

【作者】Antigravity (Updated for Open Source Release)
【更新日期】2026-01-17
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import argparse
from pathlib import Path

# 导入统一配置
try:
    from plot_config import setup_fonts_and_style, save_figure, CATEGORY_MAPPING, CAT_ORDER, POLARITY_MAPPING_EN
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from plot_config import setup_fonts_and_style, save_figure, CATEGORY_MAPPING, CAT_ORDER, POLARITY_MAPPING_EN

# ==================== 配置 ====================

def get_default_paths():
    """Smartly determine default paths based on script location"""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent 
    
    candidates = [
        project_root / "result" / "moral_bias_mvp_static_full.json",
        project_root / "data" / "moral_bias_mvp_static_full.json",
        Path("moral_bias_mvp_static_full.json")
    ]
    
    default_json = None
    for p in candidates:
        if p.exists():
            default_json = p
            break
            
    viz_dir = project_root / "result" / "viz"
    return default_json, viz_dir

POLARITY_MAP = {
    "pos": "Virtue (美德)",
    "neg": "Vice (恶行)"
}

# ==================== 绘图函数 ====================

def setup_font():
    """设置字体 (使用统一配置)"""
    return setup_fonts_and_style()

def main():
    default_json, default_viz_dir = get_default_paths()
    
    parser = argparse.ArgumentParser(description="Visualize Static Moral Bias Scatter")
    parser.add_argument('--input', type=str, default=str(default_json) if default_json else None, 
                        help='Input JSON file path')
    parser.add_argument('--output-dir', type=str, default=str(default_viz_dir), 
                        help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    if not args.input or not os.path.exists(args.input):
        print(f"Error: Single JSON file not found at {args.input}")
        return
        
    viz_dir = Path(args.output_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    font = setup_font()
    
    # 1. 加载数据
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    df['Polarity Label'] = df['polarity'].map(POLARITY_MAP)
    
    print("Plotting word scatter plot...")
    
    # 我们可以重新绘图：不使用 stripplot，而是直接用 scatter，自己控制 y
    plt.figure(figsize=(20, 15)) # 加大画布
    
    # 手动构建绘图数据
    plot_x = []
    plot_y = []
    plot_colors = []
    plot_labels = []
    
    # 颜色映射 (Green / Orange)
    color_map = {"pos": "#2ca02c", "neg": "#ff7f0e"} 
    
    # 过滤掉没有数据的类别
    valid_cats = [cat for cat in CAT_ORDER if cat in df['category'].unique()]
    
    for cat_i, cat in enumerate(valid_cats):
        cat_df = df[df['category'] == cat].sort_values('score')
        if len(cat_df) == 0:
            continue
            
        # 简单算法：为了防止字重叠，我们在 y 轴方向做“扇形”展开或者上下交替
        offsets = [0, 0.15, -0.15, 0.3, -0.3, 0.45, -0.45, 0.6, -0.6]
        
        for i, (idx, row) in enumerate(cat_df.iterrows()):
            offset = offsets[i % len(offsets)]
            real_y = cat_i + offset
            
            plot_x.append(row['score'])
            plot_y.append(real_y)
            plot_colors.append(color_map.get(row['polarity'], "gray"))
            plot_labels.append(row['word'])
            
            # 绘制点
            plt.scatter(row['score'], real_y, c=color_map.get(row['polarity'], "gray"), 
                       s=60, alpha=0.7, edgecolors='w', linewidth=0.5)
            
            # 绘制文字
            # 在点旁边
            plt.text(row['score'], real_y + 0.08, row['word'], 
                     fontsize=15, 
                     color='black', alpha=0.9,
                     ha='center', va='bottom')

    plt.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    plt.title('Static Gender Bias Scatter (Full)', fontsize=24)
    plt.xlabel('← Female      Gender Projection Score      Male →', fontsize=16)
    plt.ylabel('Moral Category', fontsize=16)
    
    # 设置Y轴 - 使用英文维度名称 (仅展示由数据类别)
    # 将长标签分成两行以提高可读性
    ytick_labels = [CATEGORY_MAPPING.get(c, c).replace(' / ', ' /\n') for c in valid_cats]
    plt.yticks(range(len(valid_cats)), ytick_labels, fontsize=14)
    plt.ylim(-1, len(valid_cats))
    
    # 手动添加图例
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Virtue',
                              markerfacecolor='#2ca02c', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Vice',
                              markerfacecolor='#ff7f0e', markersize=10)]
    plt.legend(handles=legend_elements, title='Polarity', loc='upper left')

    plt.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    save_path = viz_dir / "moral_static_full_word_scatter.png"
    save_figure(plt.gcf(), save_path)
    print(f"\nScatter plot saved: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()
