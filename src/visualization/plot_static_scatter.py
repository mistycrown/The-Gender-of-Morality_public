
"""
================================================================================
道德词典静态性别散点图 (Word Scatter Plot)
================================================================================

【功能描述】
绘制全量道德词汇的散点图：
- X轴：性别投影得分 (Gender Projection Score)
- Y轴：道德类别 (Category)
- 颜色 (Color)：道德极性 (Polarity: Virtue vs Vice)
- 标注 (Label)：自动标注每个类别中得分最高和最低的典型词汇

这使得用户可以直接看到具体的词分布在性别轴的什么位置。

【输入】
- result/20260112autodl结果/result/20251230-moral-mvp-static/moral_bias_mvp_static_full.json

【输出】
- result/20260112autodl结果/result/20251230-moral-mvp-static/viz/moral_static_full_word_scatter.png

【作者】Antigravity
【创建日期】2026-01-12
================================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# 导入统一配置
from plot_config import setup_fonts_and_style, save_figure, CATEGORY_MAPPING, CAT_ORDER, POLARITY_MAPPING_EN

# ==================== 配置 ====================

JSON_FILE = r"result\20260112autodl结果\result\20251230-moral-mvp-static\moral_bias_mvp_static_full.json"
VIZ_DIR = r"result\20260112autodl结果\result\20251230-moral-mvp-static\viz"
FONT_PATH = r'C:\Windows\Fonts\simkai.ttf'





POLARITY_MAP = {
    "pos": "Virtue (美德)",
    "neg": "Vice (恶行)"
}

# ==================== 绘图函数 ====================

def setup_font():
    """设置字体 (使用统一配置)"""
    return setup_fonts_and_style()

def main():
    if not os.path.exists(JSON_FILE):
        print(f"找不到文件: {JSON_FILE}")
        return
        
    os.makedirs(VIZ_DIR, exist_ok=True)
    font = setup_font()
    
    # 1. 加载数据
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    df['Polarity Label'] = df['polarity'].map(POLARITY_MAP)
    
    print("正在标注所有词汇...")
    # 为了避免文字完全重叠，我们可以给 y_center 加一个随机微小抖动，或者按原有jitter逻辑
    # stripplot 的 jitter 是随机的，我们无法确切知道点画在哪。
    # 既然用户想要看清"每个字"，不如我们放弃 stripplot 的随机 jitter，手动控制 y 轴位置。
    # 方法：在每个 category 内部，根据 score 排序，然后交替给 y 轴加 offset
    
    # 我们可以重新绘图：不使用 stripplot，而是直接用 scatter，自己控制 y
    plt.figure(figsize=(20, 15)) # 加大画布
    
    # 手动构建绘图数据
    plot_x = []
    plot_y = []
    plot_colors = []
    plot_labels = []
    
    # 颜色映射
    color_map = {"pos": "#1f77b4", "neg": "#d62728"} # Blue vs Red
    
    # 过滤掉没有数据的类别
    present_categories = [cat for cat in CAT_ORDER if cat in df['category'].unique()]
    
    for cat_i, cat in enumerate(present_categories):
        cat_df = df[df['category'] == cat].sort_values('score')
        if len(cat_df) == 0:
            continue
            
        # 简单算法：为了防止字重叠，我们在 y 轴方向做“扇形”展开或者上下交替
        # 或者仅仅是交替上下：0, +0.1, -0.1, +0.2, -0.2 ...
        # 这样同一直线（同类）上的字能错开
        
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
                     fontproperties=font, fontsize=9, 
                     color='black', alpha=0.9,
                     ha='center', va='bottom')

    plt.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    plt.title('Static Gender Bias Scatter (Full)', fontproperties=font, fontsize=24)
    plt.xlabel('← Female      Gender Projection Score      Male →', fontproperties=font, fontsize=16)
    plt.ylabel('Moral Category', fontproperties=font, fontsize=16)
    
    # 设置Y轴 - 使用英文维度名称 (仅展示由数据类别)
    ytick_labels = [CATEGORY_MAPPING.get(c, c) for c in present_categories]
    plt.yticks(range(len(present_categories)), ytick_labels, fontsize=10)
    plt.ylim(-1, len(present_categories))
    
    # 手动添加图例
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Virtue',
                              markerfacecolor='#1f77b4', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Vice',
                              markerfacecolor='#d62728', markersize=10)]
    plt.legend(handles=legend_elements, prop=font, title='Polarity', loc='upper left')

    plt.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    base_path = os.path.join(VIZ_DIR, "moral_static_full_word_scatter")
    save_figure(plt.gcf(), base_path)
    print(f"\n散点图已保存: {base_path}.* (svg/pdf/png)")
    plt.close()

if __name__ == "__main__":
    main()
