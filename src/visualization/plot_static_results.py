
"""
================================================================================
道德词典静态类别偏见可视化脚本 (全量版)
================================================================================

【功能描述】
1. 读取静态全量计算生成的 JSON 数据。
2. 按 Category（类别）进行聚合。
3. 计算每个类别的中位数 (Median) 和其他统计量。
4. 绘制箱线图 (Boxplot) 展示各类别得分分布。
5. 绘制中位数条形图。
6. 支持 --use-zscore 参数进行 Z-Score 标准化展示。

【输入】
- result/.../moral_bias_mvp_static_full.json
- result/.../moral_bias_mvp_static_full_zscore.json

【输出】
- result/.../viz/moral_static_full_boxplot[_zscore].png
- result/.../viz/moral_static_full_median_bar[_zscore].png

【作者】AI辅助生成
【创建日期】2026-01-11
【最后修改】2026-01-12 - 支持 Z-Score 标准化
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import argparse

# 导入统一配置
from plot_config import setup_fonts_and_style, save_figure, CATEGORY_MAPPING, CAT_ORDER, POLARITY_MAPPING_EN

# ==================== 配置 ====================

DEFAULT_JSON_FILE = r"result\20260112autodl结果\result\20251230-moral-mvp-static\moral_bias_mvp_static_full.json"
DEFAULT_ZSCORE_FILE = r"result\20260112autodl结果\result\20251230-moral-mvp-static\moral_bias_mvp_static_full_zscore.json"
VIZ_DIR = r"result\20260112autodl结果\result\20251230-moral-mvp-static\viz"
FONT_PATH = r'C:\Windows\Fonts\simkai.ttf'

# 类别中文映射 及 排序参考




# ==================== 绘图函数 ====================

def setup_font():
    """设置字体 (使用统一配置)"""
    return setup_fonts_and_style()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-zscore', action='store_true', help='使用 Z-Score 标准化数据')
    args = parser.parse_args()

    # 确定输入文件
    json_file = DEFAULT_ZSCORE_FILE if args.use_zscore else DEFAULT_JSON_FILE
    
    if not os.path.exists(json_file):
        print(f"找不到数据文件: {json_file}")
        if args.use_zscore and os.path.exists(DEFAULT_JSON_FILE):
             print(f"退回使用原始文件: {DEFAULT_JSON_FILE}")
             json_file = DEFAULT_JSON_FILE
             args.use_zscore = False
        else:
             return

    print(f"正在读取数据: {json_file}")
    print(f"模式: {'Z-Score 标准化' if args.use_zscore else '原始得分'}")
        
    os.makedirs(VIZ_DIR, exist_ok=True)
    font = setup_font()
    
    # 1. 加载数据
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    
    # 添加类别显示名称
    df['category_display'] = df['category'].map(CATEGORY_MAPPING)
    
    if df.empty:
        print("警告: 数据集为空")
        return

    # 2. 统计分析 (Split by Polarity)
    print("\n[类别静态性别偏见统计 - 分极性]")
    print(f"{'Category':<12} {'Polarity':<8} {'Median':<10} {'Mean':<10} {'Std':<10} {'Count':<5}")
    print("-" * 60)
    
    stats_list = []
    
    # 获取存在的 category-polarity 组合
    existing_combinations = df.groupby(['category', 'polarity']).size().reset_index()
    
    for cat in CAT_ORDER:
        for pol in ['pos', 'neg']:
            sub_df = df[(df['category'] == cat) & (df['polarity'] == pol)]
            if sub_df.empty:
                continue
                
            median = sub_df['score'].median()
            mean = sub_df['score'].mean()
            std = sub_df['score'].std()
            count = len(sub_df)
            
            print(f"{cat:<12} {pol:<8} {median:>8.4f} {mean:>8.4f} {std:>8.4f} {count:>5}")
            
            stats_list.append({
                'Category': cat,
                'Polarity': pol,
                'Label': f"{cat}-{pol}", # 组合标签
                'Median': median,
                'Mean': mean
            })
        
    stats_df = pd.DataFrame(stats_list)
    
    # Label adjustment
    metric_label = "Gender Bias Z-Score (Pos=Male, Neg=Female)" if args.use_zscore else "Gender Projection Score (Pos=Male, Neg=Female)"
    fname_suffix = "_zscore" if args.use_zscore else ""
    
    # 3. 绘制箱线图 (Boxplot) - 保持不变，因为 hue='polarity' 已经不仅展示了合并数据
    plt.figure(figsize=(14, 8))
    
    # 过滤掉没有数据的类别，确保 CAT_ORDER 只包含有数据的
    valid_cats = [c for c in CAT_ORDER if c in df['category'].unique()]
    
    sns.boxplot(x='category', y='score', hue='polarity', data=df, order=valid_cats, palette={"pos": "#e74c3c", "neg": "#3498db"}, width=0.6)
    sns.stripplot(x='category', y='score', hue='polarity', data=df, order=valid_cats, dodge=True, color=".3", size=3, alpha=0.6)
    
    plt.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Z-Score 区间指示
    if args.use_zscore:
        plt.axhspan(-0.2, 0.2, color='gray', alpha=0.1, label='Neutral Zone')
        
    plt.title(f'Distribution of Gender Bias by Moral Category (Full) {"(Z-Score)" if args.use_zscore else ""}', fontproperties=font, fontsize=16)
    plt.ylabel(metric_label, fontproperties=font, fontsize=12)
    plt.xlabel('Moral Category', fontproperties=font, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # 使用英文Category Mapping
    locs, labels = plt.xticks()
    new_labels = [CATEGORY_MAPPING.get(label.get_text(), label.get_text()) for label in labels]
    plt.xticks(locs, new_labels, fontproperties=font)
    
    # 修复 legend
    handles, labels = plt.gca().get_legend_handles_labels()
    # stripplot 会重复添加 legend，取前两个即可 (pos, neg)
    plt.legend(handles[:2], ["Virtue", "Vice"], title="Polarity")
    
    plt.tight_layout()
    save_path = os.path.join(VIZ_DIR, f"moral_static_full_boxplot{fname_suffix}.png")
    base_path = save_path.replace('.png', '')
    save_figure(plt.gcf(), base_path)
    print(f"\n箱线图已保存: {base_path}.* (svg/pdf/png)")
    plt.close()
    
    # 4. 绘制中位数条形图 (Median Bar Chart) - 改为 Split Bar
    plt.figure(figsize=(14, 10))
    
    # 排序：按中位数绝对值排序，或者按数值排序？
    # 用户希望看到极性。按中位数从高到低排序比较清晰
    stats_df_sorted = stats_df.sort_values('Median', ascending=False)
    
    # 颜色映射
    colors = ['#e74c3c' if p == 'pos' else '#3498db' for p in stats_df_sorted['Polarity']]
    
    sns.barplot(x='Label', y='Median', data=stats_df_sorted, palette=colors)
    
    plt.axhline(0, color='black', linewidth=1)
    
    plt.title(f'Ranked Median Gender Bias by Category {"(Z-Score)" if args.use_zscore else ""} - Full', fontproperties=font, fontsize=16)
    plt.ylabel(f'Median {metric_label}', fontproperties=font, fontsize=12)
    plt.xlabel('Category-Polarity', fontproperties=font, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # 添加数值标签
    offset = 0.05 if args.use_zscore else 0.002
    for i, row in enumerate(stats_df_sorted.itertuples()):
        label_y = row.Median + (offset if row.Median > 0 else -1.5*offset)
        plt.text(i, label_y, f"{row.Median:.2f}", ha='center', va='bottom' if row.Median > 0 else 'top', fontsize=9)

    plt.tight_layout()
    base_path = os.path.join(VIZ_DIR, f"moral_static_full_median_bar{fname_suffix}")
    save_figure(plt.gcf(), base_path)
    print(f"中位数图已保存: {base_path}.* (svg/pdf/png)")
    plt.close()

if __name__ == "__main__":
    main()
