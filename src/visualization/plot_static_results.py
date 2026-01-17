
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
- moral_bias_mvp_static_full.json
- moral_bias_mvp_static_full_zscore.json

【输出】
- viz/moral_static_full_boxplot[_zscore].png
- viz/moral_static_full_median_bar[_zscore].png

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
    # Fallback if running from a different directory
    import sys
    sys.path.append(str(Path(__file__).parent))
    from plot_config import setup_fonts_and_style, save_figure, CATEGORY_MAPPING, CAT_ORDER, POLARITY_MAPPING_EN

# ==================== 配置 ====================

def get_default_paths():
    """Smartly determine default paths based on script location"""
    script_dir = Path(__file__).resolve().parent
    # Project root assumption: src/visualization -> ../..
    project_root = script_dir.parent.parent 
    
    # Potential data locations
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
            
    # Default output dir
    viz_dir = project_root / "result" / "viz"
    
    return default_json, viz_dir

# ==================== 绘图函数 ====================

def setup_font():
    """设置字体 (使用统一配置)"""
    return setup_fonts_and_style()

def main():
    default_json, default_viz_dir = get_default_paths()
    
    parser = argparse.ArgumentParser(description="Visualize Static Moral Bias Results")
    parser.add_argument('--input', type=str, default=str(default_json) if default_json else None, 
                        help='Input JSON file path')
    parser.add_argument('--output-dir', type=str, default=str(default_viz_dir), 
                        help='Output directory for visualizations')
    parser.add_argument('--use-zscore', action='store_true', help='Use Z-Score normalized data')
    
    args = parser.parse_args()

    # Determine input file
    json_file = args.input
    if args.use_zscore and json_file:
        # Try to find the zscore version if not explicitly provided as zscore
        p = Path(json_file)
        if "zscore" not in p.name:
            zscore_candidate = p.with_name(p.stem + "_zscore" + p.suffix)
            if zscore_candidate.exists():
                json_file = str(zscore_candidate)
                print(f"Auto-detected Z-Score file: {json_file}")
            else:
                print(f"Warning: Z-Score file {zscore_candidate} not found. Using {json_file}")
    
    if not json_file or not os.path.exists(json_file):
        print(f"Error: Data file not found. Please specify with --input.")
        print(f"Searched default: {default_json}")
        return

    print(f"Reading data: {json_file}")
    print(f"Mode: {'Z-Score Normalized' if args.use_zscore else 'Raw Score'}")
        
    viz_dir = Path(args.output_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    font = setup_font()
    
    # 1. 加载数据
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    
    # 添加类别显示名称
    df['category_display'] = df['category'].map(CATEGORY_MAPPING)
    
    if df.empty:
        print("Warning: Dataset is empty")
        return

    # 2. 统计分析 (Split by Polarity)
    print("\n[Category Static Gender Bias Stats - By Polarity]")
    print(f"{'Category':<12} {'Polarity':<8} {'Median':<10} {'Mean':<10} {'Std':<10} {'Count':<5}")
    print("-" * 60)
    
    stats_list = []
    
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
    
    # 3. 绘制箱线图 (Boxplot)
    plt.figure(figsize=(14, 8))
    
    # 过滤掉没有数据的类别
    valid_cats = [c for c in CAT_ORDER if c in df['category'].unique()]
    
    # Updated Colors: Green (Virtue) / Orange (Vice)
    palette = {"pos": "#2ca02c", "neg": "#ff7f0e"}
    
    sns.boxplot(x='category', y='score', hue='polarity', data=df, order=valid_cats, palette=palette, width=0.6)
    sns.stripplot(x='category', y='score', hue='polarity', data=df, order=valid_cats, dodge=True, color=".3", size=3, alpha=0.6)
    
    plt.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Z-Score 区间指示
    if args.use_zscore:
        plt.axhspan(-0.2, 0.2, color='gray', alpha=0.1, label='Neutral Zone')
        
    plt.title(f'Distribution of Gender Bias by Moral Category (Full) {"(Z-Score)" if args.use_zscore else ""}', fontsize=20)
    plt.ylabel(metric_label, fontsize=14)
    plt.xlabel('Moral Category', fontsize=14)
    
    # 使用英文Category Mapping 并换行
    locs, labels = plt.xticks()
    new_labels = [CATEGORY_MAPPING.get(label.get_text(), label.get_text()).replace(' / ', ' /\n') for label in labels]
    plt.xticks(locs, new_labels, rotation=0, ha='center', fontsize=11)
    
    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    # stripplot 会重复添加 legend，取前两个即可 (pos, neg)
    plt.legend(handles[:2], ["Virtue", "Vice"], title="Polarity")
    
    plt.tight_layout()
    save_path = viz_dir / f"moral_static_full_boxplot{fname_suffix}.png"
    save_figure(plt.gcf(), save_path)
    print(f"\nBoxplot saved: {save_path}")
    plt.close()
    
    # 4. 绘制中位数条形图 (Median Bar Chart)
    plt.figure(figsize=(14, 10))
    
    # 排序：按中位数从高到低排序
    stats_df_sorted = stats_df.sort_values('Median', ascending=False)
    
    # 颜色映射
    colors = ['#2ca02c' if p == 'pos' else '#ff7f0e' for p in stats_df_sorted['Polarity']]
    
    sns.barplot(x='Label', y='Median', data=stats_df_sorted, palette=colors)
    
    plt.axhline(0, color='black', linewidth=1)
    
    plt.title(f'Ranked Median Gender Bias by Category {"(Z-Score)" if args.use_zscore else ""} - Full', fontsize=20)
    plt.ylabel(f'Median {metric_label}', fontsize=14)
    plt.xlabel('Category-Polarity', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # 添加数值标签
    offset = 0.05 if args.use_zscore else 0.002
    for i, row in enumerate(stats_df_sorted.itertuples()):
        label_y = row.Median + (offset if row.Median > 0 else -1.5*offset)
        plt.text(i, label_y, f"{row.Median:.2f}", ha='center', va='bottom' if row.Median > 0 else 'top', fontsize=9)

    plt.tight_layout()
    save_path = viz_dir / f"moral_static_full_median_bar{fname_suffix}.png"
    save_figure(plt.gcf(), save_path)
    print(f"Median bar chart saved: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()
