
"""
================================================================================
历时置换检验显著性可视化 (Diachronic Permutation Test Visualization) - Combined
================================================================================

【功能描述】
结合历时偏见原始数据(JSON)和置换检验结果(CSV)，绘制热力图。
包含单独图表 (All, Pos, Neg) 和 组合图表 (Virtue/Vice Side-by-Side).

【可视化逻辑】
- 颜色 (Color): 原始偏见得分的中位数 (Median Score) - 来自 JSON
  - 映射: RdBu (Blue=Female, Red=Male)
- 标记 (Annotation): 统计显著性 (P-value) - 来自 CSV
  - * : p < 0.05
  - **: p < 0.01

【输出】
- viz/diachronic_significance_heatmap_all.png
- viz/diachronic_significance_heatmap_pos.png
- viz/diachronic_significance_heatmap_neg.png
- viz/diachronic_significance_heatmap_combined.png (Virtue & Vice)

【作者】Antigravity (Updated for Open Source Release)
【更新日期】2026-01-17
"""

import os
import re
import json
import textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

# 导入统一配置
try:
    from plot_config import setup_fonts_and_style, save_figure, CATEGORY_MAPPING, CAT_ORDER, DYNASTY_MAPPING_EN, POLARITY_MAPPING_EN, DYNASTY_ORDER
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from plot_config import setup_fonts_and_style, save_figure, CATEGORY_MAPPING, CAT_ORDER, DYNASTY_MAPPING_EN, POLARITY_MAPPING_EN, DYNASTY_ORDER

# ==================== 配置 ====================

def get_default_paths():
    """Smartly determine default paths based on script location"""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent 
    
    # 1. CSV
    csv_candidates = [
        project_root / "result" / "diachronic_permutation_results.csv",
        project_root / "data" / "diachronic_permutation_results.csv",
        Path("diachronic_permutation_results.csv")
    ]
    default_csv = None
    for p in csv_candidates:
        if p.exists(): default_csv = p; break
        
    # 2. JSON
    json_candidates = [
        project_root / "result" / "moral_bias_mvp_diachronic_full.json",
        project_root / "data" / "moral_bias_mvp_diachronic_full.json",
        Path("moral_bias_mvp_diachronic_full.json")
    ]
    default_json = None
    for p in json_candidates:
        if p.exists(): default_json = p; break

    viz_dir = project_root / "result" / "viz"
    return default_csv, default_json, viz_dir

# 缩写映射 (CSV -> Full Name)
ABBREV_MAP = {
    'Aut': 'Authority',
    'Loy': 'Loyalty',
    'Fair': 'Fairness',
    'San': 'Sanctity',
    'Care': 'Care',
    'Waste': 'Waste',
    'Diligence': 'Diligence',
    'Modesty': 'Modesty',
    'Valor': 'Valor'
}

# ==================== 数据处理函数 ====================

def clean_category_name_csv(raw_name):
    """CSV中的类别名清洗: '01care' -> 'Care'"""
    name = re.sub(r'^\d+', '', raw_name)
    name = name.capitalize()
    return ABBREV_MAP.get(name, name)

def get_significance_label(p_value):
    """根据P值返回星号标记"""
    if pd.isna(p_value): return ""
    if p_value < 0.001: return "***"
    if p_value < 0.01: return "**"
    if p_value < 0.05: return "*"
    return ""

def load_json_scores(json_file):
    """加载JSON数据为DataFrame [Category, Polarity, Dynasty, Score]"""
    print(f"Loading JSON: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    flat_data = []
    for item in data:
        cat = item['category']
        polarity = item.get('polarity', 'unknown')
        scores = item['scores']
        for dynasty in DYNASTY_ORDER:
            if dynasty in scores and scores[dynasty] is not None:
                flat_data.append({
                    'Category': cat,
                    'Polarity': polarity,
                    'Dynasty': dynasty,
                    'Score': scores[dynasty]
                })
    return pd.DataFrame(flat_data)

def prepare_pivot_data(df_merged):
    """
    辅助函数：将合并后的DataFrame转换为透视表
    Returns: (pivot_score, pivot_p)
    """
    # 准备 Display Label
    df_merged['DisplayCategory'] = df_merged['Category'].apply(lambda x: CATEGORY_MAPPING.get(x, x))
    df_merged['DynastyEN'] = df_merged['Dynasty'].apply(lambda d: DYNASTY_MAPPING_EN.get(d, d))
    
    pivot_score = df_merged.pivot(index='DisplayCategory', columns='DynastyEN', values='Score')
    pivot_p = df_merged.pivot(index='DisplayCategory', columns='DynastyEN', values='P_value')
    
    # 排序行 (Category)
    # 按 CAT_ORDER
    ordered_cats = []
    for c in CAT_ORDER:
        disp = CATEGORY_MAPPING.get(c, c)
        if disp in pivot_score.index:
            ordered_cats.append(disp)
    # 添加剩余
    remaining = [c for c in pivot_score.index if c not in ordered_cats]
    ordered_index = ordered_cats + remaining
    
    pivot_score = pivot_score.reindex(ordered_index)
    pivot_p = pivot_p.reindex(ordered_index)
    
    # 排序列 (Dynasty)
    dynasty_order_en = [DYNASTY_MAPPING_EN.get(d, d) for d in DYNASTY_ORDER]
    present_cols = [d for d in dynasty_order_en if d in pivot_score.columns]
    pivot_score = pivot_score[present_cols]
    pivot_p = pivot_p[present_cols]
    
    return pivot_score, pivot_p

def draw_heatmap_on_ax(ax, pivot_score, pivot_p, title, font, vmin, vmax, cbar=True, ylabel=True, show_yticks=True):
    """辅助函数：在指定Ax上绘制热力图"""
    
    sns.heatmap(pivot_score, cmap='RdBu', center=0, vmin=vmin, vmax=vmax,
                     cbar=cbar, cbar_kws={'label': 'Median Bias Score'},
                     annot=False, ax=ax)
    
    # 标注文本
    for y in range(pivot_score.shape[0]):
        for x in range(pivot_score.shape[1]):
            score_val = pivot_score.iloc[y, x]
            p_val = pivot_p.iloc[y, x]
            
            sig_str = get_significance_label(p_val)
            text_color = 'white' if abs(score_val) > vmax * 0.5 else 'black'
            
            if not pd.isna(score_val):
                txt = f"{score_val:.3f}\n{sig_str}"
                ax.text(x + 0.5, y + 0.5, txt, 
                       ha='center', va='center', color=text_color,
                       fontsize=9, fontweight='bold')
                
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Dynasty', fontsize=11)
    if ylabel:
        ax.set_ylabel('Moral Category', fontsize=11)
    else:
        ax.set_ylabel('')
        
    # 设置刻度字体 - 横向排列，换行
    xlabels = [label.get_text().replace(' & ', '\n& ') for label in ax.get_xticklabels()]
    ax.set_xticklabels(xlabels, rotation=0, ha='center')
    
    if show_yticks:
        # Wrap Y-axis labels
        ylabels = [label.get_text().replace(' / ', ' /\n') for label in ax.get_yticklabels()]
        ax.set_yticklabels(ylabels, rotation=0)
    else:
        ax.set_yticks([])

def plot_single_heatmap(df_merged, subcategory, font, output_dir):
    """绘制单个热力图 (All, Pos, or Neg)"""
    pivot_score, pivot_p = prepare_pivot_data(df_merged)
    
    plt.figure(figsize=(12, 8))
    
    # Color Scale
    abs_max = max(abs(pivot_score.max().max()), abs(pivot_score.min().min()))
    if abs_max < 0.2: vmax, vmin = 0.15, -0.15
    elif abs_max < 0.5: vmax, vmin = 0.3, -0.3
    else: vmax, vmin = 0.5, -0.5
    
    ax = plt.gca()
    draw_heatmap_on_ax(ax, pivot_score, pivot_p, f'Diachronic Significance: {subcategory}', font, vmin, vmax, cbar=True, ylabel=True, show_yticks=True)
    
    plt.tight_layout()
    outfile = output_dir / f"diachronic_significance_heatmap_{subcategory.lower()}.png"
    save_figure(plt.gcf(), outfile)
    print(f"✓ Saved: {outfile}")
    plt.close()

def plot_combined_pos_neg_heatmap(df_pos, df_neg, font, output_dir):
    """绘制 Virtue 和 Vice 的组合热力图 (左右布局)"""
    
    pivot_score_pos, pivot_p_pos = prepare_pivot_data(df_pos)
    pivot_score_neg, pivot_p_neg = prepare_pivot_data(df_neg)
    
    # 统一致类别索引
    all_cats = list(pivot_score_pos.index.union(pivot_score_neg.index))
    
    ordered_cats = []
    for c in CAT_ORDER:
        disp = CATEGORY_MAPPING.get(c, c)
        if disp in all_cats:
            ordered_cats.append(disp)
    for c in all_cats:
        if c not in ordered_cats:
            ordered_cats.append(c)
            
    pivot_score_pos = pivot_score_pos.reindex(ordered_cats)
    pivot_p_pos = pivot_p_pos.reindex(ordered_cats)
    pivot_score_neg = pivot_score_neg.reindex(ordered_cats)
    pivot_p_neg = pivot_p_neg.reindex(ordered_cats)

    # 统一列索引
    all_cols = list(pivot_score_pos.columns.union(pivot_score_neg.columns))
    dynasty_order_en = [DYNASTY_MAPPING_EN.get(d, d) for d in DYNASTY_ORDER]
    ordered_cols = [c for c in dynasty_order_en if c in all_cols]

    pivot_score_pos = pivot_score_pos.reindex(columns=ordered_cols)
    pivot_p_pos = pivot_p_pos.reindex(columns=ordered_cols)
    pivot_score_neg = pivot_score_neg.reindex(columns=ordered_cols)
    pivot_p_neg = pivot_p_neg.reindex(columns=ordered_cols)
    
    # Determine common color scale
    vals_pos = pivot_score_pos.values.flatten()
    vals_neg = pivot_score_neg.values.flatten()
    all_vals = np.concatenate([vals_pos[~np.isnan(vals_pos)], vals_neg[~np.isnan(vals_neg)]])
    
    if len(all_vals) > 0:
        abs_max = max(abs(np.max(all_vals)), abs(np.min(all_vals)))
    else:
        abs_max = 0.5
    
    if abs_max < 0.2: vmax, vmin = 0.15, -0.15
    elif abs_max < 0.5: vmax, vmin = 0.3, -0.3
    else: vmax, vmin = 0.5, -0.5
    
    # Create Figure with GridSpec
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[10, 10, 0.4], wspace=0.1)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    cbar_ax = fig.add_subplot(gs[2])

    # Plot (A) Virtue
    sns.heatmap(pivot_score_pos, cmap='RdBu', center=0, vmin=vmin, vmax=vmax,
                 cbar=False, annot=False, ax=ax1)
    
    # Plot (B) Vice
    sns.heatmap(pivot_score_neg, cmap='RdBu', center=0, vmin=vmin, vmax=vmax,
                 cbar=True, cbar_ax=cbar_ax, cbar_kws={'label': 'Median Bias Score'},
                 annot=False, ax=ax2)
    
    # Manually Annotation reusing logic
    def annotate_ax(ax, data_score, data_p):
        for y in range(data_score.shape[0]):
            for x in range(data_score.shape[1]):
                score_val = data_score.iloc[y, x]
                p_val = data_p.iloc[y, x]
                
                sig_str = get_significance_label(p_val)
                text_color = 'white' if abs(score_val) > vmax * 0.5 else 'black'
                
                if not pd.isna(score_val):
                    txt = f"{score_val:.3f}\n{sig_str}"
                    ax.text(x + 0.5, y + 0.5, txt, 
                           ha='center', va='center', color=text_color,
                           fontsize=9, fontweight='bold')
    
    annotate_ax(ax1, pivot_score_pos, pivot_p_pos)
    annotate_ax(ax2, pivot_score_neg, pivot_p_neg)

    # Decoration Ax1
    ax1.set_title('(A) Virtue (Pos)', fontsize=14)
    ax1.set_xlabel('Dynasty', fontsize=11)
    ax1.set_ylabel('Moral Category', fontsize=11)
    xlabels1 = [label.get_text().replace(' & ', '\n& ') for label in ax1.get_xticklabels()]
    ax1.set_xticklabels(xlabels1, rotation=0, ha='center')
    ylabels1 = [label.get_text().replace(' / ', ' /\n') for label in ax1.get_yticklabels()]
    ax1.set_yticklabels(ylabels1, rotation=0)

    # Decoration Ax2
    ax2.set_title('(B) Vice (Neg)', fontsize=14)
    ax2.set_xlabel('Dynasty', fontsize=11)
    ax2.set_ylabel('', fontsize=11)
    xlabels2 = [label.get_text().replace(' & ', '\n& ') for label in ax2.get_xticklabels()]
    ax2.set_xticklabels(xlabels2, rotation=0, ha='center')
    ax2.set_yticks([]) 

    cbar_ax.tick_params(labelsize=10)
    
    outfile = output_dir / "diachronic_significance_heatmap_combined.png"
    save_figure(fig, outfile)
    print(f"✓ Saved Combined: {outfile}")
    plt.close()

def main():
    default_csv, default_json, default_viz_dir = get_default_paths()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=str(default_csv) if default_csv else None, help='Path to diachronic_permutation_results.csv')
    parser.add_argument('--json', default=str(default_json) if default_json else None, help='Path to moral_bias_mvp_diachronic_full.json')
    parser.add_argument('--output-dir', default=str(default_viz_dir))
    args = parser.parse_args()
    
    if not args.csv or not os.path.exists(args.csv):
        print("Error: CSV file not found (specify --csv)")
        return
    if not args.json or not os.path.exists(args.json):
        print("Error: JSON file not found (specify --json)")
        return
        
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    font = setup_fonts_and_style()
    
    # 1. Load Data
    df_json = load_json_scores(args.json) 
    df_csv = pd.read_csv(args.csv) 
    
    # 2. Clean CSV
    df_csv['Category'] = df_csv['Category'].apply(clean_category_name_csv)
    
    # 3. Process each subcategory (All, Pos, Neg)
    
    # --- All ---
    scores_all = df_json.groupby(['Category', 'Dynasty'])['Score'].median().reset_index()
    p_all = df_csv[df_csv['Subcategory'] == 'All'][['Category', 'Dynasty', 'P_value']]
    merged_all = pd.merge(scores_all, p_all, on=['Category', 'Dynasty'], how='left')
    plot_single_heatmap(merged_all, 'All', font, output_dir)
    
    # --- Pos ---
    scores_pos = df_json[df_json['Polarity'] == 'pos'].groupby(['Category', 'Dynasty'])['Score'].median().reset_index()
    p_pos = df_csv[df_csv['Subcategory'] == 'Pos'][['Category', 'Dynasty', 'P_value']]
    merged_pos = pd.merge(scores_pos, p_pos, on=['Category', 'Dynasty'], how='left')
    plot_single_heatmap(merged_pos, 'Pos (Virtue)', font, output_dir)
    
    # --- Neg ---
    scores_neg = df_json[df_json['Polarity'] == 'neg'].groupby(['Category', 'Dynasty'])['Score'].median().reset_index()
    p_neg = df_csv[df_csv['Subcategory'] == 'Neg'][['Category', 'Dynasty', 'P_value']]
    merged_neg = pd.merge(scores_neg, p_neg, on=['Category', 'Dynasty'], how='left')
    plot_single_heatmap(merged_neg, 'Neg (Vice)', font, output_dir)
    
    # --- Combined ---
    plot_combined_pos_neg_heatmap(merged_pos, merged_neg, font, output_dir)

if __name__ == "__main__":
    main()
