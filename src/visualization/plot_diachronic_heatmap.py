"""
================================================================================
历时置换检验显著性可视化 (Diachronic Permutation Test Visualization) - Combined
================================================================================

【功能描述】
结合历时偏见原始数据(JSON)和置换检验结果(CSV)，绘制热力图。
包含单独图表 (All, Pos, Neg) 和 组合图表 (Virtue/Vice Side-by-Side).

【可视化逻辑】
- 颜色 (Color): 原始偏见得分的中位数 (Median Score) - 来自 JSON
  - 映射: RdBu_r (Blue=Male, Red=Female)
- 标记 (Annotation): 统计显著性 (P-value) - 来自 CSV
  - * : p < 0.05
  - **: p < 0.01

【输出】
- result/.../viz/diachronic_significance_heatmap_all.png
- result/.../viz/diachronic_significance_heatmap_pos.png
- result/.../viz/diachronic_significance_heatmap_neg.png
- result/.../viz/diachronic_significance_heatmap_combined.png (Virtue & Vice)

【作者】Antigravity
【创建日期】2026-01-14
================================================================================
"""

import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 导入统一配置
from plot_config import setup_fonts_and_style, save_figure, CATEGORY_MAPPING, CAT_ORDER, DYNASTY_MAPPING_EN, POLARITY_MAPPING_EN, DYNASTY_ORDER

# ==================== 配置 ====================

CSV_FILE = r"result\20260112autodl结果\result\diachronic_permutation_test\diachronic_permutation_results.csv"
JSON_FILE = r"result\20260112autodl结果\result\20251230-moral-mvp-diachronic\moral_bias_mvp_diachronic_full.json"
VIZ_DIR = r"result\20260112autodl结果\result\diachronic_permutation_test\viz"

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

def load_json_scores():
    """加载JSON数据为DataFrame [Category, Polarity, Dynasty, Score]"""
    print(f"Loading JSON: {JSON_FILE}")
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
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
    
    sns.heatmap(pivot_score, cmap='RdBu_r', center=0, vmin=vmin, vmax=vmax,
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
                       fontproperties=font, fontsize=9, fontweight='bold')
                
    ax.set_title(title, fontproperties=font, fontsize=14)
    ax.set_xlabel('Dynasty', fontproperties=font, fontsize=11)
    if ylabel:
        ax.set_ylabel('Moral Category', fontproperties=font, fontsize=11)
    else:
        ax.set_ylabel('')
        
    # 设置刻度字体
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontproperties=font)
    
    if show_yticks:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontproperties=font)
    else:
        ax.set_yticks([])

def plot_single_heatmap(df_merged, subcategory, font):
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
    outfile = os.path.join(VIZ_DIR, f"diachronic_significance_heatmap_{subcategory.lower()}.png")
    save_figure(plt.gcf(), outfile.replace('.png', ''))
    print(f"✓ Saved: {outfile}")
    plt.close()

def plot_combined_pos_neg_heatmap(df_pos, df_neg, font):
    """绘制 Virtue 和 Vice 的组合热力图 (左右布局)"""
    
    pivot_score_pos, pivot_p_pos = prepare_pivot_data(df_pos)
    pivot_score_neg, pivot_p_neg = prepare_pivot_data(df_neg)
    
    # --- 关键修改: 统一致类别索引 (Alignment) ---
    # 获取所有出现的类别 (并集)
    all_cats = list(pivot_score_pos.index.union(pivot_score_neg.index))
    
    # 按 CAT_ORDER 重新排序
    ordered_cats = []
    for c in CAT_ORDER:
        disp = CATEGORY_MAPPING.get(c, c)
        if disp in all_cats:
            ordered_cats.append(disp)
    # 添加剩余可能的类别
    for c in all_cats:
        if c not in ordered_cats:
            ordered_cats.append(c)
            
    # Reindex 两张表，确保行完全一致
    pivot_score_pos = pivot_score_pos.reindex(ordered_cats)
    pivot_p_pos = pivot_p_pos.reindex(ordered_cats)
    pivot_score_neg = pivot_score_neg.reindex(ordered_cats)
    pivot_p_neg = pivot_p_neg.reindex(ordered_cats)
    
    # Determine common color scale
    # Filter nan before min/max
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
    
    # Create Figure
    # sharey=False (防止 matplotlib 自动隐藏ticks), 通过手动对齐 Index 保证视觉对齐
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=False, gridspec_kw={'wspace': 0.05})
    
    # Plot (A) Virtue
    draw_heatmap_on_ax(axes[0], pivot_score_pos, pivot_p_pos, '(A) Virtue (Pos)', font, vmin, vmax, cbar=False, ylabel=True, show_yticks=True)
    
    # Plot (B) Vice
    draw_heatmap_on_ax(axes[1], pivot_score_neg, pivot_p_neg, '(B) Vice (Neg)', font, vmin, vmax, cbar=True, ylabel=False, show_yticks=False)
    
    plt.tight_layout()
    outfile = os.path.join(VIZ_DIR, "diachronic_significance_heatmap_combined.png")
    save_figure(fig, outfile.replace('.png', ''))
    print(f"✓ Saved Combined: {outfile}")
    plt.close()

def main():
    if not os.path.exists(CSV_FILE) or not os.path.exists(JSON_FILE):
        print("Input files not found.")
        return
        
    os.makedirs(VIZ_DIR, exist_ok=True)
    font = setup_fonts_and_style()
    
    # 1. Load Data
    df_json = load_json_scores() 
    df_csv = pd.read_csv(CSV_FILE) 
    
    # 2. Clean CSV
    df_csv['Category'] = df_csv['Category'].apply(clean_category_name_csv)
    
    # 3. Process each subcategory (All, Pos, Neg)
    
    # --- All ---
    scores_all = df_json.groupby(['Category', 'Dynasty'])['Score'].median().reset_index()
    p_all = df_csv[df_csv['Subcategory'] == 'All'][['Category', 'Dynasty', 'P_value']]
    merged_all = pd.merge(scores_all, p_all, on=['Category', 'Dynasty'], how='left')
    plot_single_heatmap(merged_all, 'All', font)
    
    # --- Pos ---
    scores_pos = df_json[df_json['Polarity'] == 'pos'].groupby(['Category', 'Dynasty'])['Score'].median().reset_index()
    p_pos = df_csv[df_csv['Subcategory'] == 'Pos'][['Category', 'Dynasty', 'P_value']]
    merged_pos = pd.merge(scores_pos, p_pos, on=['Category', 'Dynasty'], how='left')
    plot_single_heatmap(merged_pos, 'Pos (Virtue)', font)
    
    # --- Neg ---
    scores_neg = df_json[df_json['Polarity'] == 'neg'].groupby(['Category', 'Dynasty'])['Score'].median().reset_index()
    p_neg = df_csv[df_csv['Subcategory'] == 'Neg'][['Category', 'Dynasty', 'P_value']]
    merged_neg = pd.merge(scores_neg, p_neg, on=['Category', 'Dynasty'], how='left')
    plot_single_heatmap(merged_neg, 'Neg (Vice)', font)
    
    # --- Combined ---
    plot_combined_pos_neg_heatmap(merged_pos, merged_neg, font)

if __name__ == "__main__":
    main()
