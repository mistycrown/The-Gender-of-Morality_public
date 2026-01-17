
"""
================================================================================
性别轴线验证可视化脚本 (Gender Axis Validation Visualization)
================================================================================

【功能描述】
绘制性别轴线验证的森林图（Forest Plot）。
将控制词（Control Terms）在性别轴上的投影偏差（Projection Bias）可视化。

【输入】
- 优先读取: gender_validity_results.csv (自适应路径检测)
- 降级回退: 使用内置的硬编码验证数据 (与论文 Table 3/4 一致)

【输出】
- viz/validation_plot.png

【作者】Antigravity (Updated for Open Source Release)
【更新日期】2026-01-17
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import argparse
from pathlib import Path

# 导入统一配置
try:
    from plot_config import setup_fonts_and_style, save_figure
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from plot_config import setup_fonts_and_style, save_figure

# ================= Configuration =================

def get_default_paths():
    """Smartly determine default paths"""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent 
    
    csv_candidates = [
        project_root / "result" / "gender_validity_results.csv",
        project_root / "data" / "gender_validity_results.csv",
        Path("gender_validity_results.csv")
    ]
    default_csv = None
    for p in csv_candidates:
        if p.exists(): default_csv = p; break
        
    viz_dir = project_root / "result" / "viz"
    return default_csv, viz_dir

# Mappings
CAT_TRANSLATION = {
    "配偶婚姻": "Spouse",
    "亲属称谓": "Kinship",
    "基础称谓": "Basic",
    "生物性别": "Biology",
    "哲学概念": "Philosophy",
    # English to English (Identity)
    "Spouse": "Spouse",
    "Kinship": "Kinship",
    "Basic": "Basic",
    "Biology": "Biology",
    "Philosophy": "Philosophy"
}

WORD_META = {
    "郎": ("Lang", "Gentleman/Husband"),
    "娘": ("Niang", "Lady/Wife"),
    "叔": ("Shu", "Paternal Uncle"),
    "姑": ("Gu", "Paternal Aunt"),
    "舅": ("Jiu", "Maternal Uncle"),
    "姨": ("Yi", "Maternal Aunt"),
    "公": ("Gong", "Lord/Public"),
    "婆": ("Po", "Crone/Female"),
    "翁": ("Weng", "Elderly Man"),
    "媪": ("Ao", "Elderly Woman"),
    "雄": ("Xiong", "Male (Animal)"),
    "雌": ("Ci", "Female (Animal)"),
    "牡": ("Mu", "Male (Animal)"),
    "牝": ("Pin", "Female (Animal)"),
    "乾": ("Qian", "Heaven/Male"),
    "坤": ("Kun", "Earth/Female"),
    "阳": ("Yang", "Yang"),
    "阴": ("Yin", "Yin"),
}

# Semantic Gender Mapping (Male/Female)
SEMANTIC_GENDER = {
    "郎": "Male", "娘": "Female",
    "叔": "Male", "姑": "Female", "舅": "Male", "姨": "Female",
    "公": "Male", "婆": "Female", "翁": "Male", "媪": "Female",
    "雄": "Male", "雌": "Female", "牡": "Male", "牝": "Female",
    "乾": "Male", "坤": "Female", "阳": "Male", "阴": "Female"
}

# Desired Order of Categories
CAT_ORDER = ["Spouse", "Kinship", "Basic", "Biology", "Philosophy"]

# ================= Fallback Data =================
FALLBACK_DATA = [
    # Category, Term, Character, Meaning, Bias
    ("Spouse", "Lang", "郎", "Gentleman/Husband", -0.072),
    ("Spouse", "Niang", "娘", "Lady/Wife", -0.353),
    ("Kinship", "Shu", "叔", "Paternal Uncle", 0.163),
    ("Kinship", "Gu", "姑", "Paternal Aunt", -0.239),
    ("Kinship", "Jiu", "舅", "Maternal Uncle", -0.109),
    ("Kinship", "Yi", "姨", "Maternal Aunt", -0.289),
    ("Basic", "Gong", "公", "Lord/Public", 0.136),
    ("Basic", "Po", "婆", "Crone/Female", -0.023),
    ("Basic", "Weng", "翁", "Elderly Man", 0.010),
    ("Basic", "Ao", "媪", "Elderly Woman", -0.234),
    ("Biology", "Xiong", "雄", "Male (Animal)", 0.185),
    ("Biology", "Ci", "雌", "Female (Animal)", -0.160),
    ("Biology", "Mu", "牡", "Male (Animal)", 0.173),
    ("Biology", "Pin", "牝", "Female (Animal)", 0.014),
    ("Philosophy", "Qian", "乾", "Heaven/Male", 0.116),
    ("Philosophy", "Kun", "坤", "Earth/Female", -0.045),
    ("Philosophy", "Yang", "阳", "Yang", 0.065),
    ("Philosophy", "Yin", "阴", "Yin", -0.095),
]

FALLBACK_STATS = {
    "Spouse":     (0.212, 0.001, "**"),
    "Kinship":    (0.200, 0.001, "**"),
    "Basic":      (0.101, 0.033, "*"),
    "Biology":    (0.133, 0.002, "**"),
    "Philosophy": (0.081, 0.109, ""),
}

def load_data(csv_path):
    if csv_path and os.path.exists(csv_path):
        print(f"Loading data from CSV: {csv_path}")
        try:
            raw_df = pd.read_csv(csv_path)
            
            # Process Word Data
            word_rows = raw_df[raw_df['Type'] == 'Word'].copy()
            plot_data = []
            
            for _, row in word_rows.iterrows():
                char = row['Name']
                cn_cat = row['Category']
                bias = row['Obs_Score']
                
                if char not in WORD_META:
                    # Skip unknown
                    continue
                    
                term, meaning = WORD_META[char]
                en_cat = CAT_TRANSLATION.get(cn_cat, cn_cat)
                gender = SEMANTIC_GENDER.get(char, "Unknown")
                
                plot_data.append({
                    "Category": en_cat,
                    "Term": term,
                    "Cha": char,
                    "Meaning": meaning,
                    "Bias": bias,
                    "Gender": gender
                })
            
            df = pd.DataFrame(plot_data)
            
            # Process Category Stats
            cat_rows = raw_df[raw_df['Type'] == 'Category'].copy()
            cat_stats = {}
            for _, row in cat_rows.iterrows():
                cn_cat = row['Name']
                en_cat = CAT_TRANSLATION.get(cn_cat, cn_cat)
                sig = row['Significance'] if not pd.isna(row['Significance']) else ""
                cat_stats[en_cat] = (row['Mean_Abs_Bias'], row['P_Value'], sig)
                
            return df, cat_stats
        except Exception as e:
            print(f"Error reading CSV: {e}. Falling back to built-in data.")
            
    print("Using built-in fallback data.")
    # Convert fallback data to DF
    plot_data = []
    for row in FALLBACK_DATA:
        cat, term, char, meaning, bias = row
        gender = SEMANTIC_GENDER.get(char, "Unknown")
        plot_data.append({
            "Category": cat,
            "Term": term,
            "Cha": char,
            "Meaning": meaning,
            "Bias": bias,
            "Gender": gender
        })
    df = pd.DataFrame(plot_data)
    return df, FALLBACK_STATS

def main():
    default_csv, default_viz_dir = get_default_paths()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=str(default_csv) if default_csv else None, help='Path to gender_validity_results.csv')
    parser.add_argument('--output-dir', default=str(default_viz_dir))
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_fonts_and_style()
    
    # 1. Load Data
    df, cat_stats = load_data(args.input)
    
    if df.empty:
        print("No data available to plot.")
        return
        
    # 2. Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colors
    color_male = '#377eb8'  # Blue
    color_female = '#e41a1c'  # Red
    
    y_start = 0
    yticks = []
    yticklabels = []
    
    for cat in CAT_ORDER:
        if cat not in df["Category"].unique():
            continue
            
        sub_df = df[df["Category"] == cat]
        
        # Get stats
        if cat in cat_stats:
            stat = cat_stats[cat] # (mean, p, sig)
            p_val = stat[1]
            # Handle string type p_value from fallback or CSV
            p_val_str = f"{p_val:.3f}" if isinstance(p_val, (int, float)) else str(p_val)
            header_text = f"{cat} (Mean Abs: {stat[0]:.3f}, p={p_val_str}{stat[2]})"
        else:
            header_text = f"{cat}"
        
        # Category Header
        header_y = y_start + len(sub_df) + 0.5
        ax.text(-0.5, header_y, header_text, fontsize=12, fontweight='bold', va='center', ha='left')
        
        # Plot dots
        for i, row in enumerate(sub_df.itertuples()):
            y = y_start + i
            
            # Color logic based on Semantic Gender
            if row.Gender == 'Male':
                color = color_male
            elif row.Gender == 'Female':
                color = color_female
            else:
                color = 'gray'
                
            ax.scatter(row.Bias, y, color=color, s=100, zorder=3)
            
            # Add separating lines
            ax.hlines(y, -0.4, 0.4, color='gray', alpha=0.1, linewidth=1, zorder=1)
            
            # Label: Char + Meaning
            label = f"{row.Cha} ({row.Meaning})"
            yticks.append(y)
            yticklabels.append(label)
        
        y_start += len(sub_df) + 2  # Gap between categories
    
    # Axes formatting
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=10)
    ax.set_ylim(-1, y_start)
    ax.invert_yaxis()  # Top to bottom
    
    ax.set_xlim(-0.45, 0.45)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    
    ax.set_xlabel("Gender Bias Projection (Feminine <---> Masculine)", fontsize=11)
    ax.set_title("Gender Axis Validation: Control Terms Projection", fontsize=14, pad=20)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Male Terms', markerfacecolor=color_male, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Female Terms', markerfacecolor=color_female, markersize=10)
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    
    # Add grid
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Save
    plt.tight_layout()
    outfile = output_dir / "validation_plot.png"
    save_figure(plt.gcf(), outfile)
    print(f"Chart saved to {outfile}")

if __name__ == "__main__":
    main()
