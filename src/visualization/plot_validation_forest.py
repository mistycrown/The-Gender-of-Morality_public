"""
================================================================================
性别轴线验证可视化脚本
================================================================================
每次更新代码文件时请更新代码头注释！

【功能描述】
绘制性别轴线验证的森林图（Forest Plot）。
将控制词（Control Terms）在性别轴上的投影偏差（Projection Bias）可视化，替代原论文中的表格形式。
展示不同语义类别（Spouse, Kinship, Basic, Biology, Philosophy）的偏差分布及显著性标记。

【输入】
- 代码内硬编码的验证数据（来自论文 Table 3 和 Table 4）

【输出】
- doc/article/arXiv-2406.00278v2/image/validation_plot.png - 验证结果可视化图表

【参数说明】
- 字体设置：SimHei, Microsoft YaHei, Arial Unicode MS
- 颜色设置：蓝色（男性倾向），红色（女性倾向）

【依赖库】
- matplotlib: 绘图核心库
- pandas: 数据处理
- seaborn: 样式优化

【作者】AI辅助生成
【创建日期】2026-01-15
【最后修改】2026-01-15 - 初始版本，实现森林图绘制功能

【修改历史】
- 2026-01-15: 创建脚本，替代原Latex表格可视化
================================================================================
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.font_manager as fm
import os

# Configure font for Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# Data Input
data = [
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

df = pd.DataFrame(data, columns=["Category", "Term", "Cha", "Meaning", "Bias"])

# Category Stats (Mean Abs Bias, P-value)
cat_stats = {
    "Spouse":     (0.212, 0.001, "**"),
    "Kinship":    (0.200, 0.001, "**"),
    "Basic":      (0.101, 0.033, "*"),
    "Biology":    (0.133, 0.002, "**"),
    "Philosophy": (0.081, 0.109, ""),
}

# Setup Plot
fig, ax = plt.subplots(figsize=(10, 8))

# Define colors
color_male = '#377eb8'  # Blue
color_female = '#e41a1c'  # Red

# Plot each category
categories = ["Spouse", "Kinship", "Basic", "Biology", "Philosophy"]
y_start = 0

yticks = []
yticklabels = []

for cat in categories:
    sub_df = df[df["Category"] == cat]
    stat = cat_stats[cat]
    
    # Category Header
    header_y = y_start + len(sub_df) + 0.5
    label_text = f"{cat} (Mean Abs: {stat[0]:.3f}, p={stat[1]}{stat[2]})"
    ax.text(-0.5, header_y, label_text, fontsize=12, fontweight='bold', va='center', ha='left')
    
    # Plot dots
    for i, row in enumerate(sub_df.itertuples()):
        y = y_start + i
        color = color_male if row.Bias > 0 else color_female
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
    Line2D([0], [0], marker='o', color='w', label='Masculine Bias', markerfacecolor=color_male, markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Feminine Bias', markerfacecolor=color_female, markersize=10)
]
ax.legend(handles=legend_elements, loc='upper right')

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False) # Clean look
ax.tick_params(axis='y', length=0) # Hide ticks

# Add grid
ax.grid(axis='x', linestyle='--', alpha=0.3)

# Save
output_path = Path("validation_plot.png")
if Path("src").exists():
    output_path = Path("result") / "validation_plot.png"
    output_path.parent.mkdir(exist_ok=True)
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Chart saved to {output_path}")
