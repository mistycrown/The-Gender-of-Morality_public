"""
================================================================================
绘图配置模块 - 统一管理所有可视化脚本的配置
================================================================================

【功能描述】
提供统一的绘图配置,包括:
1. 基础样式设置 (SciencePlots 可选)
2. 字体配置(中文宋体/黑体、英文Times New Roman/Arial)
3. 道德类别映射(英文维度名称)
4. 图表尺寸和输出格式

【作者】Antigravity
【创建日期】2026-01-14
================================================================================
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import platform

# ==================== 类别映射 ====================
# 基于用户提供的Dimension列

CATEGORY_MAPPING = {
    "Care": "Care / Harm",
    "Fairness": "Fairness / Partiality",
    "Loyalty": "Loyalty / Betrayal",
    "Authority": "Hierarchy / Subversion",
    "Sanctity": "Noble / Profanity",
    "Diligence": "Diligence / Laziness",
    "Modesty": "Modesty / Arrogance",
    "Valor": "Courage / Cowardice",
    "Waste": "Thrift / Extravagance",
    # 以下类别可能需要根据实际数据调整
    "Altruism": "Altruism",
    "Resilience": "Resilience",
    "Lenience": "Lenience"
}

# 朝代顺序
DYNASTY_ORDER = ['先秦两汉', '魏晋南北朝', '隋唐', '宋', '元', '明', '清']

# 道德基础排序 (基于用户提供的表格顺序)
CAT_ORDER = [
    "Care", "Authority", "Loyalty", "Fairness", "Sanctity",
    "Diligence", "Modesty", "Valor", "Waste"
]

# ==================== 翻译映射 ====================

# 朝代英文映射
DYNASTY_MAPPING_EN = {
    '先秦两汉': 'Pre-Qin & Han',
    '魏晋南北朝': 'Wei-Jin & N./S. Dynasties',
    '隋唐': 'Sui & Tang',
    '宋': 'Five Dynasties & Song',
    '元': 'Yuan',
    '明': 'Ming',
    '清': 'Qing'
}

# 极性英文映射
POLARITY_MAPPING_EN = {
    'pos': 'Virtue',
    'neg': 'Vice'
}

# 验证词类别英文映射 (用于 visualize_diachronic_axis.py)
VALIDATION_CATEGORY_MAPPING = {
    '基础称谓': 'Basic Titles',
    '哲学概念': 'Philosophical Concepts',
    '亲属称谓': 'Kinship Terms',
    '配偶/婚姻': 'Spouse / Marriage',
    '社会角色': 'Social Roles',
    '生物性别': 'Biological Sex'
}

# ==================== 图表尺寸配置 ====================
# 根据用户要求: 半版8cm, 2/3版14cm, 整版17cm

FIGURE_SIZES = {
    'half': (3.15, 2.36),      # ~8cm width (1:0.75 ratio)
    'two_thirds': (5.51, 4.13), # ~14cm width
    'full': (6.69, 5.02),      # ~17cm width
    'custom': None             # 自定义尺寸
}

# ==================== 字体配置 ====================

def setup_fonts_and_style():
    """
    设置绘图字体和样式
    
    配置策略:
    - 优先使用 Arial Unicode MS / SimHei 等常见中文字体
    - 英文字体优先使用 Times New Roman / Arial
    """
    
    # Configure font for Chinese characters (Sans-Serif fallback chain)
    # 包含了 Windows, macOS, Linux 常用中文字体
    fonts = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei', 'sans-serif']
    
    # 尝试检测系统字体
    if platform.system() == 'Windows':
        fonts = ['SimHei', 'Microsoft YaHei'] + fonts
    elif platform.system() == 'Darwin': # macOS
        fonts = ['Arial Unicode MS', 'PingFang SC'] + fonts
        
    plt.rcParams['font.sans-serif'] = fonts
    plt.rcParams['axes.unicode_minus'] = False
    
    # 全局字号设置
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    
    return None

# ==================== 输出格式配置 ====================

def save_figure(fig, base_path, formats=['svg', 'pdf', 'png'], dpi=600):
    """
    保存图表为多种格式
    
    Args:
        fig: matplotlib图表对象
        base_path: 基础路径(含文件名，不含扩展名)
        formats: 输出格式列表
        dpi: PNG分辨率
    """
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        # 如果 base_path 已经是完整路径且包含后缀，则需要处理
        # 这里假设传入的是无后缀的路径或者我们需要替换后缀
        # 为安全起见，先去掉可能的后缀再添加新后缀
        # 但 Path.with_suffix 会替换最后一个后缀
        output_path = base_path.with_suffix(f'.{fmt}')
        
        try:
            if fmt == 'png':
                fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            elif fmt == 'pdf':
                fig.savefig(output_path, format='pdf', bbox_inches='tight')
            elif fmt == 'svg':
                fig.savefig(output_path, format='svg', bbox_inches='tight')
            print(f"  ✓ Saved: {output_path}")
        except Exception as e:
            print(f"  Warning: Failed to save as {fmt}: {e}")

# ==================== 辅助函数 ====================

def get_dynasty_labels_en(dynasties):
    """获取英文朝代标签列表"""
    return [DYNASTY_MAPPING_EN.get(d, d) for d in dynasties]

def get_category_label(category, use_chinese=False):
    """
    获取类别标签
    
    Args:
        category: 原始类别名
        use_chinese: 是否使用中文(散点图中的标签)
    
    Returns:
        标签文本
    """
    # 始终返回英文维度名称 (除特殊情况外)
    return CATEGORY_MAPPING.get(category, category)
