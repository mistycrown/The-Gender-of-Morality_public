"""
================================================================================
绘图配置模块 - 统一管理所有可视化脚本的配置
================================================================================

【功能描述】
提供统一的绘图配置,包括:
1. SciencePlots样式设置
2. 字体配置(中文宋体、英文Times New Roman)
3. 道德类别映射(英文维度名称)
4. 图表尺寸和输出格式

【作者】Antigravity
【创建日期】2026-01-14
================================================================================
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

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
    
    使用SciencePlots库 + 自定义字体:
    - 中文: 宋体 (SimSun)
    - 英文: Times New Roman
    - 字号: 7-12pt范围
    """
    try:
        # 导入并应用SciencePlots样式
        import scienceplots
        plt.style.use(['science', 'no-latex'])
        print("✓ 已应用SciencePlots样式")
    except ImportError:
        print("⚠️  SciencePlots未安装,使用默认样式")
        print("   安装方法: pip install SciencePlots")
    
    # 设置字体
    # Times New Roman for English
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'SimSun']
    
    # SimSun (宋体) for Chinese
    # 尝试加载系统中的宋体
    FONT_CANDIDATES = [
        r'C:\Windows\Fonts\simsun.ttc',
        r'C:\Windows\Fonts\SimSun.ttf',
        '/System/Library/Fonts/STSong.ttf',  # macOS
    ]
    
    chinese_font = None
    for font_path in FONT_CANDIDATES:
        if Path(font_path).exists():
            try:
                chinese_font = mpl.font_manager.FontProperties(fname=font_path)
                # 将中文字体添加到sans-serif列表
                plt.rcParams['font.sans-serif'] = [chinese_font.get_name()] + plt.rcParams['font.sans-serif']
                print(f"✓ 已加载中文字体: {font_path}")
                break
            except Exception as e:
                print(f"⚠️  加载字体失败 {font_path}: {e}")
    
    if chinese_font is None:
        # 回退到系统字体
        plt.rcParams['font.sans-serif'] = ['SimSun', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        print("⚠️  使用系统默认中文字体")
    
    plt.rcParams['axes.unicode_minus'] = False
    
    # 字号设置 (7-12pt范围)
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8
    
    # 坐标轴样式 (黑色, 0.5-1.5pt)
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    
    return chinese_font

# ==================== 输出格式配置 ====================

def save_figure(fig, base_path, formats=['svg', 'pdf', 'png'], dpi=600):
    """
    保存图表为多种格式
    
    Args:
        fig: matplotlib图表对象
        base_path: 基础路径(不含扩展名)
        formats: 输出格式列表
        dpi: PNG分辨率
    """
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        output_path = base_path.with_suffix(f'.{fmt}')
        if fmt == 'png':
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        elif fmt == 'pdf':
            fig.savefig(output_path, format='pdf', bbox_inches='tight')
        elif fmt == 'svg':
            fig.savefig(output_path, format='svg', bbox_inches='tight')
        print(f"  ✓ 已保存: {output_path}")

# ==================== 翻译映射 ====================

# 朝代英文映射
DYNASTY_MAPPING_EN = {
    '先秦两汉': 'Pre-Qin & Han',
    '魏晋南北朝': 'Wei, Jin, N&S',
    '隋唐': 'Sui & Tang',
    '宋': 'Song',
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
