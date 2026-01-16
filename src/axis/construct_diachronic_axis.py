"""
================================================================================
历时性别轴线计算脚本
================================================================================

【功能描述】
计算每个朝代的性别轴线，使用动态向量（上下文向量）代替静态词向量。
这样可以捕捉不同历史时期性别概念的语义变化。

主要步骤：
1. 从每个朝代的语料中，为每个性别词随机抽取N个句子（默认200）
2. 使用SikuBERT提取这些句子中目标词的动态向量
3. 计算平均向量作为该词在该朝代的标准向量
4. 使用配对相减法计算每个朝代的性别轴线
5. 使用测试词验证每个朝代的性别轴线

【输入】
- data/核心古籍/核心古籍-*-*.txt - 分朝代的语料文件
- static/woman-man/basic.txt - 性别词对文件

【输出】
- result/20251230-diachronic-axis/dynasty_gender_vectors.json - 各朝代性别词的标准向量
- result/20251230-diachronic-axis/dynasty_gender_axes.npy - 各朝代的性别轴线
- result/20251230-diachronic-axis/validation_results.txt - 验证报告
- result/20251230-diachronic-axis/axis_comparison.png - 轴线对比可视化

【参数说明】
- SAMPLE_SIZE: 每个词在每个朝代的抽样数量（默认200）
- TEST_WORDS: 用于验证的测试词列表

【依赖库】
- transformers: BERT模型加载
- torch: 深度学习框架
- numpy: 数值计算
- matplotlib: 可视化

【作者】AI辅助生成
【创建日期】2025-12-30
【最后修改】2025-12-30 14:06 - 初始版本

【修改历史】
- 2025-12-30 14:06: 创建脚本，实现历时性别轴线计算功能
================================================================================
"""

import os
import re
import json
import random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import matplotlib as mpl

# ==================== 配置区域 ====================
MODEL_NAME = "SIKU-BERT/sikubert"
DATA_DIR = r"data\核心古籍"
BASIC_GENDER_FILE = r"static\woman-man\basic.txt"

# 实验专属文件夹
EXPERIMENT_NAME = "20251230-diachronic-axis"
RESULT_DIR = os.path.join("result", EXPERIMENT_NAME)

# 输出文件
GENDER_VECTORS_FILE = os.path.join(RESULT_DIR, "dynasty_gender_vectors.json")
GENDER_AXES_FILE = os.path.join(RESULT_DIR, "dynasty_gender_axes.npy")
VALIDATION_FILE = os.path.join(RESULT_DIR, "validation_results.txt")
COMPARISON_PLOT_FILE = os.path.join(RESULT_DIR, "axis_comparison.png")
DETAILED_REPORT_FILE = os.path.join(RESULT_DIR, "detailed_report.txt")

# 抽样参数
SAMPLE_SIZE = 200  # 每个词在每个朝代抽取的句子数

# 句子分隔符
SENTENCE_SPLITTER = '。'

# 测试词（用于验证性别轴线）
TEST_WORDS = [
    # 基础称谓
    ('公', '男性称谓', 'positive'),
    ('婆', '女性称谓', 'negative'),
    # 哲学概念
    ('乾', '阳/男性', 'positive'),
    ('坤', '阴/女性', 'negative'),
    ('阳', '阳性', 'positive'),
    ('阴', '阴性', 'negative'),
    # 亲属称谓
    ('叔', '男性长辈', 'positive'),
    ('姑', '女性长辈', 'negative'),
    ('舅', '男性长辈', 'positive'),
    ('姨', '女性长辈', 'negative'),
    # 配偶/婚姻
    ('郎', '男性配偶', 'positive'),
    ('娘', '女性配偶', 'negative'),
    # 社会角色
    ('翁', '男性老者', 'positive'),
    ('媪', '女性老者', 'negative'),
    # 生物性别
    ('雄', '雄性', 'positive'),
    ('雌', '雌性', 'negative'),
    ('牡', '雄性兽', 'positive'),
    ('牝', '雌性兽', 'negative'),
]

# 中文字体路径
FONT_PATH = r'C:\Windows\Fonts\simkai.ttf'

# ==================== 朝代映射 ====================
DYNASTY_MAP = {
    'A': '先秦两汉',
    'B': '魏晋南北朝',
    'C': '隋唐',
    'D': '宋',
    'E': '元',
    'F': '明',
    'G': '清'
}

DYNASTY_ORDER = ['先秦两汉', '魏晋南北朝', '隋唐', '宋', '元', '明', '清']

# ==================== SikuBERT 工具函数 ====================

def load_sikubert(model_name_or_path: str = MODEL_NAME, device: str = None):
    """加载SikuBERT模型（用于动态向量提取）"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"加载模型: {model_name_or_path}")
    print(f"设备: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)
    model.eval()
    model.to(device)
    
    print(f"✓ 模型加载完成\n")
    return tokenizer, model, device


def get_contextual_vector(tokenizer, model, device, sentence: str, target_word: str) -> Optional[np.ndarray]:
    """
    获取目标词在句子上下文中的动态向量
    
    Args:
        sentence: 包含目标词的句子
        target_word: 目标词
    
    Returns:
        目标词的上下文向量（取所有token的平均），如果提取失败返回None
    """
    # 找到目标词在句子中的位置
    if target_word not in sentence:
        return None
    
    # Tokenize
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 获取实际的序列长度（可能因截断而小于512）
    actual_seq_len = inputs['input_ids'].shape[1]
    
    # 获取词在tokenized序列中的位置
    tokens = tokenizer.tokenize(sentence)
    target_tokens = tokenizer.tokenize(target_word)
    
    # 如果tokenize后的序列被截断，检查目标词是否还在
    if len(tokens) > 510:  # 510 = 512 - 2 (CLS and SEP)
        tokens = tokens[:510]  # 模拟截断
    
    # 寻找目标词的token位置
    target_positions = []
    for i in range(len(tokens) - len(target_tokens) + 1):
        if tokens[i:i+len(target_tokens)] == target_tokens:
            # +1 因为有[CLS] token
            target_positions = list(range(i+1, i+1+len(target_tokens)))
            break
    
    if not target_positions:
        return None
    
    # 检查target_positions是否都在有效范围内
    if max(target_positions) >= actual_seq_len:
        # 目标词被截断了，跳过这个句子
        return None
    
    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs)
        # 取最后一层的hidden states
        last_hidden_state = outputs.last_hidden_state  # [1, seq_len, hidden_size]
    
    # 提取目标词位置的向量并取平均
    target_vectors = last_hidden_state[0, target_positions, :]  # [num_target_tokens, hidden_size]
    avg_vector = target_vectors.mean(dim=0).cpu().numpy()  # [hidden_size]
    
    return avg_vector


# ==================== 语料处理函数 ====================

def load_gender_pairs(filepath: str) -> List[Tuple[str, str]]:
    """读取性别词对"""
    pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                male_word = parts[0].strip()
                female_word = parts[1].strip()
                pairs.append((male_word, female_word))
    return pairs


def extract_sentences_with_word(filepath: str, target_word: str) -> List[str]:
    """从文本文件中提取包含目标词的所有句子"""
    sentences = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 按句号分割句子
        all_sentences = text.split(SENTENCE_SPLITTER)
        
        # 筛选包含目标词的句子
        for sent in all_sentences:
            sent = sent.strip()
            if target_word in sent and len(sent) > 0:
                sentences.append(sent + SENTENCE_SPLITTER)  # 保留句号
    
    except Exception as e:
        print(f"  ⚠ 读取文件失败 {filepath}: {e}")
    
    return sentences


def collect_sentences_by_dynasty(data_dir: str, target_word: str) -> Dict[str, List[str]]:
    """
    收集每个朝代包含目标词的所有句子
    
    Returns:
        {朝代名: [句子列表]}
    """
    sentences_by_dynasty = defaultdict(list)
    
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith('.txt'):
            continue
        
        # 解析朝代代码
        match = re.match(r'核心古籍-([A-G])', filename)
        if not match:
            continue
        
        dynasty_code = match.group(1)
        dynasty_name = DYNASTY_MAP.get(dynasty_code, dynasty_code)
        
        filepath = os.path.join(data_dir, filename)
        sentences = extract_sentences_with_word(filepath, target_word)
        sentences_by_dynasty[dynasty_name].extend(sentences)
    
    return dict(sentences_by_dynasty)


# ==================== 向量计算函数 ====================

def compute_word_vector_for_dynasty(
    tokenizer,
    model,
    device,
    sentences: List[str],
    target_word: str,
    sample_size: int
) -> Optional[np.ndarray]:
    """
    计算某个词在某个朝代的标准向量（动态向量的平均值）
    
    Args:
        sentences: 该朝代包含该词的所有句子
        target_word: 目标词
        sample_size: 抽样数量
    
    Returns:
        标准向量，如果提取失败返回None
    """
    # 检查句子数量
    if len(sentences) < sample_size:
        print(f"    ⚠ 警告: 句子数({len(sentences)})少于样本量({sample_size})，使用全部句子")
        sampled_sentences = sentences
    else:
        # 随机抽样
        sampled_sentences = random.sample(sentences, sample_size)
    
    # 提取向量
    vectors = []
    for sent in sampled_sentences:
        vec = get_contextual_vector(tokenizer, model, device, sent, target_word)
        if vec is not None:
            vectors.append(vec)
    
    if not vectors:
        print(f"    ✗ 无法提取任何有效向量")
        return None
    
    # 计算平均
    avg_vector = np.mean(vectors, axis=0)
    
    success_rate = len(vectors) / len(sampled_sentences) * 100
    print(f"    ✓ 成功提取 {len(vectors)}/{len(sampled_sentences)} 个向量 (成功率: {success_rate:.1f}%)")
    
    return avg_vector


def calculate_gender_axis_for_dynasty(
    dynasty_vectors: Dict[str, np.ndarray],
    gender_pairs: List[Tuple[str, str]]
) -> Optional[np.ndarray]:
    """
    使用配对相减法计算某个朝代的性别轴线
    
    Args:
        dynasty_vectors: 该朝代所有性别词的向量 {词: 向量}
        gender_pairs: 性别词对列表
    
    Returns:
        归一化的性别轴线
    """
    diff_vectors = []
    
    for male_word, female_word in gender_pairs:
        if male_word not in dynasty_vectors or female_word not in dynasty_vectors:
            print(f"    ⚠ 跳过词对 {male_word}-{female_word} (向量缺失)")
            continue
        
        vec_male = dynasty_vectors[male_word]
        vec_female = dynasty_vectors[female_word]
        
        # 计算差异向量
        diff_vec = vec_male - vec_female
        diff_vectors.append(diff_vec)
        
        print(f"    ✓ {male_word} - {female_word}: ||diff|| = {np.linalg.norm(diff_vec):.4f}")
    
    if not diff_vectors:
        print(f"    ✗ 无法计算性别轴线（没有有效的词对）")
        return None
    
    # 计算平均差异向量
    gender_axis = np.mean(diff_vectors, axis=0)
    
    # 归一化
    gender_axis_normalized = gender_axis / np.linalg.norm(gender_axis)
    
    print(f"    ✓ 性别轴线计算完成 (使用了 {len(diff_vectors)} 个词对)")
    
    return gender_axis_normalized


def validate_gender_axis(
    tokenizer,
    model,
    device,
    gender_axis: np.ndarray,
    dynasty_name: str,
    test_words: List[Tuple[str, str, str]],
    dynasty_sentences: Dict[str, List[str]]
) -> List[Dict]:
    """
    验证性别轴线
    
    使用该朝代的语料计算测试词的动态向量，然后投影到性别轴线上
    """
    results = []
    
    print(f"\n  验证 {dynasty_name} 的性别轴线...")
    print("  " + "-" * 60)
    
    for word, gender, expected_sign in test_words:
        # 从该朝代的语料中提取测试词的句子
        sentences = dynasty_sentences.get(word, [])
        
        if not sentences:
            result = {
                'word': word,
                'expected_gender': gender,
                'expected_sign': expected_sign,
                'score': None,
                'passed': False,
                'reason': '该朝代无此词的句子'
            }
            print(f"  {word:<6} {gender:<12} {'N/A':<12} ✗ (无句子)")
            results.append(result)
            continue
        
        # 抽样并计算向量
        sample_size = min(50, len(sentences))  # 验证时使用较小的样本
        vec = compute_word_vector_for_dynasty(
            tokenizer, model, device, sentences, word, sample_size
        )
        
        if vec is None:
            result = {
                'word': word,
                'expected_gender': gender,
                'expected_sign': expected_sign,
                'score': None,
                'passed': False,
                'reason': '无法提取向量'
            }
            print(f"  {word:<6} {gender:<12} {'N/A':<12} ✗ (提取失败)")
            results.append(result)
            continue
        
        # 计算投影得分
        score = float(np.dot(vec, gender_axis))
        
        # 验证符号
        passed = (expected_sign == 'positive' and score > 0) or \
                 (expected_sign == 'negative' and score < 0)
        
        result = {
            'word': word,
            'expected_gender': gender,
            'expected_sign': expected_sign,
            'score': score,
            'passed': passed,
            'reason': '符合预期' if passed else '不符合预期'
        }
        
        status = '✓' if passed else '✗'
        print(f"  {word:<6} {gender:<12} {score:<12.6f} {status}")
        
        results.append(result)
    
    return results


# ==================== 可视化函数 ====================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算余弦相似度"""
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)


def plot_axis_comparison(dynasty_axes: Dict[str, np.ndarray], save_path: str):
    """
    绘制朝代间性别轴线的相似度矩阵
    """
    # 设置中文字体
    custom_font = mpl.font_manager.FontProperties(fname=FONT_PATH)
    plt.rcParams['font.sans-serif'] = custom_font.get_name()
    plt.rcParams['axes.unicode_minus'] = False
    
    # 按时间顺序排列朝代
    dynasties = [d for d in DYNASTY_ORDER if d in dynasty_axes]
    n = len(dynasties)
    
    # 计算相似度矩阵
    similarity_matrix = np.zeros((n, n))
    for i, d1 in enumerate(dynasties):
        for j, d2 in enumerate(dynasties):
            similarity_matrix[i, j] = cosine_similarity(
                dynasty_axes[d1], dynasty_axes[d2]
            )
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(similarity_matrix, cmap='YlOrRd', vmin=0.9, vmax=1.0)
    
    # 设置刻度标签
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(dynasties, fontproperties=custom_font)
    ax.set_yticklabels(dynasties, fontproperties=custom_font)
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 在每个格子中显示数值
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.4f}',
                         ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title('朝代性别轴线相似度矩阵', fontproperties=custom_font, fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax, label='余弦相似度')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 轴线对比图已保存: {save_path}")


# ==================== 保存结果函数 ====================

def save_validation_results(
    validation_by_dynasty: Dict[str, List[Dict]],
    filepath: str
):
    """保存验证结果"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("历时性别轴线验证报告\n")
        f.write("=" * 60 + "\n\n")
        
        for dynasty in DYNASTY_ORDER:
            if dynasty not in validation_by_dynasty:
                continue
            
            results = validation_by_dynasty[dynasty]
            
            f.write(f"\n【{dynasty}】\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'词汇':<8} {'预期性别':<12} {'投影得分':<12} {'验证结果'}\n")
            f.write("-" * 60 + "\n")
            
            for result in results:
                word = result['word']
                gender = result['expected_gender']
                score = result['score']
                passed = result['passed']
                reason = result['reason']
                
                score_str = f"{score:.6f}" if score is not None else "N/A"
                status = '✓' if passed else '✗'
                
                f.write(f"{word:<8} {gender:<12} {score_str:<12} {status} ({reason})\n")
            
            # 统计
            passed_count = sum(1 for r in results if r['passed'])
            total_count = len(results)
            pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0
            
            f.write(f"\n  通过率: {passed_count}/{total_count} ({pass_rate:.1f}%)\n")
    
    print(f"✓ 验证报告已保存: {filepath}")


# ==================== 主函数 ====================

def main():
    print("=" * 60)
    print("历时性别轴线计算")
    print("=" * 60)
    print()
    
    # 创建结果目录
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # 1. 读取性别词对
    print("读取性别词对...")
    gender_pairs = load_gender_pairs(BASIC_GENDER_FILE)
    all_gender_words = set()
    for male, female in gender_pairs:
        all_gender_words.add(male)
        all_gender_words.add(female)
    
    print(f"✓ 共读取 {len(gender_pairs)} 对性别词")
    print(f"✓ 涉及 {len(all_gender_words)} 个词汇: {', '.join(sorted(all_gender_words))}\n")
    
    # 2. 加载模型
    tokenizer, model, device = load_sikubert(MODEL_NAME)
    
    # 3. 检查是否已有计算结果，如果有则加载，否则重新计算
    if os.path.exists(GENDER_AXES_FILE) and os.path.exists(GENDER_VECTORS_FILE):
        print("发现已有计算结果，直接加载...")
        dynasty_axes = np.load(GENDER_AXES_FILE, allow_pickle=True).item()
        # 加载向量只是为了后续流程不报错，实际验证不需要
        with open(GENDER_VECTORS_FILE, 'r', encoding='utf-8') as f:
            vectors_save = json.load(f)
        
        # 恢复dynasty_vectors结构（仅包含范数，无法用于重算轴线，但这里也不需要重算）
        # 注意：这里我们其实不需要dynasty_vectors了，因为轴线已经有了
        dynasty_vectors = {} 
        
        print(f"✓ 已加载 {len(dynasty_axes)} 个朝代的性别轴线")
    else:
        # 为每个朝代计算性别词的标准向量
        print("=" * 60)
        print(f"开始计算各朝代性别词向量 (样本量={SAMPLE_SIZE})")
        print("=" * 60)
        
        dynasty_vectors = {}  # {朝代: {词: 向量}}
        
        for dynasty in DYNASTY_ORDER:
            print(f"\n【{dynasty}】")
            print("-" * 60)
            
            dynasty_word_vectors = {}
            
            for word in sorted(all_gender_words):
                print(f"  处理词: {word}")
                
                # 收集该朝代该词的句子
                sentences_by_dynasty = collect_sentences_by_dynasty(DATA_DIR, word)
                sentences = sentences_by_dynasty.get(dynasty, [])
                
                if not sentences:
                    print(f"    ⚠ 该朝代无此词的句子，跳过")
                    continue
                
                print(f"    找到 {len(sentences)} 个句子")
                
                # 计算标准向量
                vec = compute_word_vector_for_dynasty(
                    tokenizer, model, device, sentences, word, SAMPLE_SIZE
                )
                
                if vec is not None:
                    dynasty_word_vectors[word] = vec
            
            dynasty_vectors[dynasty] = dynasty_word_vectors
            print(f"\n  {dynasty} 完成: 成功计算 {len(dynasty_word_vectors)} 个词的向量")
        
        # 4. 计算每个朝代的性别轴线
        print("\n" + "=" * 60)
        print("计算各朝代性别轴线")
        print("=" * 60)
        
        dynasty_axes = {}  # {朝代: 性别轴线}
        
        for dynasty in DYNASTY_ORDER:
            if dynasty not in dynasty_vectors:
                continue
            
            print(f"\n【{dynasty}】")
            print("-" * 60)
            
            axis = calculate_gender_axis_for_dynasty(
                dynasty_vectors[dynasty],
                gender_pairs
            )
            
            if axis is not None:
                dynasty_axes[dynasty] = axis

        # 保存结果
        # 保存向量（JSON格式，仅保存范数用于检查）
        vectors_to_save = {}
        for dynasty, word_vecs in dynasty_vectors.items():
            vectors_to_save[dynasty] = {
                word: float(np.linalg.norm(vec))
                for word, vec in word_vecs.items()
            }
        
        with open(GENDER_VECTORS_FILE, 'w', encoding='utf-8') as f:
            json.dump(vectors_to_save, f, ensure_ascii=False, indent=2)
        print(f"✓ 性别词向量范数已保存: {GENDER_VECTORS_FILE}")
        
        # 保存性别轴线（numpy格式）
        np.save(GENDER_AXES_FILE, dynasty_axes)
        print(f"✓ 性别轴线已保存: {GENDER_AXES_FILE}")
    
    # 5. 验证性别轴线
    print("\n" + "=" * 60)
    print("验证各朝代性别轴线")
    print("=" * 60)
    
    validation_by_dynasty = {}
    
    # 先收集测试词的句子
    print("\n收集测试词的句子...")
    test_word_sentences = {}
    for word, _, _ in TEST_WORDS:
        test_word_sentences[word] = collect_sentences_by_dynasty(DATA_DIR, word)
        total = sum(len(sents) for sents in test_word_sentences[word].values())
        print(f"  {word}: {total} 个句子（全部朝代）")
    
    for dynasty in DYNASTY_ORDER:
        if dynasty not in dynasty_axes:
            continue
        
        # 准备该朝代的测试词句子
        dynasty_test_sentences = {}
        for word in test_word_sentences:
            dynasty_test_sentences[word] = test_word_sentences[word].get(dynasty, [])
        
        results = validate_gender_axis(
            tokenizer, model, device,
            dynasty_axes[dynasty],
            dynasty,
            TEST_WORDS,
            dynasty_test_sentences
        )
        
        validation_by_dynasty[dynasty] = results
    
    # 6. 保存结果
    print("\n" + "=" * 60)
    print("保存结果")
    print("=" * 60)
    
    # 保存向量（JSON格式，仅保存范数用于检查）
    vectors_to_save = {}
    for dynasty, word_vecs in dynasty_vectors.items():
        vectors_to_save[dynasty] = {
            word: float(np.linalg.norm(vec))
            for word, vec in word_vecs.items()
        }
    
    with open(GENDER_VECTORS_FILE, 'w', encoding='utf-8') as f:
        json.dump(vectors_to_save, f, ensure_ascii=False, indent=2)
    print(f"✓ 性别词向量范数已保存: {GENDER_VECTORS_FILE}")
    
    # 保存性别轴线（numpy格式）
    np.save(GENDER_AXES_FILE, dynasty_axes)
    print(f"✓ 性别轴线已保存: {GENDER_AXES_FILE}")
    
    # 保存验证结果
    save_validation_results(validation_by_dynasty, VALIDATION_FILE)
    
    # 绘制轴线对比图
    plot_axis_comparison(dynasty_axes, COMPARISON_PLOT_FILE)
    
    print("\n" + "=" * 60)
    print("✓ 历时性别轴线计算完成！")
    print("=" * 60)


if __name__ == "__main__":
    # 设置随机种子以保证可复现性
    random.seed(42)
    np.random.seed(42)
    
    main()
