"""
================================================================================
性别词向量收敛性测试脚本
================================================================================

【功能描述】
测试在不同样本量下，性别词的动态向量（上下文向量）平均值的收敛性，
以确定最优抽样数量。这是构建分朝代性别轴线的前置实验。

主要步骤：
1. 从语料库中提取包含性别词的所有句子
2. 统计每个朝代各性别词的出现次数
3. 进行收敛性测试：抽取不同样本量（10, 50, 100, 200, 300, 500），
   计算平均向量，并计算相邻样本量之间的余弦相似度
4. 绘制收敛曲线，确定最优抽样数量

【输入】
- data/核心古籍/核心古籍-*-*.txt - 分朝代的语料文件
- static/woman-man/basic.txt - 性别词对文件

【输出】
- result/20251230-convergence-test/word_counts_by_dynasty.txt - 各朝代性别词统计
- result/20251230-convergence-test/convergence_results.txt - 收敛性测试结果
- result/20251230-convergence-test/convergence_curves.png - 收敛曲线图
- result/20251230-convergence-test/convergence_data.json - 详细数据（供后续分析）

【参数说明】
- DATA_DIR: 语料目录
- SAMPLE_SIZES: 测试的样本量列表
- TARGET_WORDS: 待测试的性别词列表
- SENTENCE_SPLITTER: 句子分隔符（中文句号）

【依赖库】
- transformers: BERT模型加载
- torch: 深度学习框架
- numpy: 数值计算
- matplotlib: 可视化
- re: 正则表达式

【作者】AI辅助生成
【创建日期】2025-12-30
【最后修改】2025-12-30 14:04 - 修复长句子截断导致的索引越界问题

【修改历史】
- 2025-12-30 13:58: 创建脚本，实现收敛性测试功能
- 2025-12-30 14:04: 修复get_contextual_vector函数，添加序列长度检查，避免长句子被截断时的IndexError
================================================================================
"""

import os
import re
import json
import random
from typing import List, Dict, Tuple
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

# 每个实验使用独立的结果子文件夹
EXPERIMENT_NAME = "20251230-convergence-test"
RESULT_DIR = os.path.join("result", EXPERIMENT_NAME)

# 输出文件
WORD_COUNTS_FILE = os.path.join(RESULT_DIR, "word_counts_by_dynasty.txt")
CONVERGENCE_RESULTS_FILE = os.path.join(RESULT_DIR, "convergence_results.txt")
CONVERGENCE_CURVES_FILE = os.path.join(RESULT_DIR, "convergence_curves.png")
CONVERGENCE_DATA_FILE = os.path.join(RESULT_DIR, "convergence_data.json")

# 测试的样本量
SAMPLE_SIZES = [10, 50, 100, 150, 200, 250]

# 句子分隔符（中文句号）
SENTENCE_SPLITTER = '。'

# 中文字体路径
FONT_PATH = r'C:\\Windows\\Fonts\\simkai.ttf'

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


def get_contextual_vector(tokenizer, model, device, sentence: str, target_word: str) -> np.ndarray:
    """
    获取目标词在句子上下文中的动态向量
    
    Args:
        sentence: 包含目标词的句子
        target_word: 目标词
    
    Returns:
        目标词的上下文向量（取所有token的平均）
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
    # actual_seq_len包含[CLS]和[SEP]，所以最大有效位置是actual_seq_len-1
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

def load_gender_words(filepath: str) -> List[str]:
    """读取性别词列表"""
    words = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            words.update(parts)
    return sorted(list(words))


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


def count_word_occurrences_by_dynasty(data_dir: str, target_words: List[str]) -> Dict[str, Dict[str, int]]:
    """
    统计每个朝代各性别词的出现次数
    
    Returns:
        {朝代: {词: 出现次数}}
    """
    counts = defaultdict(lambda: defaultdict(int))
    
    print("统计各朝代性别词出现次数...")
    print("=" * 60)
    
    # 遍历所有语料文件
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith('.txt'):
            continue
        
        # 解析朝代代码
        # 文件名格式: 核心古籍-A先秦两汉-02儒藏.txt
        match = re.match(r'核心古籍-([A-G])', filename)
        if not match:
            continue
        
        dynasty_code = match.group(1)
        dynasty_name = DYNASTY_MAP.get(dynasty_code, dynasty_code)
        
        filepath = os.path.join(data_dir, filename)
        
        # 统计每个目标词
        for word in target_words:
            sentences = extract_sentences_with_word(filepath, word)
            counts[dynasty_name][word] += len(sentences)
        
        print(f"✓ 处理完成: {dynasty_name} - {filename}")
    
    print("=" * 60)
    return dict(counts)


# ==================== 收敛性测试函数 ====================

def convergence_test_for_word(
    tokenizer, 
    model, 
    device,
    all_sentences: List[str], 
    target_word: str,
    sample_sizes: List[int]
) -> Dict[int, np.ndarray]:
    """
    对单个词进行收敛性测试
    
    Args:
        all_sentences: 包含目标词的所有句子
        target_word: 目标词
        sample_sizes: 要测试的样本量列表
    
    Returns:
        {样本量: 平均向量}
    """
    results = {}
    
    # 确保有足够的句子
    max_sample_size = max(sample_sizes)
    if len(all_sentences) < max_sample_size:
        print(f"  ⚠ 警告: 句子数量({len(all_sentences)})小于最大样本量({max_sample_size})")
        # 调整sample_sizes
        sample_sizes = [s for s in sample_sizes if s <= len(all_sentences)]
    
    for size in sample_sizes:
        # 随机抽样
        sampled_sentences = random.sample(all_sentences, size)
        
        # 提取向量
        vectors = []
        for sent in sampled_sentences:
            vec = get_contextual_vector(tokenizer, model, device, sent, target_word)
            if vec is not None:
                vectors.append(vec)
        
        if not vectors:
            print(f"  ⚠ 样本量{size}: 无法提取任何向量")
            continue
        
        # 计算平均
        avg_vector = np.mean(vectors, axis=0)
        results[size] = avg_vector
        
        print(f"  样本量 {size:3d}: 成功提取 {len(vectors)} 个向量")
    
    return results


def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算余弦相似度"""
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)


# ==================== 可视化函数 ====================

def plot_convergence_curves(convergence_data: Dict[str, Dict[int, float]], save_path: str):
    """
    绘制收敛曲线
    
    Args:
        convergence_data: {词: {样本量: 与前一样本量的相似度}}
    """
    # 设置中文字体
    custom_font = mpl.font_manager.FontProperties(fname=FONT_PATH)
    plt.rcParams['font.sans-serif'] = custom_font.get_name()
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 为每个词绘制曲线
    for word, similarities in convergence_data.items():
        sizes = sorted(similarities.keys())
        sims = [similarities[s] for s in sizes]
        
        ax.plot(sizes, sims, marker='o', linewidth=2, markersize=8, label=word)
    
    # 添加参考线（相似度 = 0.99）
    ax.axhline(y=0.99, color='red', linestyle='--', linewidth=1.5, label='收敛阈值 (0.99)')
    
    ax.set_xlabel('样本量', fontproperties=custom_font, fontsize=14)
    ax.set_ylabel('与前一样本量的余弦相似度', fontproperties=custom_font, fontsize=14)
    ax.set_title('性别词向量收敛性测试', fontproperties=custom_font, fontsize=16, fontweight='bold')
    ax.legend(prop=custom_font, fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 设置y轴范围
    ax.set_ylim([0.95, 1.0])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 收敛曲线已保存: {save_path}")


# ==================== 保存结果函数 ====================

def save_word_counts(counts: Dict[str, Dict[str, int]], filepath: str):
    """保存词频统计结果"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("各朝代性别词出现次数统计\n")
        f.write("=" * 60 + "\n\n")
        
        for dynasty in sorted(counts.keys()):
            f.write(f"【{dynasty}】\n")
            f.write("-" * 60 + "\n")
            
            word_counts = counts[dynasty]
            for word in sorted(word_counts.keys()):
                count = word_counts[word]
                f.write(f"  {word}: {count:6d} 次\n")
            
            total = sum(word_counts.values())
            f.write(f"\n  总计: {total:6d} 次\n\n")
    
    print(f"✓ 词频统计已保存: {filepath}")


def save_convergence_results(
    convergence_data: Dict[str, Dict[int, float]],
    filepath: str
):
    """保存收敛性测试结果"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("性别词向量收敛性测试结果\n")
        f.write("=" * 60 + "\n\n")
        
        for word in sorted(convergence_data.keys()):
            f.write(f"【{word}】\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'样本量':<10} {'与前一样本量的相似度':<20} {'是否收敛(>0.99)'}\n")
            f.write("-" * 60 + "\n")
            
            similarities = convergence_data[word]
            for size in sorted(similarities.keys()):
                sim = similarities[size]
                converged = "✓" if sim > 0.99 else "×"
                f.write(f"{size:<10} {sim:<20.6f} {converged}\n")
            
            f.write("\n")
    
    print(f"✓ 收敛性测试结果已保存: {filepath}")


# ==================== 主函数 ====================

def main():
    print("=" * 60)
    print("性别词向量收敛性测试")
    print("=" * 60)
    print()
    
    # 创建结果目录
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # 1. 读取性别词列表
    print("读取性别词列表...")
    gender_words = load_gender_words(BASIC_GENDER_FILE)
    print(f"✓ 共读取 {len(gender_words)} 个性别词: {', '.join(gender_words)}\n")
    
    # 2. 统计各朝代性别词出现次数
    word_counts = count_word_occurrences_by_dynasty(DATA_DIR, gender_words)
    save_word_counts(word_counts, WORD_COUNTS_FILE)
    
    # 3. 打印统计摘要
    print("\n统计摘要:")
    print("-" * 60)
    total_by_word = defaultdict(int)
    for dynasty, counts in word_counts.items():
        for word, count in counts.items():
            total_by_word[word] += count
    
    for word in sorted(total_by_word.keys()):
        print(f"  {word}: {total_by_word[word]:7d} 次（全部朝代）")
    print()
    
    # 4. 收敛性测试（在全部语料上）
    print("\n开始收敛性测试...")
    print("=" * 60)
    
    # 加载模型
    tokenizer, model, device = load_sikubert(MODEL_NAME)
    
    # 收集全部语料中的句子
    all_sentences_by_word = {}
    for word in gender_words:
        print(f"\n收集包含「{word}」的所有句子...")
        all_sentences = []
        
        for filename in sorted(os.listdir(DATA_DIR)):
            if not filename.endswith('.txt'):
                continue
            filepath = os.path.join(DATA_DIR, filename)
            sentences = extract_sentences_with_word(filepath, word)
            all_sentences.extend(sentences)
        
        print(f"  ✓ 共收集到 {len(all_sentences)} 个句子")
        all_sentences_by_word[word] = all_sentences
    
    # 对每个词进行收敛性测试
    print("\n\n进行收敛性测试...")
    print("=" * 60)
    
    convergence_vectors = {}  # {词: {样本量: 向量}}
    convergence_similarities = {}  # {词: {样本量: 相似度}}
    
    for word in sorted(gender_words):
        print(f"\n测试词: {word}")
        print("-" * 60)
        
        sentences = all_sentences_by_word[word]
        
        if len(sentences) < min(SAMPLE_SIZES):
            print(f"  ⚠ 跳过（句子数不足）")
            continue
        
        # 收敛性测试
        vectors_by_size = convergence_test_for_word(
            tokenizer, model, device,
            sentences, word, SAMPLE_SIZES
        )
        
        convergence_vectors[word] = vectors_by_size
        
        # 计算相似度
        similarities = {}
        sorted_sizes = sorted(vectors_by_size.keys())
        
        for i in range(1, len(sorted_sizes)):
            prev_size = sorted_sizes[i-1]
            curr_size = sorted_sizes[i]
            
            sim = calculate_cosine_similarity(
                vectors_by_size[prev_size],
                vectors_by_size[curr_size]
            )
            similarities[curr_size] = sim
            
            status = "✓ 收敛" if sim > 0.99 else "  待收敛"
            print(f"  {prev_size:3d} -> {curr_size:3d}: 相似度 = {sim:.6f} {status}")
        
        convergence_similarities[word] = similarities
    
    # 5. 保存结果
    print("\n\n保存结果...")
    print("=" * 60)
    
    save_convergence_results(convergence_similarities, CONVERGENCE_RESULTS_FILE)
    
    # 保存详细数据
    data_to_save = {
        'sample_sizes': SAMPLE_SIZES,
        'convergence_similarities': {w: {str(k): v for k, v in sims.items()} 
                                     for w, sims in convergence_similarities.items()},
        'total_sentences': {w: len(all_sentences_by_word[w]) for w in all_sentences_by_word}
    }
    
    with open(CONVERGENCE_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 详细数据已保存: {CONVERGENCE_DATA_FILE}")
    
    # 绘制收敛曲线
    plot_convergence_curves(convergence_similarities, CONVERGENCE_CURVES_FILE)
    
    print("\n" + "=" * 60)
    print("✓ 收敛性测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    # 设置随机种子以保证可复现性
    random.seed(42)
    np.random.seed(42)
    
    main()
