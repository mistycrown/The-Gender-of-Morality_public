"""
================================================================================
道德词典性别偏见 MVP计算脚本 (静态版 - 全量实验)
================================================================================
每次更新代码文件时请更新代码头注释！

【功能描述】
1. 读取 static/dic/ 下的道德词典文件（全量读取，不限制数量）。
2. 使用 SikuBERT 模型提取词向量：
   - 单义词：从语料库随机抽样200条例句，按朝代分层，提取上下文向量
   - 多义词：从筛选例句中抽样200条，按朝代分层，提取上下文向量
   - 所有词汇统一使用上下文敏感向量（BERT last hidden state 平均）
3. 计算这些词在 SikuBERT 静态性别轴线上的投影得分 (Bias Score)。
4. 输出 Markdown 格式的统计表格和 JSON 数据。

【输入】
- static/dic/*.txt: 道德词典文件
- data/ngram_match_sampled/*.txt: 多义词筛选例句文件
- data/核心古籍/*.txt: 核心古籍语料库（按朝代分文件）
- result/20251230-gender-axis/gender_axis.npy: 静态性别轴线

【输出】
- result/20251230-moral-mvp-static/moral_bias_mvp_static_full.md
- result/20251230-moral-mvp-static/moral_bias_mvp_static_full.json

【参数说明】
- --dic-dir: 词典目录
- --data-dir: ngram_match 目录 (多义词例句)
- --corpus-dir: 核心古籍语料目录
- --model-path: 模型路径
- --axis-file: 性别轴线文件 (.npy)
- --output-dir: 结果输出目录
- --sample-size: 每个词抽样例句数量 (默认200)

【依赖库】
- transformers, torch, numpy, tqdm

【作者】AI辅助生成
【创建日期】2026-01-11
【最后修改】
- 2024-12-30: 创建初始版本，支持简单的词义筛选
- 2025-01-11: 优化 & 迁移 AutoDL
    1. 增加 Batch Processing (128) 优化 3090 性能
    2. 增加 AutoDL 环境自动识别 (/root/autodl-tmp)
    3. 增加本地模型自动检测 (优先于 HF 下载)
    4. 修复词典读取逻辑（忽略分隔线后内容）
    5. 增加启动提示，避免加载时假死体验
    6. 修复 IndentationError 和变量初始化问题
- 2026-01-12: 重构词向量提取逻辑
    1. 修复多义词例句格式解析（正确提取第三个方括号的相似度和朝代标记）
    2. 单义词改用上下文向量（从语料库抽样200条例句）
    3. 添加朝代分层抽样机制（stratified_sample_by_dynasty）
    4. 统一单义词和多义词的向量提取流程（都使用 get_contextual_word_vector）
    5. 移除静态词向量函数（get_word_vector）
    6. 添加语料库目录配置和自动匹配逻辑

【修改历史】
- 2026-01-11: 基于 calculate_moral_mvp_static.py 创建全量版，增加 Colab 支持和全量数据处理能力
- 2026-01-12: 重大重构，统一词向量提取逻辑，修复多义词解析，增加朝代分层抽样
================================================================================
"""

import os

# 必须在 import transformers 之前设置，否则可能无效
if os.path.exists("/root/autodl-tmp"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("⏳ 正在初始化环境 (加载 PyTorch/Transformers)...") # 提示用户正在加载库

import sys
import glob
import re
import argparse
import random
import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple
from tqdm import tqdm  # 添加进度条库

# ==================== 默认配置 ====================
DEFAULT_MODEL_NAME = "SIKU-BERT/sikubert"
DEFAULT_DIC_DIR = "static/dic"
DEFAULT_NGRAM_MATCH_DIR = r"data\ngram_match_sampled"
DEFAULT_CORPUS_DIR = r"data\核心古籍"  # 新增：语料库目录
DEFAULT_AXIS_FILE = r"result\20251230-gender-axis\gender_axis.npy"
DEFAULT_OUTPUT_DIR = r"result\20251230-moral-mvp-static"
DEFAULT_SAMPLE_SIZE = 200  # 新增：每个词抽样例句数量
POLYSEMY_THRESHOLD = 0.05  # 多义词相似度截断阈值
BATCH_SIZE = 128  # 针对 3090 优化

# colab 默认路径
COLAB_ROOT = Path("/content/drive/MyDrive/moral_bias_data")
COLAB_DIC_DIR = COLAB_ROOT / "dic"
COLAB_NGRAM_MATCH_DIR = COLAB_ROOT / "ngram_match"
COLAB_CORPUS_DIR = COLAB_ROOT / "corpus"  # 新增：Colab 语料库路径
COLAB_AXIS_FILE = COLAB_ROOT / "gender_axis/gender_axis.npy"
COLAB_OUTPUT_DIR = COLAB_ROOT / "result/20251230-moral-mvp-static"

# 朝代映射（用于朝代分层抽样）
DYNASTY_MAP = {
    '先秦两汉': 'A先秦两汉', '魏晋南北朝': 'B魏晋南北朝', '隋唐': 'C隋唐',
    '宋': 'D宋', '元': 'E元', '明': 'F明', '清': 'G清'
}
DYNASTY_ORDER = ['先秦两汉', '魏晋南北朝', '隋唐', '宋', '元', '明', '清']

# 词典文件名到类别的映射
CAT_MAP = {
    "01care.txt": "Care",
    "02aut.txt": "Authority",
    "03loy.txt": "Loyalty",
    "04fair.txt": "Fairness",
    "05san.txt": "Sanctity",
    "06lenience.txt": "Lenience",
    "07waste.txt": "Waste",
    "08altruism.txt": "Altruism",
    "09diligence.txt": "Diligence",
    "10resilience.txt": "Resilience",
    "11modesty.txt": "Modesty",
    "12valor.txt": "Valor"
}

def stratified_sample_by_dynasty(sentences: List[Tuple[str, str]], target_count: int = 200) -> List[Tuple[str, str]]:
    """按朝代分层抽样
    
    Args:
        sentences: [(朝代, 例句正文), ...] 元组列表
        target_count: 目标样本数
    
    Returns:
        抽样后的例句列表，确保各朝代比例均衡
    """
    if len(sentences) <= target_count:
        return sentences
    
    # 按朝代分组
    dynasty_groups = {}
    for dynasty, sent in sentences:
        if dynasty not in dynasty_groups:
            dynasty_groups[dynasty] = []
        dynasty_groups[dynasty].append((dynasty, sent))
    
    # 计算每个朝代应抽样的数量
    num_dynasties = len(dynasty_groups)
    if num_dynasties == 0:
        return []
    
    samples_per_dynasty = target_count // num_dynasties
    remainder = target_count % num_dynasties
    
    # 从每个朝代抽样
    result = []
    for i, (dynasty, group) in enumerate(dynasty_groups.items()):
        # 前 remainder 个朝代多抽1个
        n_samples = samples_per_dynasty + (1 if i < remainder else 0)
        n_samples = min(n_samples, len(group))
        result.extend(random.sample(group, n_samples))
    
    return result

def extract_sentences_from_corpus(corpus_dir: str, target_words: List[str], limit_per_word: int = 200) -> Dict[str, List[Tuple[str, str]]]:
    """从语料库中提取包含目标词的句子（全朝代混合）
    
    Args:
        corpus_dir: 语料库目录
        target_words: 目标词汇列表
        limit_per_word: 每个词限制抽样数量
    
    Returns:
        {词汇: [(朝代, 例句), ...], ...}
    """
    word_sents = {w: [] for w in target_words}
    
    # 扫描所有语料文件
    files = glob.glob(os.path.join(corpus_dir, "*.txt"))
    random.shuffle(files)
    
    print(f"  语料文件: {len(files)} 个")
    
    # 用于调试：记录每个词在所有文件中找到的总数
    total_found = {w: 0 for w in target_words}
    
    pbar = tqdm(files, desc="  扫描语料", unit="file")
    for fp in pbar:
        # 从文件名提取朝代信息
        filename = os.path.basename(fp)
        # 假设文件名格式：核心古籍-{朝代代码}.txt 或类似
        dynasty = "未知"
        for dyn_name, dyn_code in DYNASTY_MAP.items():
            if dyn_code in filename:
                dynasty = dyn_name
                break
        
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                content = f.read()
                sents = re.split(r'[。！？]', content)
                
                for s in sents:
                    s = s.strip()
                    if len(s) < 5 or len(s) > 500:
                        continue
                    
                    for w in target_words:
                        if w in s:
                            total_found[w] += 1  # 记录找到的总数
                            if len(word_sents[w]) < limit_per_word:
                                word_sents[w].append((dynasty, s))
        except Exception as e:
            # only print if not a common encoding error to avoid spam, or print first few
            # For now print all to be safe
            print(f"  ❌ Error reading {filename}: {e}")
    
    # 详细统计结果
    print(f"\n     语料库扫描统计:")
    found_counts = {w: len(sents) for w, sents in word_sents.items()}
    missing = [w for w, count in found_counts.items() if count == 0]
    
    # 找出那些在语料库中出现但因为已达上限而未收录的词
    if missing:
        print(f"     ⚠️  以下词未找到任何例句:")
        for w in missing[:10]:  # 最多显示10个
            print(f"        - {w}: 在语料库中共出现{total_found[w]}次 (长度≥5的句子)")
    
    # 找出收集不足200条的词
    insufficient = [(w, count) for w, count in found_counts.items() if 0 < count < limit_per_word]
    if insufficient:
        print(f"     ℹ️  以下词例句不足{limit_per_word}条:")
        for w, count in insufficient[:5]:
            print(f"        - {w}: {count}条 (语料库中共{total_found[w]}次)")
    
    return word_sents

def is_polysemous_word(word: str, dict_filepath: str) -> bool:
    """检查词汇是否为多义词"""
    try:
        with open(dict_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('---'): continue
                
                # Check for marker first to be fast
                if '多义' in line:
                    parts = line.split(',')
                    # 关键点：必须检查这一行的“主词”（逗号前第一个词）是不是我们要找的 word
                    # 否则，因为“义”字包含在“（多义）”标记里，如果不检查主词，
                    # 任何带有“（多义）”标记的行都会被误判为“义”字的多义行。
                    if len(parts) >= 1 and parts[0].strip() == word:
                        return True
    except Exception:
        pass
    return False

def load_filtered_sentences_for_polysemy(word: str, category_code: str, label: str, data_dir: str, limit: int = 200) -> List[Tuple[str, str]]:
    """读取多义词例句，返回 (朝代, 例句正文) 列表
    
    例句格式: [D宋] [05子藏] [0.07]十一年春公至自晋晋人以公为贰于楚故止公✓
    """
    # 尝试多个标签变体
    label_variants = [label]
    if label == 'neg':
        label_variants.extend(['nag', 'negative'])
    elif label == 'pos':
        label_variants.extend(['positive'])
    
    matches = []
    for label_variant in label_variants:
        pattern = f"{category_code}-{label_variant}-{word}-*.txt"
        search_path = os.path.join(data_dir, pattern)
        matches = glob.glob(search_path)
        if matches:
            break  # 找到就停止
    
    if not matches:
        return []
        
    filepath = matches[0]
    sentences = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # 提取所有方括号内容
                brackets = re.findall(r'\[([^\]]+)\]', line)
                if len(brackets) < 3:
                    continue
                
                # 第一个方括号：朝代标记 (如 "D宋")
                dynasty_tag = brackets[0]
                
                # 第三个方括号：相似度
                try:
                    similarity = float(brackets[2])
                except:
                    similarity = 0.0
                
                # 提取例句正文：去除前三个方括号及其内容
                content = re.sub(r'^(\[[^\]]+\]\s*){3}', '', line)
                content = content.replace('✓', '').replace('✗', '').strip()
                
                is_marked_positive = '✓' in line
                is_marked_negative = '✗' in line

                # 过滤逻辑
                if is_marked_negative:
                    continue
                if not is_marked_positive and similarity <= POLYSEMY_THRESHOLD:
                    continue
                
                if len(content) >= 5 and len(content) <= 500:
                    # 提取朝代名称（去掉前缀字母，如 "D宋" -> "宋"）
                    dynasty = dynasty_tag[1:] if len(dynasty_tag) > 1 else dynasty_tag
                    sentences.append((dynasty, content))
                        
    except Exception:
        pass
        
    if len(sentences) > limit:
        sentences = random.sample(sentences, limit)
        
    return sentences

def get_contextual_word_vector(word: str, sentences: List[Tuple[str, str]], tokenizer, model, device) -> np.ndarray:
    """获取上下文向量 (批量处理优化版)
    
    Args:
        word: 目标词汇
        sentences: [(朝代, 例句正文), ...] 元组列表
        tokenizer: 分词器
        model: BERT模型
        device: 设备
    """
    vectors = []
    # 提取例句正文部分
    target_sentences = [sent for _, sent in sentences[:200]]  # 取前200个例句
    
    for i in range(0, len(target_sentences), BATCH_SIZE):
        batch = target_sentences[i : i + BATCH_SIZE]
        if not batch: continue
        
        try:
            # 批量 Tokenize
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                # [FIX]: Use last hidden state to align with axis definition
                hs = outputs.last_hidden_state
            
            # 在 Batch 内部提取目标词向量
            for j, sent in enumerate(batch):
                # 注意：tokenize 可能会导致简单的 find 索引不对齐，这里做简化假设
                # SikuBERT 是 Character-based，所以 find 的索引 + 1 (CLS) 通常是准的
                start_idx = sent.find(word)
                if start_idx == -1: continue
                
                # 检查索引是否越界 (因为 truncation)
                input_len = (inputs['input_ids'][j] != tokenizer.pad_token_id).sum().item()
                
                token_start = start_idx + 1
                token_end = token_start + len(word)
                
                if token_end < input_len and token_end < 512:
                    word_vec = hs[j, token_start:token_end, :].mean(dim=0).cpu().numpy()
                    vectors.append(word_vec)

        except Exception as e:
            # print(f"Batch error: {e}")
            continue
    
    if not vectors:
        return None
    avg_vec = np.mean(vectors, axis=0)
    norm = np.linalg.norm(avg_vec)
    if norm > 0:
        avg_vec = avg_vec / norm
    return avg_vec

def load_sikubert(model_name_or_path: str, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载模型: {model_name_or_path} ({device})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)
    model.eval()
    model.to(device)
    model.eval()
    model.to(device)
    return tokenizer, model, device

def parse_dictionary(filepath: str, limit: int = 9999) -> Dict[str, List[str]]:
    """解析字典 (limit 默认为很大，即全量)"""
    pos_words = []
    neg_words = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            # 遇到分隔线停止读取
            if line.startswith('---'): break
            
            parts = line.split(',')
            if len(parts) < 2: continue
            word = parts[0].strip()
            tag = parts[1].strip().lower()
            if tag in ['pos', 'positive']:
                if len(pos_words) < limit: pos_words.append(word)
            elif tag in ['neg', 'negative', 'nag']:
                if len(neg_words) < limit: neg_words.append(word)
    return {'pos': pos_words, 'neg': neg_words}

def main():
    # 设置随机种子，确保结果可复现
    random.seed(42)
    np.random.seed(42)
    
    parser = argparse.ArgumentParser(description="静态 MVP 计算 (全量版)")
    parser.add_argument('--dic-dir', type=str, help='词典目录')
    parser.add_argument('--data-dir', type=str, help='ngram_match 数据目录')
    parser.add_argument('--corpus-dir', type=str, help='核心古籍语料目录')
    parser.add_argument('--model-path', type=str, help='模型路径')
    parser.add_argument('--axis-file', type=str, help='性别轴线文件')
    parser.add_argument('--output-dir', type=str, help='结果输出目录')
    parser.add_argument('--sample-size', type=int, default=DEFAULT_SAMPLE_SIZE, help='每个词抽样例句数量')
    
    # 环境检测
    # 优先检测 AutoDL (即使在 Jupyter 中运行也能正确识别)
    if os.path.exists("/root/autodl-tmp"):
        print("检测到 AutoDL 环境...")
        # 为了兼容 Notebook 和 命令行，简单判断
        if 'ipykernel' in sys.modules:
            args = parser.parse_args([])
        else:
            args = parser.parse_args()
            
        AUTODL_ROOT = Path("/root/autodl-tmp")
        
        # 1. 词典
        if args.dic_dir is None:
            possible_dic = AUTODL_ROOT / "dic"
            if possible_dic.exists(): args.dic_dir = str(possible_dic)
            else: args.dic_dir = "static/dic"
            
        # 2. 多义词数据
        if args.data_dir is None:
            cands = [AUTODL_ROOT / "ngram_match_sampled", AUTODL_ROOT / "moral_bias/data/ngram_match_sampled"]
            for p in cands:
                if p.exists():
                    args.data_dir = str(p)
                    print(f"  自动匹配 Data Dir: {p}")
                    break
        
        # 3. 核心古籍语料库
        if args.corpus_dir is None:
            cands = [
                AUTODL_ROOT / "核心古籍",
                AUTODL_ROOT / "data/核心古籍",
                AUTODL_ROOT / "moral_bias/data/核心古籍"
            ]
            for p in cands:
                if p.exists():
                    args.corpus_dir = str(p)
                    print(f"  自动匹配 Corpus Dir: {p}")
                    break
                    
        # 4. 静态轴线
        if args.axis_file is None:
            cands = [
                AUTODL_ROOT / "result/20251230-gender-axis/gender_axis.npy",
                AUTODL_ROOT / "moral_bias/result/20251230-gender-axis/gender_axis.npy"
            ]
            for p in cands:
                if p.exists():
                    args.axis_file = str(p)
                    print(f"  自动匹配 Axis File: {p}")
                    break
        
        # 5. 模型路径
        if args.model_path is None:
            cands = [
                AUTODL_ROOT / "model/sikubert",
                AUTODL_ROOT / "sikubert",
                AUTODL_ROOT / "moral_bias/model/sikubert"
            ]
            for p in cands:
                if p.exists():
                    args.model_path = str(p)
                    print(f"  自动匹配本地模型: {p}")
                    break
        
        # 6. 输出
        if args.output_dir is None:
            args.output_dir = str(AUTODL_ROOT / "result/20251230-moral-mvp-static")

    elif 'ipykernel' in sys.modules or 'google.colab' in sys.modules:
        print("检测到 Jupyter/Colab (非 AutoDL) 环境...")
        args = parser.parse_args([])
        if args.dic_dir is None and COLAB_DIC_DIR.exists(): args.dic_dir = str(COLAB_DIC_DIR)
        if args.data_dir is None and COLAB_NGRAM_MATCH_DIR.exists(): args.data_dir = str(COLAB_NGRAM_MATCH_DIR)
        if args.corpus_dir is None and COLAB_CORPUS_DIR.exists(): args.corpus_dir = str(COLAB_CORPUS_DIR)
        if args.axis_file is None and COLAB_AXIS_FILE.exists(): args.axis_file = str(COLAB_AXIS_FILE)
        if args.output_dir is None: args.output_dir = str(COLAB_OUTPUT_DIR)
        
    else:
        args = parser.parse_args()

        
    # 路径回退到默认
    dic_dir = args.dic_dir if args.dic_dir else DEFAULT_DIC_DIR
    data_dir = args.data_dir if args.data_dir else DEFAULT_NGRAM_MATCH_DIR
    corpus_dir = args.corpus_dir if args.corpus_dir else DEFAULT_CORPUS_DIR
    model_path = args.model_path if args.model_path else DEFAULT_MODEL_NAME
    axis_file = args.axis_file if args.axis_file else DEFAULT_AXIS_FILE
    output_dir = args.output_dir if args.output_dir else DEFAULT_OUTPUT_DIR
    sample_size = args.sample_size
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"数据配置:\n Dic: {dic_dir}\n Data: {data_dir}\n Corpus: {corpus_dir}\n Model: {model_path}\n Axis: {axis_file}\n Out: {output_dir}\n Sample: {sample_size}")

    print("=" * 60)
    print("🚀 开始静态 MVP 计算任务")
    print("=" * 60)

    # 1. 加载资源
    print(f"\n[Step 1/3] 正在加载基础资源...")
    if not os.path.exists(axis_file):
        print(f"❌ 错误: 找不到性别轴线文件 {axis_file}")
        return
    
    print(f"  -> 加载性别轴线: {axis_file}")
    gender_axis = np.load(axis_file)
    gender_axis = gender_axis / np.linalg.norm(gender_axis)
    
    print(f"  -> 加载 SikuBERT 模型: {model_path}")
    tokenizer, model, device = load_sikubert(model_path)
    print(f"  -> 模型加载完成，使用设备: {device}")
    
    # 2. 处理字典
    print(f"\n[Step 2/3] 开始处理道德词典...")
    dict_files = glob.glob(os.path.join(dic_dir, "*.txt"))
    results = []
    
    total_files = len([f for f in dict_files if not (f.endswith(".bak") or "dic_filter" in f)])
    print(f"  -> 找到 {total_files} 个词典文件待处理")
    
    file_idx = 0
    for filepath in dict_files:
        filename = os.path.basename(filepath)
        if filename.endswith(".bak") or filename == "dic_filter.xlsx": continue
        
        file_idx += 1
        category = CAT_MAP.get(filename, filename.replace(".txt", ""))
        
        print(f"\n  📝 处理文件 ({file_idx}/{total_files}): {filename} | 类别: {category}")
        
        # 全量读取
        words_dict = parse_dictionary(filepath, limit=9999)
        
        # 提取编号
        if filename[0].isdigit(): category_code = filename[:2]
        else: category_code = filename.split('.')[0][:2] if len(filename.split('.')[0]) >= 2 else "00"
        
        all_words = []
        for polarity, words in words_dict.items():
            for w in words:
                all_words.append((w, polarity))
                
        print(f"     共 {len(all_words)} 个词 (Pos: {len(words_dict['pos'])}, Neg: {len(words_dict['neg'])})")
        
        # 预先区分单义词和多义词
        poly_words = []
        mono_words = []
        
        print(f"     - 区分单/多义词...")
        for word, polarity in all_words:
            if is_polysemous_word(word, filepath):
                poly_words.append((word, polarity))
            else:
                mono_words.append((word, polarity))
        
        print(f"     - 单义词: {len(mono_words)} 个, 多义词: {len(poly_words)} 个")
        
        # 为单义词批量抽取语料
        sentences_map = {}
        if mono_words:
            print(f"     - 从语料库批量抽取单义词例句...")
            mono_word_list = [w for w, _ in mono_words]
            bulk_sents = extract_sentences_from_corpus(corpus_dir, mono_word_list, sample_size)
            
            # 朝代分层抽样并添加调试信息
            for w, sents in bulk_sents.items():
                before_count = len(sents)
                sampled_sents = stratified_sample_by_dynasty(sents, sample_size)
                after_count = len(sampled_sents)
                sentences_map[w] = sampled_sents
                
                # 如果抽样后变成0，打印警告
                if before_count > 0 and after_count == 0:
                    print(f"       ⚠️  {w}: 原有{before_count}条例句，分层抽样后变为0")
        
        # 为多义词逐个抽取筛选例句
        if poly_words:
            print(f"     - 加载多义词筛选例句...")
            poly_load_success = 0
            poly_load_fail = []
            
            for word, polarity in tqdm(poly_words, desc="       加载例句", unit="word"):
                sents = load_filtered_sentences_for_polysemy(word, category_code, polarity, data_dir, sample_size)
                
                if sents:
                    # 朝代分层抽样
                    sampled = stratified_sample_by_dynasty(sents, sample_size)
                    sentences_map[word] = sampled
                    poly_load_success += 1
                else:
                    sentences_map[word] = []
                    poly_load_fail.append(word)
            
            print(f"       多义词加载统计: 成功{poly_load_success}/{len(poly_words)}个")
            if poly_load_fail:
                print(f"       ⚠️  以下多义词未找到筛选例句: {poly_load_fail[:10]}")
        
        # 3. 统一计算所有词的上下文向量与投影
        print(f"     - 计算词向量...")
        
        # 临时存储该文件所有提取到的向量，用于去均值
        file_vectors = []  # list of (word, polarity, vector, is_poly)
        
        pbar = tqdm(all_words, desc="       提取向量", unit="word")
        for word, polarity in pbar:
            try:
                is_poly = word in [w for w, _ in poly_words]
                sents = sentences_map.get(word, [])
                
                if not sents:
                    pbar.write(f"       [跳过] {word}: 无有效例句")
                    continue
                
                # 统一使用上下文向量
                vec = get_contextual_word_vector(word, sents, tokenizer, model, device)
                    
                if vec is not None:
                    file_vectors.append({
                        "word": word,
                        "polarity": polarity,
                        "vector": vec,
                        "is_poly": is_poly
                    })
            except Exception as e:
                print(f"Error processing {word}: {e}")
                continue
        
        # 4. 去均值与投影计算
        if file_vectors:
            # 提取该文件（类别）下所有词的向量进行去均值
            # 注意：理论上应该用全语料均值，但在MVP分析中，
            # 使用当前类别（或所有已处理类别）的均值来校准“该领域的背景”也是常用做法。
            # 这里为了操作简便且有效，我们在文件级别（Category级别）做一次校准，
            # 或者，更严谨的做法是：收集 全量 向量后再做。
            # 
            # 考虑到脚本是分文件处理的，为了避免巨大的内存开销（虽然只有几千个向量其实还好），
            # 且我们希望 Category 内部的相对位置准确。
            # 但用户希望的是“全局”的一致性。
            # 
            # 修正决策：由于我们在外层循环是逐个文件处理并 append 到 results，
            # 这里只收集了当前文件的向量。如果只减去当前文件的均值，
            # 可能会导致不同 Category 的“零点”不一样（例如“权威”类的整体偏移和“关爱”类不一样）。
            # 这对于跨类别比较（Boxplot）是危险的。
            # 
            # 纠正方案：不做局部去均值。
            # 改为：将所有文件的结果包含向量先存起来 (results_with_vectors)，
            # 循环结束后统一去均值，再计算 score。
            
            # --- 暂时先存入 results 列表，包含 vector ---
            for item in file_vectors:
                results.append({
                    "word": item['word'], 
                    "category": category, 
                    "polarity": item['polarity'],
                    "vector": item['vector'], # 暂存向量
                    "is_polysemous": item['is_poly']
                })
    
    # === 所有文件处理完毕，统一去均值并计算分数 ===
    print(f"\n[Step 3/3] 全局去均值与分数计算...")
    
    if results:
        # stack vectors
        all_vecs = np.stack([r['vector'] for r in results])
        global_mean = np.mean(all_vecs, axis=0)
        
        print(f"  ⚡ 已应用去均值 (Common Mean Removal) 策略")
        print(f"     全局向量数: {len(results)}")
        print(f"     向量维度: {all_vecs.shape}")

        # 计算并更新
        valid_count = 0
        for res in results:
            vec_centered = res['vector'] - global_mean
            # 再次归一化? 通常去均值后方向改变，长度也变。
            # 在 analyze_category_relationships_split.py 中，我们去均值后做了归一化。
            # 为了保持一致性：
            norm = np.linalg.norm(vec_centered)
            if norm > 0:
                vec_centered = vec_centered / norm
            
            score = float(np.dot(vec_centered, gender_axis))
            res['score'] = score
            
            # 移除 vector 字段以减小 JSON 体积
            del res['vector'] 
            valid_count += 1
            
        print(f"  ✅ 计算完成: {valid_count} 个词")

    # 4. 输出保存 (原 Step 3)
    print(f"  正在保存结果...")
    results.sort(key=lambda x: x['score'], reverse=True)
    json_path = os.path.join(output_dir, "moral_bias_mvp_static_full.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"保存 JSON: {json_path}")

    md_path = os.path.join(output_dir, "moral_bias_mvp_static_full.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 全量静态 MVP 分析\n\n")
        f.write(f"| 排名 | 词汇 | 类别 | 极性 | 分数 |\n|---|---|---|---|---|\n")
        for i, res in enumerate(results):
            f.write(f"| {i+1} | {res['word']} | {res['category']} | {res['polarity']} | {res['score']:.4f} |\n")
    print(f"保存 Markdown: {md_path}")

if __name__ == "__main__":
    main()
