"""
================================================================================
道德词典性别偏见 MVP计算脚本 (历时版 - 全量实验)
================================================================================
每次更新代码文件时请更新代码头注释！

【功能描述】
1. 读取 static/dic/ 下的道德词典文件（全量读取）。
2. 在各朝代语料库中抽取句子：
   - 单义词：从语料库随机抽样
   - 多义词：从 data/ngram_match 读取筛选后的例句
3. 使用 SikuBERT 不同提取历时动态词向量。
4. 计算每个词在对应朝代性别轴线上的投影得分。
5. 输出历时变化 JSON 数据和报告。

【输入】
- 词典: static/dic/*.txt
- 多义词例句: data/ngram_match/*.txt
- 性别轴线: result/20251230-diachronic-axis/dynasty_gender_axes.npy
- 语料库: data/核心古籍/核心古籍-*.txt

【输出】
- result/20251230-moral-mvp-diachronic/moral_bias_mvp_diachronic_full.json
- result/20251230-moral-mvp-diachronic/moral_bias_mvp_diachronic_full_report.md

【参数说明】
- --dic-dir: 词典目录
- --data-dir: ngram_match 目录
- --corpus-dir: 核心古籍语料目录
- --axis-file: 历时性别轴线文件
- --output-dir: 结果输出目录
- --sample-size: 每个词每朝代抽样句子数 (默认 50)

【依赖库】
- transformers, torch, numpy, tqdm

【作者】AI辅助生成
【创建日期】2026-01-11
【最后修改】- 2024-12-30: 创建初始版本，支持简单的同义筛选
- 2025-01-11: 优化 & 迁移 AutoDL
    1. 增加 Batch Processing (128) 优化 3090 性能
    2. 增加 AutoDL 环境自动识别 (/root/autodl-tmp)
    3. 增加本地模型自动检测 (优先于 HF 下载)
    4. 修复词典读取逻辑（忽略分隔线后内容）
    5. 增加启动提示，避免加载时假死体验
    6. 修复 IndentationError 和变量初始化问题

【修改历史】
- 2026-01-11: 基于 calculate_moral_mvp_diachronic.py 创建全量版，增加 Colab 支持
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
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# ==================== 默认配置 ====================
DEFAULT_MODEL_NAME = "SIKU-BERT/sikubert"
DEFAULT_DIC_DIR = "static/dic"
DEFAULT_NGRAM_MATCH_DIR = r"data\ngram_match_sampled"
DEFAULT_CORPUS_DIR = r"data\核心古籍"
DEFAULT_AXIS_FILE = r"result\20251230-diachronic-axis\dynasty_gender_axes.npy"
DEFAULT_OUTPUT_DIR = r"result\20251230-moral-mvp-diachronic"
DEFAULT_SAMPLE_SIZE = 200
BATCH_SIZE = 128  # 针对 3090 优化
POLYSEMY_THRESHOLD = 0.05  # 多义词相似度截断阈值

# colab 默认路径
COLAB_ROOT = Path("/content/drive/MyDrive/moral_bias_data")
COLAB_DIC_DIR = COLAB_ROOT / "dic"
COLAB_NGRAM_MATCH_DIR = COLAB_ROOT / "ngram_match"
COLAB_CORPUS_DIR = COLAB_ROOT / "corpus"
COLAB_AXIS_FILE = COLAB_ROOT / "gender_axis/dynasty_gender_axes.npy"
COLAB_OUTPUT_DIR = COLAB_ROOT / "result/20251230-moral-mvp-diachronic"

# 朝代映射
DYNASTY_MAP = {
    '先秦两汉': 'A先秦两汉', '魏晋南北朝': 'B魏晋南北朝', '隋唐': 'C隋唐',
    '宋': 'D宋', '元': 'E元', '明': 'F明', '清': 'G清'
}
DYNASTY_ORDER = ['先秦两汉', '魏晋南北朝', '隋唐', '宋', '元', '明', '清']

# 词典文件名到类别的映射
CAT_MAP = {
    "01care.txt": "Care", "02aut.txt": "Authority", "03loy.txt": "Loyalty",
    "04fair.txt": "Fairness", "05san.txt": "Sanctity", "06lenience.txt": "Lenience",
    "07waste.txt": "Waste", "08altruism.txt": "Altruism", "09diligence.txt": "Diligence",
    "10resilience.txt": "Resilience", "11modesty.txt": "Modesty", "12valor.txt": "Valor"
}

def is_polysemous_word(word: str, dict_filepath: str) -> bool:
    """检查词汇是否为多义词"""
    try:
        with open(dict_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('---'): continue
                
                if '多义' in line:
                    parts = line.split(',')
                    if len(parts) >= 1 and parts[0].strip() == word:
                        return True
    except Exception:
        pass
    return False

def load_filtered_sentences_for_polysemy(word, category_code, label, data_dir, dynasty_code=None, limit=200):
    """读取多义词筛选例句，返回 (朝代, 例句正文) 列表"""
    # 尝试多个标签变体
    label_variants = [label]
    if label == 'neg':
        label_variants.extend(['nag', 'negative'])
    elif label == 'pos':
        label_variants.extend(['positive'])
    
    matches = []
    for label_variant in label_variants:
        pattern = f"{category_code}-{label_variant}-{word}-*.txt"
        matches = glob.glob(os.path.join(data_dir, pattern))
        if matches:
            break
    
    if not matches: return []
    
    filepath = matches[0]
    sentences = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                # 提取所有方括号内容
                brackets = re.findall(r'\[([^\]]+)\]', line)
                if len(brackets) < 3:
                    continue
                
                # 第一个方括号：朝代标记 (如 "D宋")
                dynasty_tag = brackets[0]
                
                # 朝代筛选
                if dynasty_code and not dynasty_tag.startswith(dynasty_code):
                    continue
                
                # 第三个方括号：相似度
                try:
                    similarity = float(brackets[2])
                except:
                    similarity = 0.0
                
                # 提取例句正文：去除前三个方括号及其内容
                content = re.sub(r'^(\[[^\]]+\]\s*){3}', '', line)
                content = content.replace('✓', '').replace('✗', '').strip()
                
                is_pos = '✓' in line
                is_neg = '✗' in line
                
                if is_neg: continue
                if not is_pos and similarity <= POLYSEMY_THRESHOLD: continue
                
                if 5 <= len(content) <= 500:
                    # 提取朝代名称（去掉前缀字母）
                    dynasty = dynasty_tag[1:] if len(dynasty_tag) > 1 else dynasty_tag
                    sentences.append((dynasty, content))
    except: pass
    
    if len(sentences) > limit:
        sentences = random.sample(sentences, limit)
    return sentences

def extract_sentences_from_corpus(corpus_dir, regex_pattern, target_words, limit_per_word):
    """从语料库中提取例句，返回 {词汇: [(朝代, 例句), ...]} 格式
    
    Args:
        regex_pattern: 朝代代码，用于匹配文件名（如 "D宋"）
    """
    word_sents = {w: [] for w in target_words}
    files = glob.glob(os.path.join(corpus_dir, f"*{regex_pattern}*.txt"))
    random.shuffle(files)
    
    print(f"  语料文件: {len(files)} (模式: {regex_pattern})")
    
    # 从regex_pattern提取朝代名称（如 "D宋" -> "宋"）
    dynasty_name = regex_pattern[1:] if len(regex_pattern) > 1 and regex_pattern[0].isalpha() else regex_pattern
    
    # 增加文件级进度条
    pbar = tqdm(files, desc="  扫描语料", unit="file")
    for fp in pbar:
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                content = f.read()
                sents = re.split(r'[。！？]', content)
                for s in sents:
                    s = s.strip()
                    if len(s) < 5 or len(s) > 500: continue
                    for w in target_words:
                        if len(word_sents[w]) >= limit_per_word: continue
                        if w in s:
                            word_sents[w].append((dynasty_name, s))  # 返回元组格式
        except Exception as e:
            print(f"  ❌ Error reading {os.path.basename(fp)}: {e}")
    
    # 统计结果
    missing = [w for w, sents in word_sents.items() if len(sents) == 0]
    if missing:
        print(f"  ⚠️  未找到例句的词: {missing[:3]}{'...' if len(missing) > 3 else ''}")
    
    return word_sents

def get_contextual_vector(model, tokenizer, device, sentences: List[Tuple[str, str]], target_word: str) -> Optional[np.ndarray]:
    """获取上下文向量
    
    Args:
        sentences: [(朝代, 例句正文), ...] 元组列表
        target_word: 目标词汇
    """
    vectors = []
    # 提取例句正文部分
    target_sentences = [sent for _, sent in sentences]
    
    for i in range(0, len(target_sentences), BATCH_SIZE):
        batch = target_sentences[i : i + BATCH_SIZE]
        try:
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(**encoded, output_hidden_states=True)
                # [FIX]: Axis was calculated using only the last layer.
                # To align with the axis space, we must use the last hidden state directly.
                hs = outputs.last_hidden_state
            
            for j, s in enumerate(batch):
                start = s.find(target_word)
                if start == -1: continue
                # 简单映射：假设字粒度，[CLS]偏移1
                valid_len = min(len(target_word), 510)
                if start + 1 + valid_len >= 512: continue
                vec = hs[j, start+1:start+1+valid_len, :].mean(dim=0).cpu().numpy()
                if np.isnan(vec).any():
                    # Skip NaN vectors to prevent polluting the average
                    continue
                vectors.append(vec)
        except Exception as e:
            # print(f"Error: {e}")
            pass
        
    if not vectors: return None
    avg = np.mean(vectors, axis=0)
    norm = np.linalg.norm(avg)
    if norm > 0: avg = avg / norm
    return avg

def load_sikubert(path, device=None):
    if not device: device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载模型: {path} ({device})...")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path)
    model.eval()
    model.to(device)
    return tokenizer, model, device

def parse_dictionary(filepath, limit=9999):
    words = []
    filename = os.path.basename(filepath)
    category = CAT_MAP.get(filename, filename.replace(".txt", ""))
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            # 遇到分隔线停止读取
            if line.startswith('---'): break
            
            parts = line.split(',')
            if len(parts) < 2: continue
            w = parts[0].strip()
            tag = parts[1].strip().lower()
            pol = ""
            if tag in ['pos', 'positive']: pol = "pos"
            elif tag in ['neg', 'negative', 'nag']: pol = "neg"
            
            if pol:
                words.append({"word": w, "category": category, "polarity": pol})
    return words

def main():
    # 设置随机种子，确保结果可复现
    random.seed(42)
    np.random.seed(42)
    
    parser = argparse.ArgumentParser(description="历时 MVP 计算 (全量版)")
    parser.add_argument('--dic-dir', type=str, help='词典目录')
    parser.add_argument('--data-dir', type=str, help='多义词数据目录')
    parser.add_argument('--corpus-dir', type=str, help='语料库目录')
    parser.add_argument('--axis-file', type=str, help='性别轴线文件')
    parser.add_argument('--output-dir', type=str, help='结果目录')
    parser.add_argument('--model-path', type=str, help='模型路径')
    parser.add_argument('--sample-size', type=int, default=DEFAULT_SAMPLE_SIZE, help='抽样数量')
    
    # 环境检测
    # 优先检测 AutoDL (即使在 Jupyter 中运行也能正确识别)
    if os.path.exists("/root/autodl-tmp"):
        print("检测到 AutoDL 环境...")
        args = parser.parse_args([]) # Parse args for AutoDL environment
        # 定义 AutoDL 常用路径
        AUTODL_ROOT = Path("/root/autodl-tmp")
        
        # 1. 词典 (假设在项目内)
        if args.dic_dir is None:
            # 尝试找 /root/autodl-tmp/moral_bias/static/dic
            possible_dic = AUTODL_ROOT / "dic"
            if possible_dic.exists(): args.dic_dir = str(possible_dic)
            else: args.dic_dir = "static/dic" # 默认相对路径
            
        # 2. 多义词数据 (ngram_match_sampled)
        if args.data_dir is None:
            cands = [
                AUTODL_ROOT / "ngram_match_sampled",
                AUTODL_ROOT / "moral_bias/data/ngram_match_sampled",
                Path("/root/ngram_match_sampled")
            ]
            for p in cands:
                if p.exists():
                    args.data_dir = str(p)
                    print(f"  自动匹配 Data Dir: {p}")
                    break
        
        # 3. 核心古籍
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
                    
        if args.axis_file is None:
            cands = [
                # 根据截图调整：直接在 autodl-tmp 下
                AUTODL_ROOT / "20251230-diachronic-axis/dynasty_gender_axes.npy",
                AUTODL_ROOT / "result/20251230-diachronic-axis/dynasty_gender_axes.npy",
                AUTODL_ROOT / "moral_bias/result/20251230-diachronic-axis/dynasty_gender_axes.npy"
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
            args.output_dir = str(AUTODL_ROOT / "result/20251230-moral-mvp-diachronic")

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

    # 参数回退
    dic_dir = args.dic_dir if args.dic_dir else DEFAULT_DIC_DIR
    data_dir = args.data_dir if args.data_dir else DEFAULT_NGRAM_MATCH_DIR
    corpus_dir = args.corpus_dir if args.corpus_dir else DEFAULT_CORPUS_DIR
    model_path = args.model_path if args.model_path else DEFAULT_MODEL_NAME
    axis_file = args.axis_file if args.axis_file else DEFAULT_AXIS_FILE
    output_dir = args.output_dir if args.output_dir else DEFAULT_OUTPUT_DIR
    sample_size = args.sample_size
    
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("🚀 开始历时 MVP 计算任务 (Diachronic)")
    print("=" * 60)
    print(f"配置:\n Dic: {dic_dir}\n Data: {data_dir}\n Corpus: {corpus_dir}\n Axis: {axis_file}")
    
    # 1. 加载
    print(f"\n[Step 1/5] 加载模型与资源...")
    tokenizer, model, device = load_sikubert(model_path)
    if not os.path.exists(axis_file):
        print(f"❌ 错误: 未找到性别轴线文件 {axis_file}")
        return
    axes_data = np.load(axis_file, allow_pickle=True).item()
    print(f"  -> 性别轴线加载完成，包含朝代: {list(axes_data.keys())}")
    
    # 2. 收集词
    print(f"\n[Step 2/5] 预扫描词典...")
    all_target_info = []
    dict_files = glob.glob(os.path.join(dic_dir, "*.txt"))
    for df in dict_files:
        if "dic_filter" in df: continue
        all_target_info.extend(parse_dictionary(df, limit=9999))
    
    target_words = list(set([x['word'] for x in all_target_info]))
    word_info_map = {x['word']: x for x in all_target_info}
    print(f"  -> 提取目标词: {len(target_words)} 个")
    
    chronological_scores = {}
    
    # 3. 历时计算
    # 预先加载所有词典内容以加速多义词判断
    print(f"  -> 缓存词典内容以加速查找...")
    dict_content_cache = {}
    for df in dict_files:
        dict_content_cache[df] = open(df, encoding='utf-8').read()
        
    dynasty_idx = 0
    total_dynasties = len([d for d in DYNASTY_ORDER if DYNASTY_MAP.get(d)])

    for dynasty in DYNASTY_ORDER:
        d_key = DYNASTY_MAP.get(dynasty)
        if not d_key: continue
        
        dynasty_idx += 1
        print(f"\n" + "-" * 50)
        print(f"[Step 3/5] 处理朝代 ({dynasty_idx}/{total_dynasties}): {dynasty} ({d_key})")
        print("-" * 50)
        
        axis = axes_data.get(dynasty)
        axis = axis / np.linalg.norm(axis)
        
        # 准备数据
        sentences_map = {}
        # 预先判断哪些是多义词
        poly_words = []
        mono_words = []
        poly_load_success = 0
        
        print(f"  -> 3.1 区分单/多义词并加载多义词例句...")
        # 使用 progress bar
        for w in tqdm(target_words, desc="    词汇分类", unit="word"):
            # 简单判断，这里需要遍历所有字典文件确认是否poly，略慢
            # 优化：在parse_dictionary时就记录是否poly
            # 这里复用原有逻辑：实时查找
            is_poly = False
            found_df = None
            category_code = "00"
            polarity = word_info_map[w]['polarity']
            
            for df in dict_files:
                if w in dict_content_cache[df]: # 使用缓存加速
                     if is_polysemous_word(w, df):
                        is_poly = True
                        found_df = df
                        fname = os.path.basename(df)
                        category_code = fname[:2] if fname[0].isdigit() else "00"
                        break
            
            if is_poly:
                poly_words.append(w)
                sents = load_filtered_sentences_for_polysemy(w, category_code, polarity, data_dir, d_key, sample_size)
                sentences_map[w] = sents
                if sents:
                    poly_load_success += 1
            else:
                mono_words.append(w)
                sentences_map[w] = []
        
        # 显示多义词统计
        if poly_words:
            print(f"     多义词: {len(poly_words)}个, 成功加载: {poly_load_success}/{len(poly_words)}")
        
        # 批量采集单义词
        if mono_words:
            print(f"  -> 3.2 语料库扫描 ({len(mono_words)} 个单义词)...")
            bulk = extract_sentences_from_corpus(corpus_dir, d_key, mono_words, sample_size)
            mono_success = sum(1 for w in mono_words if bulk.get(w))
            for w, s in bulk.items():
                sentences_map[w] = s
            print(f"     单义词统计: {len(mono_words)}个, 成功采集: {mono_success}/{len(mono_words)}")
            print(f"     采集完成")
        
        # 临时存储该朝代的所有向量
        # list of (word, vector)
        epoch_vectors = []
        
        # 计算
        print(f"  -> 3.3 计算词向量...")
        pbar = tqdm(target_words, desc="    提取向量", unit="word")
        for w in pbar:
            sents = sentences_map.get(w, [])
            if not sents:
                # 详细报错逻辑 (省略以简化输出，保留核心提示)
                # print(f"\033[91m  ❌ CRITICAL ERROR: 词汇 '{w}' 在朝代 '{dynasty}' ({d_key}) 未找到任何例句！\033[0m")
                continue 
            
            vec = get_contextual_vector(model, tokenizer, device, sents, w)
            if vec is not None:
                # 检查 NaN
                if np.isnan(vec).any():
                    pbar.write(f"     [警告] {w} 向量包含 NaN，跳过")
                    continue
                    
                epoch_vectors.append((w, vec))
            else:
                 # pbar.write(f"\033[91m  ❌ CRITICAL ERROR: 词汇 '{w}' 在朝代 '{dynasty}' 计算向量失败（返回 None）！\033[0m")
                 pass

        # 统一去均值并计算这个朝代的分数
        if epoch_vectors:
            # 1. 计算当前朝代的全局均值
            all_vecs = np.stack([v for _, v in epoch_vectors])
            epoch_mean = np.mean(all_vecs, axis=0)
            
            print(f"  ⚡ [朝代: {dynasty}] 已应用去均值 (Common Mean Removal)")
            
            # 2. 去均值并投影
            for w, vec in epoch_vectors:
                # 去均值
                vec_centered = vec - epoch_mean
                
                # 归一化
                norm = np.linalg.norm(vec_centered)
                if norm > 0:
                    vec_centered = vec_centered / norm
                
                # 投影
                score = float(np.dot(vec_centered, axis))
                
                if w not in chronological_scores: chronological_scores[w] = {}
                chronological_scores[w][dynasty] = score
        else:
            print(f"  ⚠️  朝代 {dynasty} 未提取到任何有效向量")
                
    # 4. 输出
    print(f"\n[Step 4/5] 整合并输出结果...")
    output_data = []
    for w, scores in chronological_scores.items():
        info = word_info_map.get(w, {})
        output_data.append({
            "word": w, "category": info.get("category"), "polarity": info.get("polarity"),
            "scores": scores
        })
        
    json_path = os.path.join(output_dir, "moral_bias_mvp_diachronic_full.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        
    print(f"✅ [Done] JSON 已保存: {json_path}") 
    print(f"🎉 全部任务完成！")

if __name__ == "__main__":
    main()
