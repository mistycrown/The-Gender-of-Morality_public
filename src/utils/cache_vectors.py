"""
================================================================================
全局向量缓存生成脚本 (Global Vector Cache Generator)
================================================================================
每次更新代码文件时请更新代码头注释！

【功能描述】
为“静态目标词置换检验”生成所需的词向量缓存。
不分朝代，直接从整个语料库中为每个词抽取例句，计算“全局静态向量”。
同时包含中性“对照词”(Control Words) 的向量，用于构建统计检验的背景池。

主要步骤：
1. 读取性别词、道德词和内置的对照词表
2. 扫描语料库，为每个词抽取 200 个句子
3. 使用 SikuBERT 提取上下文向量并计算均值
4. 保存结果为 Python Pickle 格式

【输入】
- static/woman-man/basic.txt - 性别词文件
- static/dic/*.txt - 道德词典文件
- data/核心古籍/*.txt - 核心语料库

【输出】
- result/global_vector_cache.pkl - 全局向量缓存文件

【参数说明】
- --dic-dir: 道德词典目录
- --gender-file: 性别词文件路径
- --corpus-dir: 语料库目录
- --model-path: SikuBERT 模型路径或名称
- --output: 输出文件路径

【依赖库】
- numpy, torch, transformers
- tqdm, pickle

【作者】AI辅助生成
【创建日期】2026-01-16
【最后修改】2026-01-16 - 初始开源版本整理

【修改历史】
- 2026-01-16: 整理为开源版本，规范代码头
================================================================================
"""

import os
import sys
import glob
import re
import random
import pickle
import argparse
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Optional

# [CRITICAL FIX] 设置 Hugging Face 镜像，解决连接超时问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ==================== 配置 ====================
DEFAULT_DIC_DIR = "static/dic"
DEFAULT_GENDER_FILE = r"static\woman-man\basic.txt"
DEFAULT_CORPUS_DIR = r"data\核心古籍"
DEFAULT_MODEL_NAME = "SIKU-BERT/sikubert"
OUTPUT_FILE = r"result\global_vector_cache.pkl"

SAMPLE_SIZE = 200  # 每个词抽样数 (全量语料库)
BATCH_SIZE = 128

# 对照词表 (Control Words): 用于 Target Permutation Test 的背景池
# 选取了一些古代汉语中常见且在此语境下相对中性的词汇 (自然、物品、方位、动作等)
CONTROL_WORDS = [
    "山", "水", "云", "雨", "风", "雪", "月", "日", "星", "辰",
    "天", "地", "江", "河", "湖", "海", "石", "木", "花", "草",
    "春", "夏", "秋", "冬", "东", "西", "南", "北", "中", "外",
    "门", "户", "窗", "路", "道", "桥", "车", "舟", "书", "剑",
    "酒", "茶", "食", "衣", "冠", "履", "琴", "棋", "画", "诗",
    "行", "走", "坐", "卧", "看", "听", "言", "语", "思", "想",
    "手", "足", "身", "心", "耳", "目", "口", "鼻", "声", "色",
    "红", "黄", "蓝", "白", "黑", "青", "绿", "紫", "金", "银",
    "城", "郭", "乡", "野", "朝", "市", "宫", "殿", "楼", "台",
    "鸟", "兽", "鱼", "虫", "龙", "虎", "马", "牛", "羊", "犬"
]

# ==================== 工具函数 ====================

def load_sikubert(model_name_or_path: str, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载模型: {model_name_or_path} ({device})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)
    model.eval()
    model.to(device)
    return tokenizer, model, device

def load_gender_words(filepath: str) -> List[str]:
    words = set()
    if not os.path.exists(filepath): return []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            for p in parts:
                if p: words.add(p)
    return list(words)

def load_moral_words(dic_dir: str) -> List[str]:
    words = set()
    files = glob.glob(os.path.join(dic_dir, "*.txt"))
    for fp in files:
        fname = os.path.basename(fp)
        if "filter" in fname or fname.endswith(".bak"): continue
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("---"): break
                parts = line.split(',')
                if parts: words.add(parts[0].strip())
    return list(words)

def extract_sentences_global(corpus_dir: str, words: List[str], limit: int) -> Dict[str, List[str]]:
    """扫描语料库，为每个词抽取句子 (不分朝代混合池)"""
    res = {w: [] for w in words}
    files = glob.glob(os.path.join(corpus_dir, "*.txt"))
    random.shuffle(files) # 随机打乱文件顺序，避免只抽到某个朝代的
    
    print(f"扫描 {len(files)} 个语料文件...")
    
    # 为了效率，我们不能等到所有词都满了才停，那样太慢。
    # 我们扫描文件，直到大部分词都满了，或者扫描完一定数量的文件。
    # 这里采用“遍历所有文件，但每个词满则止”的策略。
    
    # 优化：构建 set 用于快速检查是否还需要收集
    needed_words = set(words)
    
    pbar = tqdm(files, desc="Parsing Corpus", unit="file")
    for fp in pbar:
        if not needed_words: break
        
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                content = f.read()
                # 简单分句
                sents = re.split(r'[。！？]', content)
                
                for s in sents:
                    s = s.strip()
                    if len(s) < 5 or len(s) > 500: continue
                    
                    # 检查此句包含哪些需要的词
                    # 优化：只检查 needed_words
                    found_in_sent = []
                    for w in needed_words:
                        if w in s:
                            res[w].append(s)
                            found_in_sent.append(w)
                    
                    # 检查是否已满
                    for w in found_in_sent:
                        if len(res[w]) >= limit:
                            needed_words.remove(w)
                            
        except Exception:
            pass
            
        pbar.set_postfix({"Remaining Words": len(needed_words)})
        
    return res

def get_contextual_vector(tokenizer, model, device, word: str, sentences: List[str]) -> Optional[np.ndarray]:
    """计算单个词的平均向量"""
    if not sentences: return None
    
    vecs = []
    # Batch processing
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i:i+BATCH_SIZE]
        if not batch: continue
        
        try:
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hs = outputs.last_hidden_state
            
            for j, sent in enumerate(batch):
                s_idx = sent.find(word)
                if s_idx == -1: continue
                
                # Check truncation
                input_len = (inputs['input_ids'][j] != tokenizer.pad_token_id).sum().item()
                token_start = s_idx + 1
                token_end = token_start + len(word)
                
                if token_end < input_len and token_end < 512:
                    v = hs[j, token_start:token_end, :].mean(dim=0).cpu().numpy()
                    vecs.append(v)
        except:
            continue
            
    if not vecs: return None
    avg = np.mean(vecs, axis=0)
    norm = np.linalg.norm(avg)
    return avg / norm if norm > 0 else avg

# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser()
    # 路径参数
    parser.add_argument('--dic-dir', default=DEFAULT_DIC_DIR)
    parser.add_argument('--gender-file', default=DEFAULT_GENDER_FILE)
    parser.add_argument('--corpus-dir', default=DEFAULT_CORPUS_DIR)
    parser.add_argument('--model-path', default=DEFAULT_MODEL_NAME)
    parser.add_argument('--output', default=OUTPUT_FILE)
    
    # 兼容
    import sys
    if 'ipykernel' in sys.modules: args = parser.parse_args([])
    else: args = parser.parse_args()
    
    # AutoDL 路径修正
    if os.path.exists("/root/autodl-tmp"):
        root = "/root/autodl-tmp"
        if args.dic_dir == DEFAULT_DIC_DIR: args.dic_dir = os.path.join(root, "dic")
        if args.gender_file == DEFAULT_GENDER_FILE: args.gender_file = os.path.join(root, "static/woman-man/basic.txt")
        if args.corpus_dir == DEFAULT_CORPUS_DIR: args.corpus_dir = os.path.join(root, "核心古籍")
        if args.output == OUTPUT_FILE: args.output = os.path.join(root, "result/global_vector_cache.pkl")
        
        # [CRITICAL] 优先使用本地模型路径，避免联网
        p1 = os.path.join(root, "model/sikubert")
        p2 = os.path.join(root, "sikubert")
        if os.path.exists(p1): 
            args.model_path = p1
            print(f"Found local model at: {p1}")
        elif os.path.exists(p2): 
            args.model_path = p2
            print(f"Found local model at: {p2}")
        else:
            # 只有找不到本地模型时才设置镜像
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            print("Local model not found, using HF mirror.")

    print("="*60)
    print("全局向量缓存生成 (Global Vector Cache)")
    print("="*60)
    
    # 1. 准备词表
    gender_words = load_gender_words(args.gender_file)
    moral_words = load_moral_words(args.dic_dir)
    control_words = CONTROL_WORDS
    
    all_words = list(set(gender_words + moral_words + control_words))
    print(f"目标词汇统计:")
    print(f"  - 性别词: {len(gender_words)}")
    print(f"  - 道德词: {len(moral_words)}")
    print(f"  - 对照词: {len(control_words)}")
    print(f"  - 总计: {len(all_words)}")
    
    # 2. 抽取语料
    print("\n正在从全语料库中抽取例句 (Limit=200)...")
    sent_map = extract_sentences_global(args.corpus_dir, all_words, SAMPLE_SIZE)
    
    # 3. 计算向量
    print("\n正在计算向量...")
    tokenizer, model, device = load_sikubert(args.model_path)
    
    cache = {"Global": {}}
    valid_count = 0
    
    for w in tqdm(all_words, desc="Encoding"):
        sents = sent_map.get(w, [])
        if not sents: continue
        
        vec = get_contextual_vector(tokenizer, model, device, w, sents)
        if vec is not None:
            cache["Global"][w] = vec
            valid_count += 1
            
    print(f"\n计算完成: 有效向量 {valid_count}/{len(all_words)}")
    
    # 4. 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(cache, f)
    print(f"缓存已保存至: {args.output}")
    print("请使用 verify_gender_axis_significance.py 并指定 --cache 路径来运行检验。")

if __name__ == "__main__":
    main()
