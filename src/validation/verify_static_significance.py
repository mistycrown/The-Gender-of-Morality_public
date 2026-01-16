"""
================================================================================
静态目标词置换检验脚本 (Static Target Permutation Test)
================================================================================
每次更新代码文件时请更新代码头注释！

【功能描述】
对静态的道德词汇进行置换检验（Permutation Test），验证其性别偏见是否显著。
测试“特定的道德词组”是否比“随机抽取的词组”在性别轴上偏离得更远。

主要步骤：
1. 加载全局词向量缓存
2. 构建静态性别轴 (Male - Female)
3. 构建背景池 (Background Pool)
4. 对每个道德类别及其子类 (All, Pos, Neg) 进行 N 次随机置换
5. 计算 P 值并输出显著性结果

【输入】
- result/global_vector_cache.pkl - 全局词向量缓存
- static/woman-man/basic.txt - 性别种子词
- static/dic/*.txt - 道德词典文件

【输出】
- result/permutation_test_target/target_permutation_results.csv - 检验结果表格

【参数说明】
- --cache: 向量缓存文件路径
- --permutations: 置换次数 (默认 1000)
- --output: 结果输出目录

【依赖库】
- numpy, pandas, scipy
- argparse, pickle

【作者】AI辅助生成
【创建日期】2026-01-16
【最后修改】2026-01-16 - 初始开源版本整理

【修改历史】
- 2026-01-16: 整理为开源版本，规范代码头
================================================================================
"""

import os
import sys
import pickle
import random
import argparse
import numpy as np
import glob
from typing import Dict, List
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==================== 配置 ====================
DEFAULT_CACHE_FILE = r"result/global_vector_cache.pkl"  # [UPDATE] 默认使用全局缓存
DEFAULT_GENDER_FILE = r"static/woman-man/basic.txt"
DEFAULT_DIC_DIR = r"static/dic"
OUTPUT_DIR = r"result/permutation_test_target"
N_PERMUTATIONS = 1000  # 抽样次数

def get_axis_vector(male_vecs: List[np.ndarray], female_vecs: List[np.ndarray]) -> np.ndarray:
    """计算固定的性别轴线"""
    if not male_vecs or not female_vecs:
        return None
    mean_m = np.mean(male_vecs, axis=0)
    mean_f = np.mean(female_vecs, axis=0)
    axis = mean_m - mean_f
    return axis

def load_gender_pairs(filepath):
    male_words, female_words = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                male_words.append(parts[0])
                female_words.append(parts[1])
    return male_words, female_words

def load_moral_categories(dic_dir):
    cat_map = {}
    files = glob.glob(os.path.join(dic_dir, "*.txt"))
    for fp in files:
        fname = os.path.basename(fp)
        if "filter" in fname or fname.endswith(".bak"): continue
        cat_name = fname.replace(".txt", "")
        
        words_all = []
        words_pos = []
        words_neg = []
        
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                if "---" in line: break
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    w = parts[0]
                    pol = parts[1].strip()
                    if w:
                        words_all.append(w)
                        if pol == 'pos': words_pos.append(w)
                        elif pol == 'nag': words_neg.append(w)
                        
        cat_map[cat_name] = {
            'All': words_all,
            'Pos': words_pos,
            'Neg': words_neg
        }
    return cat_map

def run_target_permutation(target_scores, pool_scores, n_perms):
    obs_mean_abs = np.mean([abs(s) for s in target_scores])
    n_target = len(target_scores)
    
    if len(pool_scores) < n_target:
        return obs_mean_abs, 1.0 
    
    pool_scores_arr = np.array(pool_scores)
    null_means = []
    
    for _ in range(n_perms):
        random_sample = np.random.choice(pool_scores_arr, n_target, replace=False)
        null_means.append(np.mean(np.abs(random_sample)))
        
    n_extreme = sum(1 for x in null_means if x >= obs_mean_abs)
    p_value = (n_extreme + 1) / (n_perms + 1)
    
    return obs_mean_abs, p_value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', default=DEFAULT_CACHE_FILE)
    parser.add_argument('--gender-file', default=DEFAULT_GENDER_FILE)
    parser.add_argument('--dic-dir', default=DEFAULT_DIC_DIR) 
    parser.add_argument('--output', default=OUTPUT_DIR)
    parser.add_argument('--permutations', type=int, default=N_PERMUTATIONS)
    
    if 'ipykernel' in sys.modules:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    # AutoDL 自动适配
    if os.path.exists("/root/autodl-tmp"):
        root = "/root/autodl-tmp"
        if args.cache == DEFAULT_CACHE_FILE: 
            p1 = os.path.join(root, "result/global_vector_cache.pkl")
            p2 = os.path.join(root, "result/vector_cache.pkl")
            if os.path.exists(p1): args.cache = p1
            else: args.cache = p2
        if args.gender_file == DEFAULT_GENDER_FILE: args.gender_file = os.path.join(root, "static/woman-man/basic.txt")
        if args.dic_dir == DEFAULT_DIC_DIR: args.dic_dir = os.path.join(root, "dic")
        if args.output == OUTPUT_DIR: args.output = os.path.join(root, "result/permutation_test_target")
    
    print("="*60)
    print(f"Target Word Permutation Test (N={args.permutations})")
    print("="*60)
        
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Loading cache: {args.cache}")
    if not os.path.exists(args.cache):
        print(f"Error: Cache file not found: {args.cache}")
        return

    with open(args.cache, 'rb') as f:
        cache = pickle.load(f)
        
    m_words, f_words = load_gender_pairs(args.gender_file)
    categories = load_moral_categories(args.dic_dir)
    print(f"Gender words: {len(m_words)} male, {len(f_words)} female")
    print(f"Categories: {len(categories)}")
    
    results = []
    
    # 2. 遍历朝代 (Global Cache 通常只有 'Global' 键，但也可能兼容 dynasty cache)
    for dynasty, vocab_dict in cache.items():
        print(f"\nProcessing {dynasty}...")
        
        valid_vocab = {w: v for w, v in vocab_dict.items() 
                       if v is not None and not np.any(np.isnan(v))}
        
        m_vecs = [valid_vocab[w] for w in m_words if w in valid_vocab]
        f_vecs = [valid_vocab[w] for w in f_words if w in valid_vocab]
        
        if not m_vecs or not f_vecs:
            print("  Skipping: Insufficient gender words.")
            continue
            
        gender_axis = get_axis_vector(m_vecs, f_vecs)
        
        all_moral_words = set()
        for subcats in categories.values():
             for words in subcats.values():
                 all_moral_words.update(words)
        
        all_gender_words = set(m_words + f_words)
        forbidden_words = all_moral_words.union(all_gender_words)
        
        pool_scores = []
        for w, v in valid_vocab.items():
            if w not in forbidden_words:
                score = np.dot(v, gender_axis)
                pool_scores.append(score)
        
        if len(pool_scores) < 100:
            print("  Skipping: Vocabulary pool too small.")
            continue

        dynasty_res = []
        # C. 对每个类别及其子类进行检验
        pbar = tqdm(categories.items(), desc=f"  Testing {dynasty}", unit="cat")
        
        for cat_name, subcats in pbar:
            for sub_type, words in subcats.items():
                if not words: continue
                
                target_scores = []
                valid_w_count = 0
                for w in words:
                    if w in valid_vocab:
                        score = np.dot(valid_vocab[w], gender_axis)
                        target_scores.append(score)
                        valid_w_count += 1
                
                if valid_w_count < 3: 
                    tqdm.write(f"  - {cat_name:<10} [{sub_type:<3}]: (n={valid_w_count}, skipped)")
                    continue
                    
                pbar.set_postfix_str(f"{cat_name}-{sub_type}")
                obs_bias, p_val = run_target_permutation(target_scores, pool_scores, args.permutations)
                
                sig = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "")
                
                # 用户要求打印所有结果
                tqdm.write(f"  - {cat_name:<10} [{sub_type:<3}]: Bias={obs_bias:.4f}, p={p_val:.4f} {sig} (n={valid_w_count})")
                
                dynasty_res.append({
                    "Dynasty": dynasty,
                    "Category": cat_name,
                    "Subcategory": sub_type, # New Column
                    "Count": valid_w_count,
                    "Obs_Bias": obs_bias,
                    "P_value": p_val,
                    "Significance": sig
                })
            
        results.extend(dynasty_res)

    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(args.output, "target_permutation_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved results to {csv_path}")
    else:
        print("\nNo results generated.")

if __name__ == "__main__":
    main()
