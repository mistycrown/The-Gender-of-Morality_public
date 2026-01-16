"""
================================================================================
历时目标词置换检验脚本 (Diachronic Target Permutation Test)
================================================================================
每次更新代码文件时请更新代码头注释！

【功能描述】
对历史各朝代的道德词汇进行置换检验，验证性别偏见的历时演变显著性。

主要步骤：
1. 加载历时词向量缓存 (Dynasty -> Word -> Vector)
2. 遍历每个朝代：
   a. 构建当朝性别轴
   b. 构建当朝背景池 (由中性控制词组成)
   c. 对每个道德类别进行 1000 次置换检验
3. 生成显著性统计报告

【输入】
- result/diachronic_vector_cache.pkl - 历时向量缓存
- static/woman-man/basic.txt - 性别种子词

【输出】
- result/diachronic_permutation_test/diachronic_permutation_results.csv - 历时检验结果表

【参数说明】
- --cache: 向量缓存路径
- --permutations: 置换次数 (默认 1000)

【依赖库】
- numpy, pandas, scipy, tqdm

【作者】AI辅助生成
【创建日期】2026-01-16
【最后修改】2026-01-16 - 初始开源版本整理

【修改历史】
- 2026-01-16: 整理为开源版本，规范代码头
================================================================================
"""

import os
import sys
from typing import List, Dict
import pickle
import argparse
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from scipy.stats import norm

# ==================== 配置 ====================
DEFAULT_CACHE_FILE = r"result/diachronic_vector_cache.pkl"
DEFAULT_GENDER_FILE = r"static/woman-man/basic.txt"
DEFAULT_DIC_DIR = r"static/dic"
OUTPUT_DIR = r"result/diachronic_permutation_test"
N_PERMUTATIONS = 1000

# 对照词表 (必须与 cache 脚本一致)
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

DYNASTY_ORDER = ['先秦两汉', '魏晋南北朝', '隋唐', '宋', '元', '明', '清']

def get_axis_vector(male_vecs: List[np.ndarray], female_vecs: List[np.ndarray]) -> np.ndarray:
    if not male_vecs or not female_vecs: return None
    mean_m = np.mean(male_vecs, axis=0)
    mean_f = np.mean(female_vecs, axis=0)
    return mean_m - mean_f

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
                    pol = parts[1].strip() # 'pos' or 'nag'
                    
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
        return obs_mean_abs, 1.0 # Pool too small
    
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
    parser.add_argument('--output', default=OUTPUT_DIR)
    parser.add_argument('--permutations', type=int, default=N_PERMUTATIONS)

    # 兼容 Jupyter / AutoDL
    if 'ipykernel' in sys.modules:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    
    # AutoDL 适配
    if os.path.exists("/root/autodl-tmp"):
        root = "/root/autodl-tmp"
        if args.cache == DEFAULT_CACHE_FILE: args.cache = os.path.join(root, "result/diachronic_vector_cache.pkl")
        if args.output == OUTPUT_DIR: args.output = os.path.join(root, "result/diachronic_permutation_test")
        # 修正DIC路径
        DEFAULT_DIC_DIR = os.path.join(root, "dic")
        DEFAULT_GENDER_FILE = os.path.join(root, "static/woman-man/basic.txt")
    
    final_dic_dir = DEFAULT_DIC_DIR
    final_gender_file = DEFAULT_GENDER_FILE
    if os.path.exists("/root/autodl-tmp"):
         final_dic_dir = "/root/autodl-tmp/dic" if os.path.exists("/root/autodl-tmp/dic") else "static/dic"
         final_gender_file = "/root/autodl-tmp/static/woman-man/basic.txt"
    
    print(f"Loading cache from {args.cache} ...")
    if not os.path.exists(args.cache):
        print(f"Cache not found: {args.cache}")
        return
        
    with open(args.cache, 'rb') as f:
        cache = pickle.load(f) # {Dynasty: {Word: Vector}}
    print(f"✅ Cache loaded. Found dynasties: {list(cache.keys())}")
        
    m_words, f_words = load_gender_pairs(final_gender_file)
    categories = load_moral_categories(final_dic_dir)
    os.makedirs(args.output, exist_ok=True)
    
    results = []
    
    for dynasty in DYNASTY_ORDER:
        if dynasty not in cache: 
            print(f"⚠️  Dynasty {dynasty} not found in cache, skipping.")
            continue
        
        print(f"\n[{dynasty}] Preparing data...")
        vocab = cache[dynasty]
        
        # 1. 构建当朝性别轴
        m_vecs = [vocab[w] for w in m_words if w in vocab]
        f_vecs = [vocab[w] for w in f_words if w in vocab]
        
        if not m_vecs or not f_vecs:
            print("  Skipping: Insufficient gender words")
            continue
            
        axis = get_axis_vector(m_vecs, f_vecs)
        axis = axis / np.linalg.norm(axis)
        print(f"  Gender Axis constructed ({len(m_vecs)}M-{len(f_vecs)}F).")
        
        # 2. 构建当朝背景池 (Control Words)
        pool_scores = []
        for w in CONTROL_WORDS:
            if w in vocab:
                v = vocab[w]
                score = np.dot(v, axis)
                pool_scores.append(score)
                
        if len(pool_scores) < 50:
            print(f"  Skipping: Control pool too small ({len(pool_scores)})")
            continue
        print(f"  Control Pool size: {len(pool_scores)}")
            
        # 3. 检验每个类别 (All, Pos, Neg)
        pbar = tqdm(categories.items(), desc=f"  Testing {dynasty}", unit="cat")
        
        for cat, subcats in pbar:
            # subcats is {'All': [], 'Pos': [], 'Neg': []}
            
            for sub_type, words in subcats.items():
                if not words: continue
                
                target_scores = []
                valid_w_count = 0
                for w in words:
                    if w in vocab:
                        score = np.dot(vocab[w], axis)
                        target_scores.append(score)
                        valid_w_count += 1
                
                # 至少要有3个词才做统计检验
                if valid_w_count < 3: 
                    tqdm.write(f"    - {cat:<15} [{sub_type:<3}]: (n={valid_w_count}, skipped)")
                    continue
                
                pbar.set_postfix_str(f"Calc {cat}-{sub_type}")
                obs_bias, p_val = run_target_permutation(target_scores, pool_scores, args.permutations)
                
                sig = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "")
                
                # 用户要求打印所有结果
                tqdm.write(f"    - {cat:<15} [{sub_type:<3}]: Bias={obs_bias:.4f}, p={p_val:.4f} {sig} (n={valid_w_count})")
                
                results.append({
                    "Dynasty": dynasty,
                    "Category": cat,
                    "Subcategory": sub_type, # New Column
                    "Obs_Bias": obs_bias,
                    "P_value": p_val,
                    "Significance": sig,
                    "Pool_Size": len(pool_scores),
                    "Category_Size": valid_w_count
                })
            
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.output, "diachronic_permutation_results.csv"), index=False)
    print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()
