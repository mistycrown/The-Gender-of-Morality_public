"""
================================================================================
性别轴线计算与验证脚本
================================================================================

【功能描述】
使用配对相减法计算性别轴线，并验证其有效性。
主要步骤：
1. 从basic.txt中读取性别词对（如：男-女、夫-妻等）
2. 使用SikuBERT获取每个词的词向量
3. 计算配对差异向量的平均值作为性别轴线
4. 归一化性别轴线
5. 使用测试词验证轴线有效性

【输入】
- static/woman-man/basic.txt - 性别词对文件
  格式：每行两个词，空格分隔，左边为男性词，右边为女性词

【输出】
- result/20251230-gender-axis/gender_axis.npy - 归一化的性别轴线向量
- result/20251230-gender-axis/gender_axis_validation.txt - 验证报告
- result/20251230-gender-axis/gender_pairs_vectors.txt - 性别词对的向量信息

【参数说明】
- MODEL_NAME: SikuBERT模型路径
- TEST_WORDS: 用于验证的测试词对

【依赖库】
- transformers: BERT模型加载
- torch: 深度学习框架
- numpy: 数值计算

【作者】AI辅助生成
【创建日期】2025-12-30
【最后修改】2025-12-30 13:50 - 调整结果文件夹结构

【修改历史】
- 2025-12-30 13:30: 创建脚本，实现性别轴线计算和验证功能
- 2025-12-30 13:43: 扩展测试词集，新增12个测试词，涵盖称谓、尊称、亲属、配偶、社会角色、生物性别等类别
- 2025-12-30 13:50: 调整结果输出路径，将所有结果存储到result/20251230-gender-axis/子文件夹
================================================================================
"""

import os
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# ==================== 配置区域 ====================
MODEL_NAME = "SIKU-BERT/sikubert"
BASIC_GENDER_FILE = r"static\woman-man\basic.txt"
# 每个实验使用独立的结果子文件夹
EXPERIMENT_NAME = "20251230-gender-axis"
RESULT_DIR = os.path.join("result", EXPERIMENT_NAME)
GENDER_AXIS_FILE = os.path.join(RESULT_DIR, "gender_axis.npy")
VALIDATION_FILE = os.path.join(RESULT_DIR, "gender_axis_validation.txt")
PAIRS_INFO_FILE = os.path.join(RESULT_DIR, "gender_pairs_vectors.txt")

# 测试词对：(词, 预期性别, 预期符号)
# 预期符号：'positive' 表示应该是正数（男性端），'negative' 表示应该是负数# 测试词（用于验证性别轴线）
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

# ==================== SikuBERT 工具函数 ====================

def load_sikubert(model_name_or_path: str = MODEL_NAME, device: Optional[str] = None):
    """加载SikuBERT模型"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"加载模型: {model_name_or_path}")
    print(f"设备: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)
    model.eval()
    model.to(device)
    embedding_weight: torch.Tensor = model.get_input_embeddings().weight.detach().cpu()
    
    print(f"✓ 模型加载完成\n")
    return tokenizer, model, embedding_weight, device


def get_token_ids(tokenizer: AutoTokenizer, text: str) -> List[int]:
    """获取文本的token IDs"""
    tokens = tokenizer.tokenize(text)
    if not tokens:
        return []
    ids = tokenizer.convert_tokens_to_ids(tokens)
    unk_id = tokenizer.unk_token_id
    return [int(i) for i in ids if i is not None and i != unk_id]


def get_static_vector(embedding_weight: torch.Tensor, token_ids: List[int]) -> Optional[np.ndarray]:
    """获取token IDs的平均词向量"""
    if not token_ids:
        return None
    with torch.no_grad():
        index_tensor = torch.tensor(token_ids, dtype=torch.long)
        vectors = embedding_weight[index_tensor]
        avg_vector = vectors.mean(dim=0).cpu().numpy()
    return avg_vector


def get_word_vector(tokenizer, embedding_weight, word: str) -> Optional[np.ndarray]:
    """获取单个词的词向量"""
    token_ids = get_token_ids(tokenizer, word)
    return get_static_vector(embedding_weight, token_ids)


# ==================== 性别轴线计算函数 ====================

def load_gender_pairs(filepath: str) -> List[Tuple[str, str]]:
    """
    读取性别词对文件
    
    返回:
        [(男性词, 女性词), ...]
    """
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


def calculate_gender_axis(tokenizer, embedding_weight, pairs: List[Tuple[str, str]]) -> Tuple[np.ndarray, List[Dict]]:
    """
    计算性别轴线
    
    步骤：
    1. 计算每一对的差异向量：vec_male - vec_female
    2. 求平均得到性别轴
    3. 归一化
    
    返回:
        (归一化的性别轴线, 每对的详细信息)
    """
    diff_vectors = []
    pairs_info = []
    
    print("计算每对性别词的差异向量...")
    print("-" * 60)
    
    for male_word, female_word in pairs:
        # 获取词向量
        vec_male = get_word_vector(tokenizer, embedding_weight, male_word)
        vec_female = get_word_vector(tokenizer, embedding_weight, female_word)
        
        # 检查是否成功获取向量
        if vec_male is None or vec_female is None:
            print(f"⚠ 警告: 无法获取「{male_word} - {female_word}」的词向量，跳过")
            pairs_info.append({
                'male': male_word,
                'female': female_word,
                'success': False,
                'reason': '无法获取词向量'
            })
            continue
        
        # 计算差异向量
        diff_vec = vec_male - vec_female
        diff_vectors.append(diff_vec)
        
        # 保存信息
        pairs_info.append({
            'male': male_word,
            'female': female_word,
            'success': True,
            'male_norm': float(np.linalg.norm(vec_male)),
            'female_norm': float(np.linalg.norm(vec_female)),
            'diff_norm': float(np.linalg.norm(diff_vec))
        })
        
        print(f"✓ {male_word} - {female_word}: ||diff|| = {np.linalg.norm(diff_vec):.4f}")
    
    print("-" * 60)
    
    if not diff_vectors:
        raise ValueError("无法计算性别轴线：所有词对均无法获取词向量")
    
    # 计算平均差异向量
    gender_axis = np.mean(diff_vectors, axis=0)
    print(f"\n原始性别轴线范数: {np.linalg.norm(gender_axis):.4f}")
    
    # 归一化
    gender_axis_normalized = gender_axis / np.linalg.norm(gender_axis)
    print(f"归一化后性别轴线范数: {np.linalg.norm(gender_axis_normalized):.4f}")
    
    return gender_axis_normalized, pairs_info


def validate_gender_axis(tokenizer, embedding_weight, gender_axis: np.ndarray, test_words: List[Tuple[str, str, str]]) -> List[Dict]:
    """
    验证性别轴线
    
    计算测试词在性别轴线上的投影（点乘）
    预期：男性词得分为正，女性词得分为负
    
    返回:
        测试结果列表
    """
    results = []
    
    print("\n验证性别轴线...")
    print("-" * 60)
    print(f"{'词汇':<10} {'预期性别':<12} {'投影得分':<12} {'验证结果'}")
    print("-" * 60)
    
    for word, gender, expected_sign in test_words:
        # 获取词向量
        vec = get_word_vector(tokenizer, embedding_weight, word)
        
        if vec is None:
            result = {
                'word': word,
                'expected_gender': gender,
                'expected_sign': expected_sign,
                'score': None,
                'passed': False,
                'reason': '无法获取词向量'
            }
            print(f"{word:<10} {gender:<12} {'N/A':<12} ✗ (无法获取词向量)")
        else:
            # 计算投影得分（点乘）
            score = float(np.dot(vec, gender_axis))
            
            # 验证符号是否符合预期
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
            print(f"{word:<10} {gender:<12} {score:<12.6f} {status} ({result['reason']})")
        
        results.append(result)
    
    print("-" * 60)
    
    # 统计通过率
    passed_count = sum(1 for r in results if r['passed'])
    total_count = len(results)
    pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0
    
    print(f"\n验证通过率: {passed_count}/{total_count} ({pass_rate:.1f}%)")
    
    return results


# ==================== 保存结果函数 ====================

def save_gender_axis(gender_axis: np.ndarray, filepath: str):
    """保存性别轴线到文件"""
    np.save(filepath, gender_axis)
    print(f"\n✓ 性别轴线已保存: {filepath}")


def save_validation_report(pairs_info: List[Dict], validation_results: List[Dict], filepath: str):
    """保存验证报告"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("性别轴线计算与验证报告\n")
        f.write("=" * 60 + "\n\n")
        
        # 1. 性别词对信息
        f.write("【1. 性别词对信息】\n")
        f.write("-" * 60 + "\n")
        for info in pairs_info:
            if info['success']:
                f.write(f"{info['male']} - {info['female']}:\n")
                f.write(f"  男性词范数: {info['male_norm']:.4f}\n")
                f.write(f"  女性词范数: {info['female_norm']:.4f}\n")
                f.write(f"  差异向量范数: {info['diff_norm']:.4f}\n\n")
            else:
                f.write(f"{info['male']} - {info['female']}: ✗ {info['reason']}\n\n")
        
        # 2. 验证结果
        f.write("\n【2. 验证结果】\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'词汇':<10} {'预期性别':<15} {'预期符号':<10} {'投影得分':<15} {'验证结果'}\n")
        f.write("-" * 60 + "\n")
        
        for result in validation_results:
            word = result['word']
            gender = result['expected_gender']
            sign = result['expected_sign']
            score = result['score']
            passed = result['passed']
            reason = result['reason']
            
            score_str = f"{score:.6f}" if score is not None else "N/A"
            status = '✓' if passed else '✗'
            
            f.write(f"{word:<10} {gender:<15} {sign:<10} {score_str:<15} {status} ({reason})\n")
        
        # 3. 统计摘要
        passed_count = sum(1 for r in validation_results if r['passed'])
        total_count = len(validation_results)
        pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("【3. 统计摘要】\n")
        f.write("-" * 60 + "\n")
        f.write(f"使用的性别词对数: {sum(1 for p in pairs_info if p['success'])}/{len(pairs_info)}\n")
        f.write(f"测试词数: {total_count}\n")
        f.write(f"验证通过数: {passed_count}\n")
        f.write(f"通过率: {pass_rate:.1f}%\n")
    
    print(f"✓ 验证报告已保存: {filepath}")


def save_pairs_info(pairs_info: List[Dict], filepath: str):
    """保存性别词对的详细信息"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("性别词对向量信息\n")
        f.write("=" * 60 + "\n\n")
        
        for info in pairs_info:
            if info['success']:
                f.write(f"词对: {info['male']} - {info['female']}\n")
                f.write(f"  男性词向量范数: {info['male_norm']:.6f}\n")
                f.write(f"  女性词向量范数: {info['female_norm']:.6f}\n")
                f.write(f"  差异向量范数: {info['diff_norm']:.6f}\n")
                f.write("\n")
            else:
                f.write(f"词对: {info['male']} - {info['female']}\n")
                f.write(f"  状态: 失败 ({info['reason']})\n\n")
    
    print(f"✓ 词对信息已保存: {filepath}")


# ==================== 主函数 ====================

def main():
    print("=" * 60)
    print("性别轴线计算与验证")
    print("=" * 60)
    print()
    
    # 创建结果目录
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # 1. 加载模型
    tokenizer, model, embedding_weight, device = load_sikubert(MODEL_NAME)
    
    # 2. 读取性别词对
    print(f"读取性别词对文件: {BASIC_GENDER_FILE}")
    pairs = load_gender_pairs(BASIC_GENDER_FILE)
    print(f"✓ 共读取 {len(pairs)} 对性别词\n")
    
    # 3. 计算性别轴线
    gender_axis, pairs_info = calculate_gender_axis(tokenizer, embedding_weight, pairs)
    
    # 4. 验证性别轴线
    validation_results = validate_gender_axis(tokenizer, embedding_weight, gender_axis, TEST_WORDS)
    
    # 5. 保存结果
    print("\n保存结果文件...")
    save_gender_axis(gender_axis, GENDER_AXIS_FILE)
    save_validation_report(pairs_info, validation_results, VALIDATION_FILE)
    save_pairs_info(pairs_info, PAIRS_INFO_FILE)
    
    print("\n" + "=" * 60)
    print("✓ 性别轴线计算和验证完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
