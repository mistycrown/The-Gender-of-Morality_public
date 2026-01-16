"""
================================================================================
DeepSeek API 语义标注脚本
================================================================================
每次更新代码文件时请更新代码头注释！

【功能描述】
1. 读取 data/ngram_match/ 下的 *-all.txt 文件
2. 解析句子，过滤已标注（结尾含 ✓ 或 ✗）的句子
3. 对每个文件，按朝代均匀抽样共 50 句
4. 调用 DeepSeek API 判断句子中目标词是否为道德义项
5. 结果分类保存到 result/annotation_task/文件名/{positive,negative,uncertain}.txt
6. 支持断点续传：检查已输出的文件，避免重复标注
7. 支持自定义提示词：通过 --custom-prompt 参数注入高优先级指令

【输入】
- data/ngram_match/文件名.txt - 包含待标注句子的源文件
- 命令行参数 --custom-prompt (可选) - 自定义提示词

【输出】
- result/YYYYMMDD-annotation_task/文件名/positive.txt - 正例结果
- result/YYYYMMDD-annotation_task/文件名/negative.txt - 反例结果
- result/YYYYMMDD-annotation_task/文件名/uncertain.txt - 不确定结果

【参数说明】
- files: 指定处理的文件路径
- --all: 处理 data/ngram_match/ 下所有 *-all.txt
- --dry-run: 仅抽样不调用API
- --custom-prompt: 自定义提示词，最高优先级

【依赖库】
- openai: 调用 DeepSeek API
- re, json, random, time, pathlib: 基础库

【作者】AI辅助生成
【创建日期】2026-01-03
【最后修改】2026-01-09 - 增加自定义提示词功能

【修改历史】
- 2026-01-03: 初版创建
- 2026-01-04: 增加断点续传功能（检查result目录已处理数量）
- 2026-01-04: 修复mkdir逻辑bug，优化prompt增加类别全称
- 2026-01-09: 增加 --custom-prompt 参数，允许用户自定义高优先级提示词
================================================================================
"""

import os
import re
import random
import json
import time
from pathlib import Path
from collections import defaultdict
import argparse

# 尝试导入 openai，如果不存在则报错
try:
    from openai import OpenAI
except ImportError:
    print("错误: 请先安装 openai 库: pip install openai")
    exit(1)

# 配置
# 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"  # DeepSeek API 地址

def setup_client():
    if not DEEPSEEK_API_KEY:
        print("Error: DEEPSEEK_API_KEY environment variable not set.")
        print("Please set it in your environment or via .env file to use the LLM features.")
        return None
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

def parse_line(line):
    """
    解析行：[朝代] [分类] 句子
    返回：(dynasty, category, sentence_content)
    """
    match = re.match(r'^\[(.*?)\] \[(.*?)\] (.*)', line)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None, None, line

def load_and_filter_sentences(file_path):
    """
    加载文件，返回按朝代分组的句子列表
    filtered_sentences[dynasty] = [(category, content, original_line)]
    """
    sentences_by_dynasty = defaultdict(list)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 过滤已标注的
            if line.endswith('✓') or line.endswith('✗'):
                continue
                
            dynasty, category, content = parse_line(line)
            if dynasty:
                sentences_by_dynasty[dynasty].append((category, content, line))
                
    return sentences_by_dynasty

def sample_sentences(sentences_by_dynasty, total_target=50):
    """
    均匀抽样
    """
    dynasties = list(sentences_by_dynasty.keys())
    if not dynasties:
        return []
    
    selected_sentences = []
    remaining_target = total_target
    
    # 第一轮分配
    for i, dynasty in enumerate(dynasties):
        # 剩余朝代数
        remaining_dynasties = len(dynasties) - i
        # 本轮应抽数量 (均分 surplus)
        target = int(remaining_target / remaining_dynasties)
        if target == 0 and remaining_target > 0:
            target = 1 # 至少抽1个
            
        candidates = sentences_by_dynasty[dynasty]
        k = min(len(candidates), target)
        
        sampled = random.sample(candidates, k)
        selected_sentences.extend(sampled)
        
        remaining_target -= k
        
    # 如果还有余额（因为某些朝代不够），且还有剩余池子，可以继续补（简化起见，不再补，除非必要）
    # 用户要求“均匀分布”，如果某朝代不够，就全取了，不再从其他朝代多取，以免不均匀。
    # 但如果要凑够50，可以再遍历一遍。
    
    # 简单策略：如果总数不够50，且还有未选的，随机补齐（优先未选的朝代？不，随机补齐即可）
    if len(selected_sentences) < total_target:
        # 收集所有未选的
        pool = []
        for dynasty, items in sentences_by_dynasty.items():
            for item in items:
                if item not in selected_sentences:
                    pool.append(item)
        
        needed = total_target - len(selected_sentences)
        if needed > 0 and pool:
            selected_sentences.extend(random.sample(pool, min(len(pool), needed)))
            
    return selected_sentences

def get_target_word(filename):
    """
    从文件名解析目标词
    文件名格式：01-nag-伐-all.txt -> 伐
    """
    parts = filename.split('-')
    if len(parts) >= 3:
        return parts[2]
    return None

CATEGORY_NAMES = {
    "01": "关爱 / 伤害 (Care / Harm)",
    "02": "权威 / 颠覆 (Hierarchy / Subversion)",
    "03": "忠诚 / 背叛 (Loyalty / Betrayal)",
    "04": "公平 / 偏私 (Fairness / Partiality)",
    "05": "高洁 / 淫亵 (Noble / Profanity)",
    "07": "浪费 / 效率 (Waste / Efficiency)",
    "09": "勤奋 / 懒惰 (Diligence / Laziness)",
    "11": "谦逊 / 傲慢 (Modesty / Arrogance)",
    "12": "勇敢 / 怯懦 (Courage / Cowardice)",
}

def call_deepseek_api(client, word, sentence, category_str, custom_prompt=None):
    """
    调用 API 判断
    """
    cat_id = category_str.split("-")[0]
    cat_name = CATEGORY_NAMES.get(cat_id, "未知类别")

    base_prompt = f"""
请判断以下句子中“{word}”一词的用法。

背景：我们在构建一个“性别道德”词典。
该词属于类别“{cat_name}”（编号 {category_str}）。
注意：{category_str} 中，nag 代表负面/伤害/颠覆等，pos 代表正面/关爱/权威等。

任务：
我们需要区分该词在句子中是否具有“道德/社会交互/评价”属性（即【正例】），还是仅表达“纯物理/非道德”含义（即【反例】）。
"""

    if custom_prompt:
        base_prompt += f"""
【特别重要指令】(优先级最高)：
{custom_prompt}
请严格遵循上述特别指令进行判断。如果特别指令与通用标准冲突，以特别指令为准。
"""

    base_prompt += f"""
通用判断标准：
1. 【正例】(Positive / Moral)：
   - 表达道德评价、行为规范、社会交互中的善恶/对错。
   - 包含具有道德隐喻的行为。例如：“攻”、“伐”如果是指“攻打国家”、“讨伐罪恶”，涉及暴力、伤害或正义，**属于道德范畴（关爱/伤害）**，是正例。
   - 只有纯粹的物理动作（如“伐木”、“攻击石头”）且无拟人/社会隐喻时，才不算。

2. 【反例】(Negative / Non-Moral)：
   - 纯物理动作，不涉及人际/社会/政治交互。
   - 专有名词（人名、爵位、地名）或与道德评价无关的含义。

3. 【不确定】(Uncertain)：
   - 确实无法判断，或上下文缺失导致歧义。
   - 仅是中性描述，难以归类为道德或物理。

句子：{sentence}

请只以 JSON 格式返回结果：
{{
    "result": "positive", // (道德/社会含义)
    "reason": "简短理由"
}}
或者 result 为 "negative" (纯物理/无关) 或 "uncertain"。
"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个古汉语语义分析专家。请以JSON格式输出。"},
                {"role": "user", "content": base_prompt}
            ],
            response_format={ "type": "json_object" },
            temperature=0.1
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"API 调用失败: {e}")
        return {"result": "uncertain", "reason": f"API Error: {e}"}

def process_file(file_path, client, dry_run=False, custom_prompt=None):
    print(f"\n处理文件: {file_path.name}")
    
    word = get_target_word(file_path.name)
    if not word:
        print(f"无法从文件名解析目标词: {file_path.name}")
        return

    sentences_by_dynasty = load_and_filter_sentences(file_path)
    total_candidates = sum(len(v) for v in sentences_by_dynasty.values())
    print(f"  候选句子总数: {total_candidates}")
    
    sampled = sample_sentences(sentences_by_dynasty, total_target=50)
    print(f"  抽样数量: {len(sampled)}")
    
    if dry_run:
        print("  Dry Run 模式，不调用 API。")
        for cat, content, line in sampled[:5]:
            print(f"    [示例] {content[:20]}...")
        return

    # 准备输出目录
    import datetime
    date_str = datetime.date.today().strftime("%Y%m%d")
    # 为了保持一致性，这里强制使用用户指定的 20260103
    date_str = "20260103" 
    
    base_dir = file_path.parent.parent.parent
    output_base = base_dir / "result" / f"{date_str}-annotation_task" / file_path.stem
    
    # 检查已处理的
    processed_lines = set()
    categories = ["positive", "negative", "uncertain"]
    
    for cat in categories:
        fpath = output_base / f"{cat}.txt"
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" | ")
                    if parts:
                        original = parts[0]
                        processed_lines.add(original)
    
    current_count = len(processed_lines)
    print(f"  已处理条数: {current_count}")
    
    if current_count >= 50:
        print(f"  已达到目标数量(50)，跳过。")
        return

    # 过滤掉已处理的候选句子，避免重复采样
    # 注意：load_and_filter_sentences 返回的是 dict[dynasty] -> list(tuple)
    # 我们需要先过滤一下
    filtered_sentences_by_dynasty = defaultdict(list)
    for dynasty, items in sentences_by_dynasty.items():
        for item in items:
            # item: (category, content, original_line)
            if item[2] not in processed_lines:
                filtered_sentences_by_dynasty[dynasty].append(item)
    
    needed = 50 - current_count
    print(f"  还需抽取: {needed}")
    
    sampled = sample_sentences(filtered_sentences_by_dynasty, total_target=needed)
    print(f"  新抽样数量: {len(sampled)}")
    
    if not sampled:
        print("  无更多候选句子可抽样。")
        return

    if dry_run:
        print("  Dry Run 模式，不调用 API。")
        for cat, content, line in sampled[:5]:
            print(f"    [示例] {content[:20]}...")
        return

    output_base.mkdir(parents=True, exist_ok=True)

    output_files = {
        "positive": open(output_base / "positive.txt", "a", encoding="utf-8"), # append mode
        "negative": open(output_base / "negative.txt", "a", encoding="utf-8"),
        "uncertain": open(output_base / "uncertain.txt", "a", encoding="utf-8")
    }
    
    # 提取类别信息，例如 01-nag
    category_parts = file_path.name.split('-')[:2]
    category_str = "-".join(category_parts)
    
    count = {"positive": 0, "negative": 0, "uncertain": 0}
    
    for i, (category, content, original_line) in enumerate(sampled):
        print(f"  [{i+1}/{len(sampled)}] 分析: {content[:10]}...", end="", flush=True)
        
        # 传递 custom_prompt
        api_res = call_deepseek_api(client, word, content, category_str, custom_prompt=custom_prompt)
        result_type = api_res.get("result", "uncertain").lower()
        reason = api_res.get("reason", "无理由")
        
        if result_type not in output_files:
            result_type = "uncertain"
            
        print(f" -> {result_type}")
        
        # 写入文件
        # 格式：原始行 | 理由
        output_files[result_type].write(f"{original_line} | {reason}\n")
        # 立即刷新，防止中断丢失太远
        output_files[result_type].flush()
        
        count[result_type] += 1
        
        # 避免速率限制
        time.sleep(0.5)
        
    for f in output_files.values():
        f.close()
        
    print(f"  本轮完成。新增: 正例 {count['positive']}, 反例 {count['negative']}, 不确定 {count['uncertain']}")
    print(f"  结果保存在: {output_base}")

def main():
    parser = argparse.ArgumentParser(description="DeepSeek API 语义标注")
    parser.add_argument("files", nargs="*", help="指定处理的文件路径，支持通配符（需shell展开）")
    parser.add_argument("--all", action="store_true", help="处理 data/ngram_match/ 下所有 *-all.txt")
    parser.add_argument("--dry-run", action="store_true", help="仅抽样不调用API")
    parser.add_argument("--custom-prompt", help="自定义提示词（最高优先级）")
    args = parser.parse_args()
    
    # Use current working directory or script parent as base
    # Assuming script is in src/preprocessing and we want to go up to project root
    script_path = Path(__file__).resolve()
    # Go up 2 levels: src/preprocessing -> src -> root
    project_root = script_path.parent.parent.parent
    
    # Or simply use CWD if running from root
    base_dir = Path.cwd()
    if (base_dir / "src").exists():
        pass # We are likely in root
    else:
        # Fallback to relative to script
        base_dir = project_root

    data_dir = base_dir / "data" / "ngram_match"
    
    target_files = []
    if args.all:
        target_files = list(data_dir.glob("*-all.txt"))
    elif args.files:
        for f in args.files:
            path = Path(f)
            if path.exists():
                target_files.append(path)
            else:
                # 尝试相对于 data_dir
                path = data_dir / f
                if path.exists():
                    target_files.append(path)
    
    if not target_files:
        print("未找到目标文件。请使用 --all 或指定文件名。")
        return

    client = None
    if not args.dry_run:
        client = setup_client()
        if not client:
            return

    for file_path in target_files:
        process_file(file_path, client, dry_run=args.dry_run, custom_prompt=args.custom_prompt)

if __name__ == "__main__":
    main()
