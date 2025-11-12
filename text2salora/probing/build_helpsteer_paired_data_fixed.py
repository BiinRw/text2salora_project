"""
修正版: 基于HelpSteer数据集构建多维度探针训练数据
关键修改: 将评分转换为0/1标签
- 高分回复 -> 标签1 (好)
- 低分回复 -> 标签0 (坏)
"""

from datasets import load_dataset
from collections import defaultdict
import json
import os
from tqdm import tqdm
import random


def build_paired_data_for_dimension(dataset, dimension, min_pairs=5000, score_threshold=2):
    """
    为指定维度构建配对数据
    
    关键变化: 不再保存原始评分,而是转换为0/1标签
    - good_pairs (高分) -> 标签1
    - bad_pairs (低分) -> 标签0
    
    Args:
        dataset: HelpSteer数据集
        dimension: 评分维度名称 (helpfulness, correctness, etc.)
        min_pairs: 最少需要的数据对数量
        score_threshold: 分数差阈值,只有差距>=threshold的才算有效对比
    
    Returns:
        good_pairs: [(prompt, response, label=1), ...] 高分回复
        bad_pairs: [(prompt, response, label=0), ...] 低分回复
    """
    print(f"\n{'='*70}")
    print(f"📊 构建维度: {dimension}")
    print(f"{'='*70}")
    
    # 按prompt分组
    prompt_groups = defaultdict(list)
    
    print("📥 按prompt分组数据...")
    for item in tqdm(dataset):
        if item[dimension] is not None:
            prompt_groups[item['prompt']].append({
                'response': item['response'],
                'score': item[dimension]
            })
    
    print(f"✅ 共有 {len(prompt_groups)} 个不同的prompt")
    
    # 对于每个prompt,找出评分差异最大的回复对
    good_pairs = []  # 高分 -> 标签1
    bad_pairs = []   # 低分 -> 标签0
    
    print(f"🔍 提取评分对比明显的回复对...")
    for prompt, responses in tqdm(prompt_groups.items()):
        if len(responses) < 2:
            continue
        
        # 按分数排序
        responses.sort(key=lambda x: x['score'])
        
        # 取最低分和最高分
        lowest = responses[0]
        highest = responses[-1]
        
        # 只有分数差距足够大才加入
        if highest['score'] - lowest['score'] >= score_threshold:
            # ✅ 关键修改: 保存为0/1标签,而不是原始评分
            good_pairs.append({
                'prompt': prompt,
                'response': highest['response'],
                'label': 1,  # 高分 = 好的回复
                'original_score': float(highest['score'])  # 仅供参考
            })
            bad_pairs.append({
                'prompt': prompt,
                'response': lowest['response'],
                'label': 0,  # 低分 = 坏的回复
                'original_score': float(lowest['score'])  # 仅供参考
            })
    
    print(f"\n📈 初步统计:")
    print(f"   原始配对数: {len(good_pairs)}")
    
    # 如果数据不够,降低阈值重新提取
    if len(good_pairs) < min_pairs:
        print(f"⚠️  数据量不足 {min_pairs},降低阈值到1重新提取...")
        good_pairs = []
        bad_pairs = []
        
        for prompt, responses in prompt_groups.items():
            if len(responses) < 2:
                continue
            
            responses.sort(key=lambda x: x['score'])
            lowest = responses[0]
            highest = responses[-1]
            
            if highest['score'] - lowest['score'] >= 1:
                good_pairs.append({
                    'prompt': prompt,
                    'response': highest['response'],
                    'label': 1,
                    'original_score': float(highest['score'])
                })
                bad_pairs.append({
                    'prompt': prompt,
                    'response': lowest['response'],
                    'label': 0,
                    'original_score': float(lowest['score'])
                })
    
    # 如果还是不够,直接按分数中位数分割
    if len(good_pairs) < min_pairs:
        print(f"⚠️  数据量仍不足,使用中位数分割策略...")
        all_responses = []
        for prompt, responses in prompt_groups.items():
            for r in responses:
                all_responses.append({
                    'prompt': prompt,
                    'response': r['response'],
                    'score': r['score']
                })
        
        # 计算中位数
        scores = [r['score'] for r in all_responses]
        median_score = sorted(scores)[len(scores)//2]
        
        # 高于中位数=好,低于中位数=坏
        good_pairs = []
        bad_pairs = []
        for r in all_responses:
            if r['score'] > median_score:
                good_pairs.append({
                    'prompt': r['prompt'],
                    'response': r['response'],
                    'label': 1,
                    'original_score': float(r['score'])
                })
            elif r['score'] < median_score:
                bad_pairs.append({
                    'prompt': r['prompt'],
                    'response': r['response'],
                    'label': 0,
                    'original_score': float(r['score'])
                })
        
        # 打乱并截取
        random.shuffle(good_pairs)
        random.shuffle(bad_pairs)
        good_pairs = good_pairs[:min_pairs]
        bad_pairs = bad_pairs[:min_pairs]
    
    # 确保数量一致且满足最小要求
    min_available = min(len(good_pairs), len(bad_pairs))
    if min_available < min_pairs:
        print(f"⚠️  警告: 只能提取 {min_available} 对数据")
    else:
        min_available = min_pairs
    
    good_pairs = good_pairs[:min_available]
    bad_pairs = bad_pairs[:min_available]
    
    # 统计信息
    good_scores = [p['original_score'] for p in good_pairs]
    bad_scores = [p['original_score'] for p in bad_pairs]
    
    print(f"\n✅ 最终数据统计:")
    print(f"   配对数量: {len(good_pairs)} 对")
    print(f"   Good数据 (label=1): {len(good_pairs)} 条")
    print(f"   Bad数据 (label=0): {len(bad_pairs)} 条")
    print(f"   Good原始评分: {min(good_scores):.1f}-{max(good_scores):.1f} (avg={sum(good_scores)/len(good_scores):.2f})")
    print(f"   Bad原始评分: {min(bad_scores):.1f}-{max(bad_scores):.1f} (avg={sum(bad_scores)/len(bad_scores):.2f})")
    print(f"   平均分差: {sum(good_scores)/len(good_scores) - sum(bad_scores)/len(bad_scores):.2f}")
    print(f"   ✅ 所有数据已标记为0/1标签 (而非原始评分)")
    
    return good_pairs, bad_pairs


def main():
    """主函数"""
    print("="*70)
    print("🔧 修正版: HelpSteer多维度配对数据构建")
    print("="*70)
    print("关键修改: 评分转换为0/1标签")
    print("  - 高分回复 -> label=1 (好)")
    print("  - 低分回复 -> label=0 (坏)")
    print("="*70)
    
    # 加载数据集
    print("\n📥 加载HelpSteer数据集...")
    dataset = load_dataset("nvidia/HelpSteer", split="train")
    print(f"✅ 加载完成! 总样本数: {len(dataset)}")
    
    # 定义维度
    dimensions = {
        'helpfulness': '有用性',
        'correctness': '正确性',
        'coherence': '连贯性',
        'complexity': '复杂性',
        'verbosity': '冗长度'
    }
    
    # 输出目录
    output_dir = 'data/helpsteer_paired'
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个维度构建数据
    for dim_key, dim_name in dimensions.items():
        good_pairs, bad_pairs = build_paired_data_for_dimension(
            dataset, 
            dim_key, 
            min_pairs=5000,
            score_threshold=2
        )
        
        # 保存数据
        good_file = os.path.join(output_dir, f'{dim_key}_good_pairs.json')
        bad_file = os.path.join(output_dir, f'{dim_key}_bad_pairs.json')
        
        with open(good_file, 'w', encoding='utf-8') as f:
            json.dump(good_pairs, f, ensure_ascii=False, indent=2)
        
        with open(bad_file, 'w', encoding='utf-8') as f:
            json.dump(bad_pairs, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 已保存到:")
        print(f"   {good_file}")
        print(f"   {bad_file}")
    
    print("\n" + "="*70)
    print("✅ 所有维度数据构建完成!")
    print(f"📂 输出目录: {output_dir}")
    print(f"📊 总维度数: {len(dimensions)}")
    print(f"📊 总数据量: {len(dimensions) * 5000 * 2} 条 (5个维度 × 5000对 × 2)")
    print("="*70)
    
    # 验证标签
    print("\n🔍 验证标签格式...")
    for dim_key in dimensions.keys():
        good_file = os.path.join(output_dir, f'{dim_key}_good_pairs.json')
        with open(good_file, 'r') as f:
            sample = json.load(f)[0]
        print(f"   {dim_key}: label={sample['label']}, original_score={sample['original_score']}")
    print("✅ 所有标签格式正确!")


if __name__ == '__main__':
    main()
