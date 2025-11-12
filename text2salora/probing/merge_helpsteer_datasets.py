"""
整合 HelpSteer 和 HelpSteer2 数据集

策略:
1. 使用与 ultra-extreme 相同的逻辑: 只选择 0 vs 4 分的极端对比
2. 合并两个数据集的极端样本
3. 为每个维度生成整合后的 good/bad pairs
"""

import json
from datasets import load_dataset
from pathlib import Path
from collections import Counter

def build_ultra_extreme_pairs_from_dataset(dataset, dataset_name):
    """
    从数据集中构建 ultra-extreme pairs (只选择 0 vs 4 分)
    
    Returns:
        dict: {dimension: {'good': [...], 'bad': [...]}}
    """
    dimensions = ['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity']
    results = {dim: {'good': [], 'bad': []} for dim in dimensions}
    
    for dimension in dimensions:
        # 收集 0 分和 4 分的样本
        score_0_samples = []
        score_4_samples = []
        
        for item in dataset:
            score = item.get(dimension)
            if score is None:
                continue
            
            if score == 0.0 or score == 0:
                score_0_samples.append(item)
            elif score == 4.0 or score == 4:
                score_4_samples.append(item)
        
        # 配对: 为每个 prompt 创建 good (4分) 和 bad (0分) 对
        # 如果是同一个 prompt 有不同响应,优先配对
        # 否则分别作为独立样本
        
        # 简化策略: 分别处理 good 和 bad
        for item in score_4_samples:
            results[dimension]['good'].append({
                'prompt': item['prompt'],
                'response': item['response'],
                'source': dataset_name,
                'original_score': 4,
                'label': 0
            })
        
        for item in score_0_samples:
            results[dimension]['bad'].append({
                'prompt': item['prompt'],
                'response': item['response'],
                'source': dataset_name,
                'original_score': 0,
                'label': 1
            })
        
        print(f"\n  {dimension}:")
        print(f"    4分样本: {len(score_4_samples)}")
        print(f"    0分样本: {len(score_0_samples)}")
    
    return results


def merge_helpsteer_datasets(output_dir="data/helpsteer_merged_ultra"):
    """
    整合 HelpSteer 和 HelpSteer2 数据集
    """
    print("=" * 70)
    print("📥 加载 HelpSteer 和 HelpSteer2 数据集")
    print("=" * 70)
    
    # 加载 HelpSteer (原始)
    print("\n正在加载 HelpSteer...")
    helpsteer1 = load_dataset("nvidia/HelpSteer", split="train")
    print(f"✅ HelpSteer: {len(helpsteer1)} 条")
    
    # 加载 HelpSteer2
    print("正在加载 HelpSteer2...")
    helpsteer2 = load_dataset("nvidia/HelpSteer2", split="train")
    print(f"✅ HelpSteer2: {len(helpsteer2)} 条")
    
    print(f"\n总数据量: {len(helpsteer1) + len(helpsteer2)} 条")
    
    # 构建 ultra-extreme pairs
    print("\n" + "=" * 70)
    print("🏗️  构建 Ultra-Extreme Pairs (0 vs 4 分)")
    print("=" * 70)
    
    print("\n--- HelpSteer ---")
    pairs_hs1 = build_ultra_extreme_pairs_from_dataset(helpsteer1, "helpsteer")
    
    print("\n--- HelpSteer2 ---")
    pairs_hs2 = build_ultra_extreme_pairs_from_dataset(helpsteer2, "helpsteer2")
    
    # 合并数据
    print("\n" + "=" * 70)
    print("🔗 合并数据")
    print("=" * 70)
    
    dimensions = ['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity']
    merged_pairs = {}
    
    for dim in dimensions:
        merged_pairs[dim] = {
            'good': pairs_hs1[dim]['good'] + pairs_hs2[dim]['good'],
            'bad': pairs_hs1[dim]['bad'] + pairs_hs2[dim]['bad']
        }
        
        print(f"\n  {dim.upper()}:")
        print(f"    HelpSteer  - Good: {len(pairs_hs1[dim]['good']):4d}, Bad: {len(pairs_hs1[dim]['bad']):4d}")
        print(f"    HelpSteer2 - Good: {len(pairs_hs2[dim]['good']):4d}, Bad: {len(pairs_hs2[dim]['bad']):4d}")
        print(f"    合并后     - Good: {len(merged_pairs[dim]['good']):4d}, Bad: {len(merged_pairs[dim]['bad']):4d}")
    
    # 保存数据
    print("\n" + "=" * 70)
    print("💾 保存整合数据")
    print("=" * 70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for dim in dimensions:
        good_file = output_path / f"{dim}_good_pairs.json"
        bad_file = output_path / f"{dim}_bad_pairs.json"
        
        with open(good_file, 'w', encoding='utf-8') as f:
            json.dump(merged_pairs[dim]['good'], f, indent=2, ensure_ascii=False)
        
        with open(bad_file, 'w', encoding='utf-8') as f:
            json.dump(merged_pairs[dim]['bad'], f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ {dim}: {good_file.name}, {bad_file.name}")
    
    # 统计摘要
    print("\n" + "=" * 70)
    print("📈 整合数据统计摘要")
    print("=" * 70)
    
    print("\n维度数据量对比:")
    print("-" * 70)
    print(f"{'维度':<15} {'原HelpSteer':<15} {'整合后':<15} {'增长':<10}")
    print("-" * 70)
    
    # 读取原始 ultra-extreme 数据进行对比
    original_dir = Path("data/helpsteer_ultra_extreme")
    for dim in dimensions:
        original_good = 0
        if (original_dir / f"{dim}_good_pairs.json").exists():
            with open(original_dir / f"{dim}_good_pairs.json", 'r') as f:
                original_good = len(json.load(f))
        
        merged_good = len(merged_pairs[dim]['good'])
        increase = merged_good - original_good
        increase_pct = (increase / original_good * 100) if original_good > 0 else 0
        
        print(f"{dim:<15} {original_good:<15} {merged_good:<15} +{increase} (+{increase_pct:.0f}%)")
    
    print("\n" + "=" * 70)
    print("✅ HelpSteer 数据整合完成!")
    print("=" * 70)
    print(f"\n输出目录: {output_dir}/")
    print("\n使用方法:")
    print("  bash train_extreme_single.sh helpfulness")
    print("  (训练脚本会自动使用 helpsteer_merged_ultra 目录下的数据)")


if __name__ == "__main__":
    merge_helpsteer_datasets()
