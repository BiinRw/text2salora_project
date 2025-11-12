"""
整合 HelpSteer 和 HelpSteer2 数据集 (配对策略)

正确策略:
1. 对于每个维度,按 prompt 分组
2. 对于同一个 prompt:
   - 找到所有 4分 的响应 (Good)
   - 找到所有 0分 的响应 (Bad)
   - 如果两者都存在,则配对
3. 这样确保 Good 和 Bad 是对同一个问题的不同质量响应
"""

import json
from datasets import load_dataset
from pathlib import Path
from collections import defaultdict, Counter

def build_paired_extreme_data(dataset, dataset_name):
    """
    从数据集中构建配对的极端数据
    
    策略: 对于同一个 prompt,找到 4分响应(Good) 和 0分响应(Bad) 配对
    
    Returns:
        dict: {dimension: {'pairs': [(good_item, bad_item), ...]}}
    """
    dimensions = ['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity']
    results = {dim: {'pairs': []} for dim in dimensions}
    
    for dimension in dimensions:
        print(f"\n  处理 {dimension}...")
        
        # 按 prompt 分组
        prompt_groups = defaultdict(lambda: {'score_0': [], 'score_4': []})
        
        for item in dataset:
            score = item.get(dimension)
            if score is None:
                continue
            
            prompt = item['prompt']
            
            if score == 0.0 or score == 0:
                prompt_groups[prompt]['score_0'].append(item)
            elif score == 4.0 or score == 4:
                prompt_groups[prompt]['score_4'].append(item)
        
        # 统计
        total_prompts = len(prompt_groups)
        prompts_with_0 = sum(1 for g in prompt_groups.values() if g['score_0'])
        prompts_with_4 = sum(1 for g in prompt_groups.values() if g['score_4'])
        prompts_with_both = sum(1 for g in prompt_groups.values() if g['score_0'] and g['score_4'])
        
        print(f"    总 prompts: {total_prompts}")
        print(f"    有0分的: {prompts_with_0}")
        print(f"    有4分的: {prompts_with_4}")
        print(f"    同时有0和4分的: {prompts_with_both}")
        
        # 创建配对
        pair_count = 0
        for prompt, group in prompt_groups.items():
            score_0_items = group['score_0']
            score_4_items = group['score_4']
            
            if not score_0_items or not score_4_items:
                continue
            
            # 为每个 0分响应 和 4分响应 创建配对
            for bad_item in score_0_items:
                for good_item in score_4_items:
                    results[dimension]['pairs'].append({
                        'prompt': prompt,
                        'good_response': good_item['response'],
                        'bad_response': bad_item['response'],
                        'source': dataset_name,
                        'dimension': dimension
                    })
                    pair_count += 1
        
        print(f"    ✅ 创建配对: {pair_count} 对")
    
    return results


def merge_helpsteer_paired(output_dir="data/helpsteer_merged_paired"):
    """
    整合 HelpSteer 和 HelpSteer2 数据集 (配对策略)
    """
    print("=" * 70)
    print("📥 加载 HelpSteer 和 HelpSteer2 数据集")
    print("=" * 70)
    
    # 加载数据集
    print("\n正在加载 HelpSteer...")
    helpsteer1 = load_dataset("nvidia/HelpSteer", split="train")
    print(f"✅ HelpSteer: {len(helpsteer1)} 条")
    
    print("正在加载 HelpSteer2...")
    helpsteer2 = load_dataset("nvidia/HelpSteer2", split="train")
    print(f"✅ HelpSteer2: {len(helpsteer2)} 条")
    
    print(f"\n总数据量: {len(helpsteer1) + len(helpsteer2)} 条")
    
    # 构建配对数据
    print("\n" + "=" * 70)
    print("��️  构建配对数据 (同一 prompt 的 0分 vs 4分)")
    print("=" * 70)
    
    print("\n--- HelpSteer ---")
    pairs_hs1 = build_paired_extreme_data(helpsteer1, "helpsteer")
    
    print("\n--- HelpSteer2 ---")
    pairs_hs2 = build_paired_extreme_data(helpsteer2, "helpsteer2")
    
    # 合并数据
    print("\n" + "=" * 70)
    print("🔗 合并配对数据")
    print("=" * 70)
    
    dimensions = ['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity']
    merged_pairs = {}
    
    for dim in dimensions:
        merged_pairs[dim] = pairs_hs1[dim]['pairs'] + pairs_hs2[dim]['pairs']
        
        print(f"\n  {dim.upper()}:")
        print(f"    HelpSteer:  {len(pairs_hs1[dim]['pairs']):5d} 配对")
        print(f"    HelpSteer2: {len(pairs_hs2[dim]['pairs']):5d} 配对")
        print(f"    合并后:     {len(merged_pairs[dim]):5d} 配对")
    
    # 转换为 good/bad pairs 格式
    print("\n" + "=" * 70)
    print("📦 转换为训练格式")
    print("=" * 70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for dim in dimensions:
        good_pairs = []
        bad_pairs = []
        
        for pair in merged_pairs[dim]:
            # Good pair
            good_pairs.append({
                'prompt': pair['prompt'],
                'response': pair['good_response'],
                'source': pair['source'],
                'original_score': 4,
                'label': 0
            })
            
            # Bad pair
            bad_pairs.append({
                'prompt': pair['prompt'],
                'response': pair['bad_response'],
                'source': pair['source'],
                'original_score': 0,
                'label': 1
            })
        
        # 保存
        good_file = output_path / f"{dim}_good_pairs.json"
        bad_file = output_path / f"{dim}_bad_pairs.json"
        
        with open(good_file, 'w', encoding='utf-8') as f:
            json.dump(good_pairs, f, indent=2, ensure_ascii=False)
        
        with open(bad_file, 'w', encoding='utf-8') as f:
            json.dump(bad_pairs, f, indent=2, ensure_ascii=False)
        
        # 验证配对
        assert len(good_pairs) == len(bad_pairs), f"{dim}: Good/Bad 数量不匹配!"
        
        # 验证 prompt 匹配
        good_prompts = [p['prompt'] for p in good_pairs]
        bad_prompts = [p['prompt'] for p in bad_pairs]
        assert good_prompts == bad_prompts, f"{dim}: Prompt 顺序不匹配!"
        
        print(f"  ✅ {dim}: {len(good_pairs)} 配对 (100% 匹配)")
    
    # 统计摘要
    print("\n" + "=" * 70)
    print("📈 最终统计摘要")
    print("=" * 70)
    
    print("\n配对数据量:")
    print("-" * 70)
    print(f"{'维度':<15} {'配对数':<10} {'说明':<30}")
    print("-" * 70)
    
    for dim in dimensions:
        pair_count = len(merged_pairs[dim])
        desc = "同一prompt的0分vs4分响应"
        print(f"{dim:<15} {pair_count:<10,} {desc}")
    
    print("\n" + "=" * 70)
    print("✅ HelpSteer 配对数据整合完成!")
    print("=" * 70)
    print(f"\n输出目录: {output_dir}/")
    print("\n特点:")
    print("  ✅ 每个 Good 样本都有对应的 Bad 样本")
    print("  ✅ 都是对同一个 prompt 的不同质量响应")
    print("  ✅ 最大化差异 (0分 vs 4分)")
    print("\n使用方法:")
    print("  bash train_extreme_single.sh helpfulness --merged")


if __name__ == "__main__":
    merge_helpsteer_paired()
