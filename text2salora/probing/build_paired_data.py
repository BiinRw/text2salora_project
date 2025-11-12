"""
构建正确的 Prompt-Response 配对数据
根据 SaLoRA 论文要求构建
"""

from datasets import load_dataset
import json
import os
from tqdm import tqdm
import random


def build_correct_safety_data(output_dir='data/safety_paired', max_samples=500):
    """
    构建正确的安全性配对数据
    
    根据SaLoRA论文:
    - Safe Scenario: Benign Prompt + Safe Response
    - Harmful Scenario: Harmful Prompt + Safe Response (拒绝回答)
    """
    print("=" * 70)
    print("🔒 构建正确的 Prompt-Response 配对数据 (SaLoRA方法)")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n📥 加载 PKU-SafeRLHF 数据集...")
    cache_dir = "/var/models/huggingface/hub"
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train", cache_dir=cache_dir)
    print(f"✅ 加载成功,共 {len(dataset)} 条样本")
    
    safe_pairs = []      # 安全场景: benign prompt + safe response
    harmful_pairs = []   # 危险场景: harmful prompt + safe response
    
    print("\n🔍 处理数据...")
    for item in tqdm(dataset):
        prompt = item['prompt']
        response_0 = item['response_0']
        response_1 = item['response_1']
        is_r0_safe = item.get('is_response_0_safe', True)
        is_r1_safe = item.get('is_response_1_safe', True)
        
        # 场景1: 安全场景 (benign prompt + safe response)
        # 如果response是安全的,说明prompt是benign的
        if is_r0_safe:
            safe_pairs.append({
                'prompt': prompt,
                'response': response_0,
                'label': 0,  # safe scenario
                'scenario': 'safe',
                'source': 'PKU-SafeRLHF'
            })
        
        if is_r1_safe:
            safe_pairs.append({
                'prompt': prompt,
                'response': response_1,
                'label': 0,  # safe scenario
                'scenario': 'safe',
                'source': 'PKU-SafeRLHF'
            })
        
        # 场景2: 危险场景 (harmful prompt + safe response/拒绝回答)
        # 如果response是不安全的,说明prompt是harmful的
        # 我们需要找到对harmful prompt的安全拒绝回答
        if not is_r0_safe:
            # 这个prompt是harmful的,我们需要paired的safe response
            # 检查是否有safer_response
            safer_id = item.get('safer_response_id', -1)
            if safer_id == 1 and is_r1_safe:
                harmful_pairs.append({
                    'prompt': prompt,
                    'response': response_1,  # 使用更安全的response
                    'label': 1,  # harmful scenario
                    'scenario': 'harmful',
                    'harm_category': item.get('response_0_harm_category', {}),
                    'source': 'PKU-SafeRLHF'
                })
        
        if not is_r1_safe:
            safer_id = item.get('safer_response_id', -1)
            if safer_id == 0 and is_r0_safe:
                harmful_pairs.append({
                    'prompt': prompt,
                    'response': response_0,  # 使用更安全的response
                    'label': 1,  # harmful scenario
                    'scenario': 'harmful',
                    'harm_category': item.get('response_1_harm_category', {}),
                    'source': 'PKU-SafeRLHF'
                })
    
    # 去重
    print("\n🔄 去重...")
    print(f"   去重前: safe={len(safe_pairs)}, harmful={len(harmful_pairs)}")
    
    # 基于 (prompt, response) 组合去重
    safe_pairs = list({(p['prompt'], p['response']): p for p in safe_pairs}.values())
    harmful_pairs = list({(p['prompt'], p['response']): p for p in harmful_pairs}.values())
    
    print(f"   去重后: safe={len(safe_pairs)}, harmful={len(harmful_pairs)}")
    
    # 采样
    if len(safe_pairs) > max_samples:
        safe_pairs = random.sample(safe_pairs, max_samples)
    if len(harmful_pairs) > max_samples:
        harmful_pairs = random.sample(harmful_pairs, max_samples)
    
    # 保存
    safe_path = os.path.join(output_dir, 'safe_pairs.json')
    harmful_path = os.path.join(output_dir, 'harmful_pairs.json')
    
    with open(safe_path, 'w', encoding='utf-8') as f:
        json.dump(safe_pairs, f, indent=2, ensure_ascii=False)
    
    with open(harmful_path, 'w', encoding='utf-8') as f:
        json.dump(harmful_pairs, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 保存成功:")
    print(f"   {safe_path}: {len(safe_pairs)} 条")
    print(f"   {harmful_path}: {len(harmful_pairs)} 条")
    
    # 显示样本
    print(f"\n📋 安全场景示例 (Benign Prompt + Safe Response):")
    for i, item in enumerate(safe_pairs[:2], 1):
        print(f"\n   {i}. Prompt: {item['prompt'][:80]}...")
        print(f"      Response: {item['response'][:80]}...")
        print(f"      Scenario: {item['scenario']}")
    
    print(f"\n📋 危险场景示例 (Harmful Prompt + Safe Response/拒绝):")
    for i, item in enumerate(harmful_pairs[:2], 1):
        print(f"\n   {i}. Prompt: {item['prompt'][:80]}...")
        print(f"      Response: {item['response'][:80]}...")
        print(f"      Scenario: {item['scenario']}")
    
    return safe_path, harmful_path


def main():
    print("\n🚀 构建 SaLoRA 论文要求的配对数据")
    print("=" * 70)
    print("\n📖 论文要求:")
    print("   - Safe Scenario: Benign Prompt + Safe Response")
    print("   - Harmful Scenario: Harmful Prompt + Safe Response")
    print("\n   探针的目标: 区分模型处理这两种场景时的内部表征差异")
    print("=" * 70)
    
    try:
        safe_path, harmful_path = build_correct_safety_data(max_samples=500)
        print(f"\n✅ 数据构建完成!")
    except Exception as e:
        print(f"\n❌ 数据构建失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("📝 使用说明:")
    print("   这些数据是 prompt-response 配对的")
    print("   训练时需要将 prompt + response 拼接后输入模型")
    print("   探针基于完整对话的attention head激活值进行训练")
    print("=" * 70)


if __name__ == '__main__':
    main()
