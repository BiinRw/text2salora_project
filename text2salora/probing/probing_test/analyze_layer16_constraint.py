"""
分析只约束第16层的实验效果
对比：base、all_layers、layers_0-8、layers_8-16、layer_16、layers_17-27
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# 定义实验组
experiments = {
    'layers_0-8': 'Qwen2.5-1.5B-Instruct+safety-lora_wo_g_r16_a32-ep1-svd_rank16-salora_0_8-lr_5e-5',
    'layers_8-16': 'Qwen2.5-1.5B-Instruct+safety-lora_wo_g_r16_a32-ep1-svd_rank16-salora_8_16-lr_5e-5',
    'layer_16_only': 'Qwen2.5-1.5B-Instruct+safety-lora_wo_g_r16_a32-ep1-svd_rank16-salora_16-lr_5e-5',
    'layers_17-27': 'Qwen2.5-1.5B-Instruct+safety-lora_wo_g_r16_a32-ep1-svd_rank16-salora_17_27-lr_5e-5',
    'all_layers': 'Qwen2.5-1.5B-Instruct+safety-lora_wo_g_r16_a32-ep1-svd_rank16_salora_hard-lr_5e-5',
}

results_dir = Path('results/batch_multi_position_test')
dimensions = ['safety', 'helpfulness', 'correctness', 'coherence']
position = 'assistant_last'

print("="*100)
print("只约束第16层 vs 其他层约束策略 - 对比分析")
print("="*100)

# 收集所有实验的数据
all_data = {}

for exp_name, model_name in experiments.items():
    exp_dir = results_dir / model_name
    exp_data = {}
    
    for dim in dimensions:
        dim_dir = exp_dir / f'{dim}_lora'
        
        if dim_dir.exists():
            # 找到最后一个checkpoint
            json_files = list(dim_dir.glob(f'*_ckpt*_{dim}_multi_position.json'))
            if json_files:
                # 按checkpoint编号排序
                json_files.sort(key=lambda x: int(str(x.stem).split('_ckpt')[1].split('_')[0]))
                last_file = json_files[-1]
                
                with open(last_file, 'r') as f:
                    data = json.load(f)
                    
                    # 提取assistant_last位置的数据
                    if position in data:
                        exp_data[dim] = data[position]
    
    all_data[exp_name] = exp_data
    print(f"✓ 加载实验: {exp_name}")

# 按层提取准确率
def extract_layer_accuracies(exp_data, dim):
    """从实验数据中提取每层的平均准确率"""
    layer_accs = {}
    
    if dim in exp_data:
        for key, value in exp_data[dim].items():
            if key.startswith('layer-'):
                parts = key.split('-')
                layer_id = int(parts[1])
                
                if layer_id not in layer_accs:
                    layer_accs[layer_id] = []
                
                layer_accs[layer_id].append(value['accuracy'])
    
    # 计算每层的平均准确率
    layer_avg = {}
    for layer_id, accs in layer_accs.items():
        layer_avg[layer_id] = np.mean(accs)
    
    return layer_avg

print("\n" + "="*100)
print("1. Safety 维度：各实验组的逐层探针准确率对比")
print("="*100)

# 提取所有实验的safety数据
safety_data = {}
for exp_name in experiments.keys():
    safety_data[exp_name] = extract_layer_accuracies(all_data[exp_name], 'safety')

# 打印表格
print(f"\n{'Layer':>6}", end="")
for exp_name in experiments.keys():
    print(f" | {exp_name:>14}", end="")
print()
print("-"*100)

for layer in range(28):
    print(f"{layer:>6}", end="")
    
    for exp_name in experiments.keys():
        if layer in safety_data[exp_name]:
            acc = safety_data[exp_name][layer]
            print(f" | {acc:>14.3f}", end="")
        else:
            print(f" | {'N/A':>14}", end="")
    print()

print("\n" + "="*100)
print("2. 关键层（第16层）在所有维度上的对比")
print("="*100)

print(f"\n{'实验组':<16}", end="")
for dim in dimensions:
    print(f" | {dim:>12}", end="")
print(" | 平均")
print("-"*100)

for exp_name in experiments.keys():
    print(f"{exp_name:<16}", end="")
    
    layer16_accs = []
    for dim in dimensions:
        layer_data = extract_layer_accuracies(all_data[exp_name], dim)
        if 16 in layer_data:
            acc = layer_data[16]
            layer16_accs.append(acc)
            print(f" | {acc:>12.3f}", end="")
        else:
            print(f" | {'N/A':>12}", end="")
    
    if layer16_accs:
        avg = np.mean(layer16_accs)
        print(f" | {avg:.3f}")
    else:
        print(" | N/A")

print("\n" + "="*100)
print("3. 各实验组在Safety维度的统计信息")
print("="*100)

for exp_name in experiments.keys():
    layer_data = safety_data[exp_name]
    
    if layer_data:
        accs = list(layer_data.values())
        avg = np.mean(accs)
        std = np.std(accs)
        min_acc = np.min(accs)
        max_acc = np.max(accs)
        
        # 第16层
        layer16 = layer_data.get(16, 0)
        
        print(f"\n{exp_name}:")
        print(f"  平均准确率: {avg:.3f} ± {std:.3f}")
        print(f"  范围: [{min_acc:.3f}, {max_acc:.3f}]")
        print(f"  第16层: {layer16:.3f} (相对平均: {layer16-avg:+.3f})")

print("\n" + "="*100)
print("4. 关键发现")
print("="*100)

# 计算每个实验组第16层的准确率
layer16_safety = {}
for exp_name in experiments.keys():
    layer_data = safety_data[exp_name]
    if 16 in layer_data:
        layer16_safety[exp_name] = layer_data[16]

print("\n📌 第16层在Safety维度的准确率对比：")
print("-"*60)
for exp_name, acc in sorted(layer16_safety.items(), key=lambda x: x[1], reverse=True):
    print(f"{exp_name:>16}: {acc:.3f}")

# 计算差异
print("\n📌 相对差异分析：")
print("-"*60)
if 'layer_16_only' in layer16_safety:
    layer16_only_acc = layer16_safety['layer_16_only']
    
    print(f"\n以 'layer_16_only' 为基准 ({layer16_only_acc:.3f})：")
    for exp_name, acc in layer16_safety.items():
        if exp_name != 'layer_16_only':
            diff = acc - layer16_only_acc
            print(f"{exp_name:>16}: {diff:+.3f} ({acc:.3f})")

# 计算层级变异系数
print("\n�� 各实验组的层间变异性（变异系数 CV）：")
print("-"*60)
for exp_name in experiments.keys():
    layer_data = safety_data[exp_name]
    if layer_data:
        accs = list(layer_data.values())
        avg = np.mean(accs)
        std = np.std(accs)
        cv = (std / avg) * 100 if avg > 0 else 0
        print(f"{exp_name:>16}: CV = {cv:>5.2f}% (std={std:.3f}, mean={avg:.3f})")

print("\n" + "="*100)

