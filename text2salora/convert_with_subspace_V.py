#!/usr/bin/env python3
"""
从子空间向量 V 构建约束矩阵 C，然后转换为 SaLoRA ABC.pt 格式

正确流程:
1. 加载子空间向量 V (从 preference_subspace/saved_subspaces/)
2. 构建约束矩阵 C = I - V @ V^T  
3. 加载 LoRA adapter (A, B 矩阵)
4. 组合成 ABC.pt 文件
"""

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
import argparse
import os
from pathlib import Path

def load_and_build_constraint_matrix(subspace_dir, dimension, layer_id, device='cpu'):
    """从子空间向量 V 构建约束矩阵 C"""
    
    # 尝试加载 fused 子空间或层级子空间
    fused_path = Path(subspace_dir) / f"{dimension}_fused_subspace.pt"
    layer_path = Path(subspace_dir) / f"{dimension}_layer{layer_id}_subspace.pt"
    
    if fused_path.exists():
        print(f"  ✓ Layer {layer_id}: 使用 fused 子空间")
        data = torch.load(fused_path, map_location=device)
    elif layer_path.exists():
        print(f"  ✓ Layer {layer_id}: 使用层级子空间")
        data = torch.load(layer_path, map_location=device)
    else:
        raise FileNotFoundError(
            f"找不到 layer {layer_id} 的子空间文件:\n"
            f"  - {fused_path}\n"
            f"  - {layer_path}"
        )
    
    # 提取子空间向量 V
    V = data['V']  # shape: [feature_dim, subspace_rank]
    
    # 构建约束矩阵 C = I - V @ V^T
    feature_dim = V.shape[0]
    I = torch.eye(feature_dim, dtype=V.dtype, device=V.device)
    C = I - V @ V.T
    
    return C, V

def load_lora_adapter(base_model_name, adapter_path):
    """加载 LoRA adapter"""
    print(f"\n�� 加载 base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )
    
    print(f"📂 加载 LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    return model, base_model

def extract_lora_weights(model, base_model):
    """提取 LoRA A, B 矩阵和 base model 权重"""
    weight_list = {}
    
    print("\n🔍 提取 LoRA 权重...")
    
    # 提取 LoRA A, B 矩阵
    for name, module in model.named_modules():
        if 'lora_A' in name and 'default' in name and hasattr(module, 'weight'):
            if 'layers.' in name and ('q_proj' in name or 'v_proj' in name):
                parts = name.split('.')
                layer_id, proj_type = None, None
                
                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts):
                        layer_id = parts[i + 1]
                    if part in ['q_proj', 'v_proj']:
                        proj_type = part
                
                if layer_id and proj_type:
                    key = f"{proj_type}_{layer_id}lora_A"
                    weight_list[key] = module.weight.data.cpu().clone()
        
        if 'lora_B' in name and 'default' in name and hasattr(module, 'weight'):
            if 'layers.' in name and ('q_proj' in name or 'v_proj' in name):
                parts = name.split('.')
                layer_id, proj_type = None, None
                
                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts):
                        layer_id = parts[i + 1]
                    if part in ['q_proj', 'v_proj']:
                        proj_type = part
                
                if layer_id and proj_type:
                    key = f"{proj_type}_{layer_id}lora_B"
                    weight_list[key] = module.weight.data.cpu().clone()
    
    print(f"✓ 提取到 {len([k for k in weight_list if 'lora_A' in k])} 个 LoRA 层")
    
    
    # 计算 merged weights (base + LoRA delta)
    print("\n🔍 计算 merged weights (base + LoRA)...")
    
    # 获取 LoRA scaling factor
    try:
        lora_config = model.peft_config['default']
        lora_alpha = lora_config.lora_alpha
        lora_r = lora_config.r
        scaling = lora_alpha / lora_r
        print(f"   LoRA scaling: {scaling} (alpha={lora_alpha}, r={lora_r})")
    except:
        scaling = 1.0
        print(f"   使用默认 scaling: {scaling}")
    
    for name, module in base_model.named_modules():
        if 'layers.' in name and ('q_proj' in name or 'v_proj' in name):
            if hasattr(module, 'weight') and 'lora' not in name.lower():
                if name.endswith('q_proj') or name.endswith('v_proj'):
                    parts = name.split('.')
                    layer_id, proj_type = None, None
                    
                    for i, part in enumerate(parts):
                        if part == 'layers' and i + 1 < len(parts):
                            layer_id = parts[i + 1]
                        if part in ['q_proj', 'v_proj']:
                            proj_type = part
                    
                    if layer_id and proj_type:
                        # Base weight
                        base_weight = module.weight.data.cpu().clone()
                        
                        # 查找对应的 LoRA A, B
                        key_A = f"{proj_type}_{layer_id}lora_A"
                        key_B = f"{proj_type}_{layer_id}lora_B"
                        
                        if key_A in weight_list and key_B in weight_list:
                            # 计算 LoRA delta: B @ A * scaling
                            lora_A = weight_list[key_A]
                            lora_B = weight_list[key_B]
                            lora_delta = (lora_B @ lora_A) * scaling
                            
                            # Merged weight = base + delta
                            merged_weight = base_weight + lora_delta
                            
                            key = f"{proj_type}_{layer_id}weight"
                            weight_list[key] = merged_weight
                        else:
                            # 没有 LoRA,使用原始 base weight
                            key = f"{proj_type}_{layer_id}weight"
                            weight_list[key] = base_weight
    
    num_weights = len([k for k in weight_list if 'weight' in k])
    print(f"✓ 生成 {num_weights} 个 merged weights")
    return weight_list

def add_constraint_from_subspace(weight_list, subspace_dir, dimension, num_layers=28):
    """从子空间向量 V 构建约束矩阵 C 并添加到 weight_list"""
    print(f"\n🔧 从子空间构建约束矩阵 (维度: {dimension})...")
    print(f"   子空间目录: {subspace_dir}")
    
    added_count = 0
    for layer_id in range(num_layers):
        try:
            # 为每层构建约束矩阵
            C, V = load_and_build_constraint_matrix(subspace_dir, dimension, layer_id)
            
            for proj_type in ['q_proj', 'v_proj']:
                key_prefix = f"{proj_type}_{layer_id}"
                
                # 检查该层是否有 LoRA B
                if f"{key_prefix}lora_B" in weight_list:
                    out_dim = weight_list[f"{key_prefix}lora_B"].shape[0]
                    
                    # 裁剪到匹配维度
                    C_block = C[:out_dim, :out_dim].clone()
                    V_block = V[:out_dim, :].clone()
                    
                    weight_list[f"{key_prefix}lora_C"] = C_block
                    weight_list[f"{key_prefix}_V"] = V_block
                    
                    added_count += 1
        
        except FileNotFoundError as e:
            print(f"  ⚠️ Layer {layer_id}: 未找到子空间文件，跳过")
            continue
    
    print(f"✓ 成功添加 {added_count} 个约束矩阵")
    
    # 添加元数据
    weight_list['divide_num'] = 2
    
    return weight_list

def save_abc_file(weight_list, output_path):
    """保存为 ABC.pt 文件"""
    # 自动创建父目录(如果不存在)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 保存 ABC.pt 文件: {output_path}")
    torch.save(weight_list, str(output_path))
    
    # 统计
    stats = {
        'total': len(weight_list),
        'lora_A': sum(1 for k in weight_list if 'lora_A' in k),
        'lora_B': sum(1 for k in weight_list if 'lora_B' in k),
        'lora_C': sum(1 for k in weight_list if 'lora_C' in k),
        'V': sum(1 for k in weight_list if '_V' in k and 'lora' not in k),
        'weight': sum(1 for k in weight_list if 'weight' in k)
    }
    
    print(f"✓ 保存成功!")
    print(f"  - 总键数: {stats['total']}")
    print(f"  - lora_A: {stats['lora_A']}")
    print(f"  - lora_B: {stats['lora_B']}")
    print(f"  - lora_C: {stats['lora_C']}")
    print(f"  - V (subspace): {stats['V']}")
    print(f"  - weight: {stats['weight']}")
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  - 文件大小: {file_size_mb:.2f} MB")

def main():
    parser = argparse.ArgumentParser(description="从子空间 V 构建 C 并转换为 ABC.pt")
    parser.add_argument('--lora_adapter_path', type=str, required=True,
                        help='LoRA adapter 路径')
    parser.add_argument('--subspace_dir', type=str, required=True,
                        help='子空间文件目录')
    parser.add_argument('--dimension', type=str, default='safety',
                        help='偏好维度 (safety, helpfulness, etc.)')
    parser.add_argument('--base_model_name', type=str, required=True,
                        help='Base model 名称')
    parser.add_argument('--output_path', type=str, required=True,
                        help='输出 ABC.pt 文件路径')
    parser.add_argument('--num_layers', type=int, default=28,
                        help='模型层数')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🔄 Text2SaLoRA → SaLoRA ABC.pt 转换器 (从子空间 V 构建 C)")
    print("="*70)
    print(f"\n📋 配置:")
    print(f"  - LoRA adapter: {args.lora_adapter_path}")
    print(f"  - 子空间目录: {args.subspace_dir}")
    print(f"  - 偏好维度: {args.dimension}")
    print(f"  - Base model: {args.base_model_name}")
    print(f"  - 输出文件: {args.output_path}")
    print(f"  - 层数: {args.num_layers}")
    print()
    
    # 1. 加载 LoRA adapter
    model, base_model = load_lora_adapter(args.base_model_name, args.lora_adapter_path)
    
    # 2. 提取 LoRA A, B 矩阵和 base weights
    weight_list = extract_lora_weights(model, base_model)
    
    # 3. 从子空间 V 构建约束矩阵 C
    weight_list = add_constraint_from_subspace(
        weight_list, 
        args.subspace_dir, 
        args.dimension, 
        args.num_layers
    )
    
    # 4. 保存为 ABC.pt
    save_abc_file(weight_list, args.output_path)
    
    print("\n" + "="*70)
    print("✅ 转换完成!")
    print(f"\n💡 约束矩阵 C 是从子空间向量 V 构建的:")
    print(f"   C = I - V @ V^T")
    print("="*70)

if __name__ == "__main__":
    main()
