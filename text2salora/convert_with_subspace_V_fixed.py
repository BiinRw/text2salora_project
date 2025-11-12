#!/usr/bin/env python3
"""
将基于 V (子空间向量) 的训练结果转换为 SaLoRA 格式的 ABC.pt 文件

步骤:
1. 加载训练得到的子空间 V
2. 计算约束矩阵 C = I - V @ V^T
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
    """
    加载子空间 V 并构建约束矩阵 C = I - V @ V^T
    """
    # 先尝试加载 fused 文件
    fused_path = Path(subspace_dir) / f"{dimension}_layer{layer_id}_fused_subspace.pkl"
    layer_path = Path(subspace_dir) / f"{dimension}_layer{layer_id}_subspace.pkl"
    
    if fused_path.exists():
        data = torch.load(fused_path, map_location=device)
    elif layer_path.exists():
        data = torch.load(layer_path, map_location=device)
    else:
        return None, None
    
    V = data['V']  # shape: [feature_dim, subspace_dim]
    
    # 构建约束矩阵 C = I - V @ V^T
    feature_dim = V.shape[0]
    
    # 确保 V 在正确的设备上
    V = V.to(device)
    
    # 创建单位矩阵
    I = torch.eye(feature_dim, dtype=V.dtype, device=V.device)
    
    # 计算约束矩阵
    C = I - torch.matmul(V, V.transpose(0, 1))
    
    return C, V

def load_lora_adapter(base_model_name, adapter_path, device='cpu'):
    """加载 LoRA adapter"""
    print(f"\n🔧 加载 base model: {base_model_name}")
    print(f"   设备: {device}")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map=device if device != 'cpu' else 'cpu',
        trust_remote_code=True
    )
    
    print(f"📂 加载 LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    return model, base_model

def extract_lora_weights(model, base_model):
    """提取 LoRA A, B 矩阵和 base model 权重"""
    weight_dict = {}
    
    # 获取 LoRA 权重
    lora_state_dict = model.state_dict()
    base_state_dict = base_model.state_dict()
    
    for name, param in lora_state_dict.items():
        # 提取 LoRA A 和 B 矩阵
        if 'lora_A' in name or 'lora_B' in name:
            # 清理名称
            clean_name = name.replace('base_model.model.', '')
            clean_name = clean_name.replace('.default', '')
            weight_dict[clean_name] = param.cpu()
    
    # 提取 base model 权重
    for name, param in base_state_dict.items():
        if any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            if 'weight' in name:
                weight_dict[name] = param.cpu()
    
    return weight_dict

def add_constraint_from_subspace(weight_dict, subspace_dir, dimension, num_layers, device='cpu'):
    """
    为每一层添加约束矩阵 C 和子空间 V
    """
    print(f"\n📊 处理 {num_layers} 层的约束矩阵...")
    
    for layer_id in range(num_layers):
        print(f"  层 {layer_id}...", end=' ')
        
        C, V = load_and_build_constraint_matrix(subspace_dir, dimension, layer_id, device=device)
        
        if C is not None:
            # 为 q_proj, k_proj, v_proj 添加约束
            for proj_name in ['q_proj', 'k_proj', 'v_proj']:
                key_c = f'model.layers.{layer_id}.self_attn.{proj_name}.lora_C'
                key_v = f'model.layers.{layer_id}.self_attn.{proj_name}.V'
                
                weight_dict[key_c] = C.cpu()
                weight_dict[key_v] = V.cpu()
            
            print("✓")
        else:
            print("⚠️ 未找到子空间文件")
    
    return weight_dict

def compute_merged_weights(weight_dict, num_layers, lora_alpha=16, lora_rank=16):
    """
    计算合并权重: merged_weight = base_weight + (lora_B @ lora_A) * scaling
    """
    print(f"\n🔗 计算合并权重...")
    scaling = lora_alpha / lora_rank
    
    for layer_id in range(num_layers):
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            base_key = f'model.layers.{layer_id}.self_attn.{proj_name}.weight'
            lora_a_key = f'model.layers.{layer_id}.self_attn.{proj_name}.lora_A.weight'
            lora_b_key = f'model.layers.{layer_id}.self_attn.{proj_name}.lora_B.weight'
            merged_key = f'model.layers.{layer_id}.self_attn.{proj_name}.merged_weight'
            
            if base_key in weight_dict and lora_a_key in weight_dict and lora_b_key in weight_dict:
                base_weight = weight_dict[base_key]
                lora_a = weight_dict[lora_a_key]
                lora_b = weight_dict[lora_b_key]
                
                # 计算 LoRA 增量
                lora_delta = torch.matmul(lora_b, lora_a) * scaling
                
                # 合并权重
                merged_weight = base_weight + lora_delta
                weight_dict[merged_key] = merged_weight
                
                print(f"  层 {layer_id} {proj_name}: ✓")
    
    return weight_dict

def save_abc_file(weight_dict, output_path):
    """保存为 ABC.pt 文件"""
    # 确保输出目录存在
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 保存到: {output_path}")
    print(f"   包含 {len(weight_dict)} 个权重")
    
    torch.save(weight_dict, output_path)
    
    # 显示文件大小
    file_size = output_path.stat().st_size / (1024 ** 2)
    print(f"   文件大小: {file_size:.2f} MB")

def count_keys_by_type(weight_dict):
    """统计不同类型的键"""
    lora_a_count = sum(1 for k in weight_dict.keys() if 'lora_A' in k)
    lora_b_count = sum(1 for k in weight_dict.keys() if 'lora_B' in k)
    lora_c_count = sum(1 for k in weight_dict.keys() if 'lora_C' in k)
    v_count = sum(1 for k in weight_dict.keys() if '.V' in k)
    merged_count = sum(1 for k in weight_dict.keys() if 'merged_weight' in k)
    base_count = sum(1 for k in weight_dict.keys() if 'weight' in k and 'lora' not in k and 'merged' not in k)
    
    print(f"\n📊 统计:")
    print(f"   LoRA A 矩阵: {lora_a_count}")
    print(f"   LoRA B 矩阵: {lora_b_count}")
    print(f"   约束矩阵 C: {lora_c_count}")
    print(f"   子空间 V: {v_count}")
    print(f"   合并权重: {merged_count}")
    print(f"   基础权重: {base_count}")
    print(f"   总计: {len(weight_dict)}")

def main():
    parser = argparse.ArgumentParser(description='转换 V-based 训练结果为 ABC.pt 格式')
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
    parser.add_argument('--device', type=str, default='cpu',
                        help='设备 (cpu, cuda:0, cuda:1, ...)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 V-based → ABC.pt 转换工具")
    print("=" * 70)
    print(f"LoRA Adapter: {args.lora_adapter_path}")
    print(f"子空间目录: {args.subspace_dir}")
    print(f"偏好维度: {args.dimension}")
    print(f"Base Model: {args.base_model_name}")
    print(f"输出路径: {args.output_path}")
    print(f"层数: {args.num_layers}")
    print(f"设备: {args.device}")
    print("=" * 70)
    
    # 1. 加载 LoRA adapter
    model, base_model = load_lora_adapter(args.base_model_name, args.lora_adapter_path, args.device)
    
    # 2. 提取权重
    print("\n📦 提取 LoRA 权重...")
    weight_dict = extract_lora_weights(model, base_model)
    
    # 3. 添加约束矩阵
    weight_dict = add_constraint_from_subspace(
        weight_dict, 
        args.subspace_dir, 
        args.dimension, 
        args.num_layers,
        args.device
    )
    
    # 4. 计算合并权重
    weight_dict = compute_merged_weights(weight_dict, args.num_layers)
    
    # 5. 统计
    count_keys_by_type(weight_dict)
    
    # 6. 保存
    save_abc_file(weight_dict, args.output_path)
    
    # 清理内存
    del model
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print("\n✅ 转换完成!")

if __name__ == "__main__":
    main()
