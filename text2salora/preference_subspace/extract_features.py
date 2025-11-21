"""
偏好子空间特征提取 (支持分投影层提取)
提取 chosen/rejected 样本的激活值,计算特征差分
v2: 支持为每个投影层(q_proj, k_proj, v_proj, o_proj, up_proj, down_proj)分别提取子空间
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path


class ActivationExtractor:
    """提取模型激活值的工具类 (支持指定投影层)"""
    
    def __init__(self, model, tokenizer, projection_type='q_proj', device='cuda:0'):
        """
        Args:
            model: 预训练模型
            tokenizer: 分词器
            projection_type: 要提取的投影层类型 (q_proj/k_proj/v_proj/o_proj/up_proj/down_proj)
            device: 设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.projection_type = projection_type  # 新增: 投影层类型
        
        # 自动检测模型层的正确路径
        self.model_layers = self._get_model_layers()
        self.num_layers = model.config.num_hidden_layers
        
        self.activations = {}
        self.hooks = []
        
        print(f"   ✅ 提取投影层: {projection_type}")
    
    def _get_model_layers(self):
        """自动检测模型层的正确访问路径"""
        possible_paths = [
            ('model.model.layers', lambda m: m.model.layers),
            ('model.model.model.layers', lambda m: m.model.model.layers),
            ('model.base_model.model.model.layers', lambda m: m.base_model.model.model.layers),
        ]
        
        for path_name, path_fn in possible_paths:
            try:
                layers = path_fn(self.model)
                if layers is not None and len(layers) > 0:
                    print(f"   ✅ 检测到模型层路径: {path_name}")
                    return layers
            except (AttributeError, TypeError):
                continue
        
        raise RuntimeError("无法找到模型的层结构!")
    
    def _get_activation_hook(self, layer_id):
        """创建hook函数来捕获激活值"""
        def hook(module, input, output):
            key = f"layer-{layer_id}"
            if key not in self.activations:
                self.activations[key] = []
            # 提取最后一个token的激活值
            self.activations[key].append(output[:, -1, :].detach().cpu())
        return hook
    
    def register_hooks(self):
        """注册hooks到所有层的指定投影层
        
        投影层位置:
        - q_proj, k_proj, v_proj, o_proj: 在 layer.self_attn 中
        - up_proj, down_proj, gate_proj: 在 layer.mlp 中
        """
        print(f"🔧 注册 {self.projection_type} hooks...")
        
        for layer_id in range(self.num_layers):
            layer = self.model_layers[layer_id]
            
            # 根据投影类型选择要hook的模块
            if self.projection_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                # 注意力层投影
                try:
                    module = getattr(layer.self_attn, self.projection_type)
                except AttributeError:
                    raise ValueError(f"模型层没有 self_attn.{self.projection_type} 属性!")
                    
            elif self.projection_type in ['up_proj', 'down_proj', 'gate_proj']:
                # MLP层投影 (gate_proj是某些模型的额外投影)
                try:
                    module = getattr(layer.mlp, self.projection_type)
                except AttributeError:
                    raise ValueError(f"模型层没有 mlp.{self.projection_type} 属性!")
            else:
                raise ValueError(f"不支持的投影类型: {self.projection_type}. "
                               f"支持的类型: q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj")
            
            # 注册hook
            hook = module.register_forward_hook(self._get_activation_hook(layer_id))
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """移除所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def format_conversation(self, prompt, response):
        """格式化对话为模型输入"""
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        
        try:
            text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
        except:
            text = f"User: {prompt}\nAssistant: {response}"
        
        return text
    
    def extract_activations(self, data_samples, max_samples=None):
        """提取数据样本的激活值
        
        Args:
            data_samples: 数据样本列表
            max_samples: 最大样本数
            
        Returns:
            dict: {layer_id: numpy_array} 每层的激活值
        """
        self.activations = {}
        self.register_hooks()
        
        if max_samples:
            data_samples = data_samples[:max_samples]
        
        print(f"   提取 {len(data_samples)} 个样本的激活值...")
        self.model.eval()
        
        with torch.no_grad():
            for sample in tqdm(data_samples, desc=f"   提取 {self.projection_type}"):
                text = self.format_conversation(sample['prompt'], sample['response'])
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=2048
                ).to(self.device)
                
                # 前向传播
                _ = self.model(**inputs)
        
        # 移除hooks并返回结果
        self.remove_hooks()
        
        # 转换为numpy数组
        activations_np = {}
        for key, values in self.activations.items():
            activations_np[key] = torch.cat(values, dim=0).numpy()
        
        return activations_np


def load_preference_data(data_dir: str, dimension: str) -> Tuple[List, List]:
    """加载偏好数据对
    
    Args:
        data_dir: 数据目录
        dimension: 维度名称 (safety, helpfulness, correctness, coherence)
        
    Returns:
        chosen_samples, rejected_samples
    """
    data_dir = Path(data_dir)
    
    if dimension == 'safety':
        # Safety 维度: safe=chosen, harmful=rejected
        chosen_file = data_dir / 'safety_paired' / 'safe_pairs.json'
        rejected_file = data_dir / 'safety_paired' / 'harmful_pairs.json'
    else:
        # 其他维度: good=chosen, bad=rejected
        chosen_file = data_dir / 'helpsteer_paired' / f'{dimension}_good_pairs.json'
        rejected_file = data_dir / 'helpsteer_paired' / f'{dimension}_bad_pairs.json'
    
    print(f"\n📂 加载数据:")
    print(f"   Chosen:   {chosen_file}")
    print(f"   Rejected: {rejected_file}")
    
    with open(chosen_file, 'r') as f:
        chosen_samples = json.load(f)
    
    with open(rejected_file, 'r') as f:
        rejected_samples = json.load(f)
    
    print(f"   ✅ Chosen: {len(chosen_samples)} 样本")
    print(f"   ✅ Rejected: {len(rejected_samples)} 样本")
    
    return chosen_samples, rejected_samples


def extract_and_save_features(
    model_name: str,
    data_dir: str,
    dimension: str,
    projection_type: str,  # 新增: 投影层类型
    output_dir: str,
    max_samples: int = None,
    device: str = 'cuda:0'
):
    """提取并保存特征差分
    
    Args:
        model_name: 模型名称或路径
        data_dir: 数据目录
        dimension: 偏好维度
        projection_type: 投影层类型 (q_proj/k_proj/v_proj/o_proj/up_proj/down_proj)
        output_dir: 输出目录
        max_samples: 最大样本数
        device: 设备
    """
    print("=" * 80)
    print(f"🚀 开始提取 {dimension} 维度 - {projection_type} 投影层的特征")
    print("=" * 80)
    
    # 1. 加载模型
    print(f"\n📦 加载模型: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 2. 加载数据
    chosen_samples, rejected_samples = load_preference_data(data_dir, dimension)
    
    # 限制样本数
    if max_samples:
        chosen_samples = chosen_samples[:max_samples]
        rejected_samples = rejected_samples[:max_samples]
        print(f"\n⚠️  限制样本数: {max_samples}")
    
    # 3. 提取激活值 (使用指定的投影层)
    extractor = ActivationExtractor(model, tokenizer, projection_type, device)
    
    print(f"\n📊 提取 Chosen 样本激活值 ({projection_type}):")
    h_chosen = extractor.extract_activations(chosen_samples, max_samples)
    
    print(f"\n📊 提取 Rejected 样本激活值 ({projection_type}):")
    h_rejected = extractor.extract_activations(rejected_samples, max_samples)
    
    # 4. 计算特征差分并按层保存
    print(f"\n💾 计算并保存 {projection_type} 的特征差分...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 按层组织特征差分
    layer_diffs = {}
    
    num_layers = model.config.num_hidden_layers
    
    for layer_id in range(num_layers):
        layer_key = f"layer-{layer_id}"
        
        if layer_key not in h_chosen or layer_key not in h_rejected:
            print(f"   ⚠️  跳过 {layer_key}: 缺少激活值")
            continue
        
        # 计算特征差分: Δh = h_chosen - h_rejected
        diff = h_chosen[layer_key] - h_rejected[layer_key]
        layer_diffs[layer_id] = diff
        
        print(f"   ✅ Layer {layer_id:2d} | Shape: {diff.shape} | "
              f"Mean: {diff.mean():.4f} | Std: {diff.std():.4f}")
    
    # 5. 保存到文件 (文件名包含投影层类型)
    output_file = output_dir / f'{dimension}_{projection_type}_feature_diff.npz'
    np.savez_compressed(
        output_file,
        **{f'layer_{layer_id}': diff for layer_id, diff in layer_diffs.items()},
        num_layers=num_layers,
        num_samples=len(chosen_samples),
        hidden_size=list(layer_diffs.values())[0].shape[1] if layer_diffs else 0
    )
    
    print(f"\n✅ 特征差分已保存到: {output_file}")
    print(f"   - 投影层: {projection_type}")
    print(f"   - 层数: {len(layer_diffs)}")
    print(f"   - 样本数: {len(chosen_samples)}")
    print(f"   - 输出维度: {list(layer_diffs.values())[0].shape[1] if layer_diffs else 'N/A'}")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='提取偏好特征差分 (支持指定投影层)')
    parser.add_argument('--model_name', type=str, required=True,
                       help='模型名称或路径')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据目录 (包含 {dimension}_chosen.jsonl 和 {dimension}_rejected.jsonl)')
    parser.add_argument('--dimension', type=str, required=True,
                       help='偏好维度 (safety/helpfulness/correctness/coherence)')
    parser.add_argument('--projection', type=str, required=True,
                       choices=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj'],
                       help='投影层类型 (q/k/v/o在self_attn, up/down/gate在mlp)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大样本数 (默认使用全部)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='设备 (cuda:0, cuda:1, etc.)')
    
    args = parser.parse_args()
    
    extract_and_save_features(
        model_name=args.model_name,
        data_dir=args.data_dir,
        dimension=args.dimension,
        projection_type=args.projection,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        device=args.device
    )