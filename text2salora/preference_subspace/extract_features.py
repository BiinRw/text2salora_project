"""
偏好子空间特征提取
提取 chosen/rejected 样本的激活值,计算特征差分
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
    """提取模型激活值的工具类 (复用 probing 实现)"""
    
    def __init__(self, model, tokenizer, device='cuda:0'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # 自动检测模型层的正确路径
        self.model_layers = self._get_model_layers()
        self.num_layers = model.config.num_hidden_layers
        
        self.activations = {}
        self.hooks = []
    
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
        """注册hooks到所有层的Q投影"""
        for layer_id in range(self.num_layers):
            layer = self.model_layers[layer_id]
            hook = layer.self_attn.q_proj.register_forward_hook(
                self._get_activation_hook(layer_id)
            )
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
            data_samples: 数据样本列表,每个样本包含 prompt 和 response
            max_samples: 最大样本数量限制
            
        Returns:
            dict: {layer_id: numpy_array} 每层的激活值
        """
        self.activations = {}
        self.register_hooks()
        
        if max_samples:
            data_samples = data_samples[:max_samples]
        
        print(f"📥 提取 {len(data_samples)} 个样本的激活值...")
        self.model.eval()
        
        with torch.no_grad():
            for sample in tqdm(data_samples, desc="提取激活"):
                text = self.format_conversation(sample['prompt'], sample['response'])
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512
                ).to(self.device)
                self.model(**inputs)
        
        self.remove_hooks()
        
        # 整理激活值为numpy数组,并按注意力头分割
        head_activations = {}
        
        for layer_id in range(self.num_layers):
            layer_key = f"layer-{layer_id}"
            if layer_key in self.activations:
                # 合并该层所有样本的激活值
                layer_acts = torch.cat(self.activations[layer_key], dim=0).numpy()
                
                # 计算每个头的维度
                num_heads = self.model.config.num_attention_heads
                head_dim = self.model.config.hidden_size // num_heads
                
                # 按头分割激活值
                for head_id in range(num_heads):
                    start_idx = head_id * head_dim
                    end_idx = (head_id + 1) * head_dim
                    head_key = f"layer-{layer_id}-head-{head_id}"
                    head_activations[head_key] = layer_acts[:, start_idx:end_idx]
        
        return head_activations


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
    output_dir: str,
    max_samples: int = None,
    device: str = 'cuda:0'
):
    """提取并保存特征差分
    
    Args:
        model_name: 模型名称或路径
        data_dir: 数据目录
        dimension: 偏好维度
        output_dir: 输出目录
        max_samples: 最大样本数
        device: 设备
    """
    print("=" * 80)
    print(f"🚀 开始提取 {dimension} 维度的特征")
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
    
    # 3. 提取激活值
    extractor = ActivationExtractor(model, tokenizer, device)
    
    print(f"\n📊 提取 Chosen 样本激活值:")
    h_chosen = extractor.extract_activations(chosen_samples, max_samples)
    
    print(f"\n📊 提取 Rejected 样本激活值:")
    h_rejected = extractor.extract_activations(rejected_samples, max_samples)
    
    # 4. 计算特征差分并按层保存
    print(f"\n💾 计算并保存特征差分...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 按层组织特征差分
    layer_diffs = {}
    
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    for layer_id in range(num_layers):
        # 合并该层所有头的激活值
        layer_chosen = []
        layer_rejected = []
        
        for head_id in range(num_heads):
            head_key = f"layer-{layer_id}-head-{head_id}"
            if head_key in h_chosen and head_key in h_rejected:
                layer_chosen.append(h_chosen[head_key])
                layer_rejected.append(h_rejected[head_key])
        
        if layer_chosen:
            # 拼接所有头 (N, hidden_size)
            layer_chosen = np.concatenate(layer_chosen, axis=1)
            layer_rejected = np.concatenate(layer_rejected, axis=1)
            
            # 计算差分
            diff = layer_chosen - layer_rejected
            layer_diffs[layer_id] = diff
            
            print(f"   Layer {layer_id:2d}: diff shape = {diff.shape}")
    
    # 5. 保存
    output_file = output_dir / f'{dimension}_feature_diff.npz'
    np.savez(
        output_file,
        **{f'layer_{i}': diff for i, diff in layer_diffs.items()},
        num_layers=num_layers,
        num_samples=len(chosen_samples),
        hidden_size=model.config.hidden_size
    )
    
    print(f"\n✅ 特征差分已保存: {output_file}")
    print(f"   包含 {len(layer_diffs)} 层的特征差分")
    
    # 清理
    del model
    torch.cuda.empty_cache()
    
    return output_file


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='提取偏好特征差分')
    parser.add_argument('--model_name', type=str, required=True,
                        help='模型名称或路径')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据目录路径')
    parser.add_argument('--dimension', type=str, required=True,
                        choices=['safety', 'helpfulness', 'correctness', 'coherence'],
                        help='偏好维度')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大样本数(用于测试)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='设备')
    
    args = parser.parse_args()
    
    extract_and_save_features(
        model_name=args.model_name,
        data_dir=args.data_dir,
        dimension=args.dimension,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        device=args.device
    )
