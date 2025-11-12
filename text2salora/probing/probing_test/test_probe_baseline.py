"""
Baseline测试：不使用LoRA和约束，只用base model测试探针
用于验证探针和位置提取逻辑是否正确
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import pickle
import json
import argparse
from typing import List, Dict


class BaselineActivationExtractor:
    """Base model激活值提取器"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.activations = {}
        self.hooks = []
        
        if hasattr(model, 'model'):
            self.transformer = model.model
        else:
            self.transformer = model
    
    def register_hooks(self):
        """注册hook提取q_proj激活值"""
        def get_activation_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self.activations[name] = output.detach().cpu()
            return hook
        
        for i, layer in enumerate(self.transformer.layers):
            hook = layer.self_attn.q_proj.register_forward_hook(get_activation_hook(f'layer_{i}'))
            self.hooks.append(hook)
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def find_token_positions(self, inputs) -> Dict[str, int]:
        """定位关键token位置"""
        token_ids = inputs['input_ids'][0]
        
        im_start_id = self.tokenizer.convert_tokens_to_ids('<|im_start|>')
        im_end_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        
        im_start_positions = (token_ids == im_start_id).nonzero(as_tuple=True)[0].tolist()
        im_end_positions = (token_ids == im_end_id).nonzero(as_tuple=True)[0].tolist()
        
        if len(im_start_positions) < 2 or len(im_end_positions) < 2:
            seq_len = len(token_ids)
            mid_point = seq_len // 2
            positions = {
                'user_last': max(0, mid_point - 1),
                'assistant_first': mid_point,
                'assistant_last': seq_len - 2,
                'assistant_range': (mid_point, seq_len - 1)
            }
            return positions
        
        user_end_pos = im_end_positions[0]
        assistant_start_marker = im_start_positions[1]
        assistant_end_pos = im_end_positions[1]
        
        positions = {
            'user_last': user_end_pos - 1,
            'assistant_first': assistant_start_marker + 2,
            'assistant_last': assistant_end_pos - 1,
            'assistant_range': (assistant_start_marker + 2, assistant_end_pos)
        }
        
        return positions
    
    def extract_activations(self, text: str, position: str) -> Dict:
        """提取指定位置的激活值（按head分离）"""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, 
                               truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        self.activations = {}
        self.register_hooks()
        
        with torch.no_grad():
            _ = self.model(**inputs)
        
        token_positions = self.find_token_positions(inputs)
        
        # 获取模型配置
        if hasattr(self.model, 'config'):
            num_heads = self.model.config.num_attention_heads
            head_dim = self.model.config.hidden_size // num_heads
        else:
            num_heads = 12
            head_dim = 128
        
        result = {}
        
        for layer_key, activation in self.activations.items():
            activation = activation[0].numpy()
            layer_id = int(layer_key.split('_')[1])
            
            if position == 'assistant_mean':
                start, end = token_positions['assistant_range']
                pos_activation = activation[start:end].mean(axis=0)
            else:
                token_idx = token_positions[position]
                pos_activation = activation[token_idx]
            
            # 分离成各个头
            for head_id in range(num_heads):
                start_idx = head_id * head_dim
                end_idx = start_idx + head_dim
                head_activation = pos_activation[start_idx:end_idx]
                head_key = f'layer_{layer_id}_head_{head_id}'
                result[head_key] = head_activation
        
        self.remove_hooks()
        return result


def load_probes(probe_dir: Path, dimension: str, position: str) -> Dict:
    """加载探针"""
    probe_dir_path = probe_dir / dimension / f"{dimension}_{position}_probes"
    
    if not probe_dir_path.exists():
        raise FileNotFoundError(f"探针目录不存在: {probe_dir_path}")
    
    probes = {}
    probe_files = list(probe_dir_path.glob("layer-*-head-*.pkl"))
    
    if not probe_files:
        raise FileNotFoundError(f"探针目录为空: {probe_dir_path}")
    
    for probe_file in probe_files:
        parts = probe_file.stem.split('-')
        layer_id = int(parts[1])
        head_id = int(parts[3])
        
        with open(probe_file, 'rb') as f:
            probe_data = pickle.load(f)
        
        if layer_id not in probes:
            probes[layer_id] = {}
        probes[layer_id][head_id] = probe_data
    
    print(f"✓ 加载探针: {len(probes)} 层, 共 {sum(len(h) for h in probes.values())} 个探针")
    return probes


def load_test_data(data_path: str, dimension: str, max_samples: int = None):
    """加载测试数据"""
    data_path = Path(data_path)
    
    if dimension == 'safety':
        good_file = data_path / 'safe_pairs.json'
        bad_file = data_path / 'harmful_pairs.json'
    else:
        good_file = data_path / f'{dimension}_good_pairs.json'
        bad_file = data_path / f'{dimension}_bad_pairs.json'
    
    texts = []
    labels = []
    
    with open(good_file, 'r', encoding='utf-8') as f:
        good_data = json.load(f)
        if max_samples:
            good_data = good_data[:max_samples // 2]
        
        for item in good_data:
            text = f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n{item['response']}<|im_end|>"
            texts.append(text)
            labels.append(0)
    
    with open(bad_file, 'r', encoding='utf-8') as f:
        bad_data = json.load(f)
        if max_samples:
            bad_data = bad_data[:max_samples // 2]
        
        for item in bad_data:
            text = f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n{item['response']}<|im_end|>"
            texts.append(text)
            labels.append(1)
    
    return texts, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--probe_dir', type=str, default='../trained_probes/multi_position-1103')
    parser.add_argument('--test_data', type=str, default='../data/safety_paired')
    parser.add_argument('--dimension', type=str, default='safety')
    parser.add_argument('--position', type=str, default='assistant_last')
    parser.add_argument('--max_samples', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:3')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🧪 Baseline测试：Base Model + 探针")
    print("="*80)
    print(f"模型: {args.model_path}")
    print(f"维度: {args.dimension}")
    print(f"位置: {args.position}")
    print(f"设备: {args.device}")
    print("="*80 + "\n")
    
    # 加载base model
    print("📥 加载 Base Model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True
    )
    model.eval()
    print("✓ Base Model 加载完成\n")
    
    # 加载探针
    print("📂 加载探针...")
    probes = load_probes(Path(args.probe_dir), args.dimension, args.position)
    print()
    
    # 加载测试数据
    print("📊 加载测试数据...")
    texts, labels = load_test_data(args.test_data, args.dimension, args.max_samples)
    print(f"✓ 加载 {len(texts)} 条数据 (Good: {labels.count(0)}, Bad: {labels.count(1)})\n")
    
    # 提取激活值
    print("🔍 提取激活值...")
    extractor = BaselineActivationExtractor(model, tokenizer, args.device)
    activations_list = []
    
    for i, text in enumerate(texts):
        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(texts)}")
        
        act = extractor.extract_activations(text, args.position)
        activations_list.append(act)
    
    print(f"  进度: {len(texts)}/{len(texts)}")
    print()
    
    # 测试探针准确度
    print("📊 测试探针准确度...")
    results = {}
    y = np.array(labels)
    
    for layer_id, heads in probes.items():
        layer_accuracies = []
        
        for head_id, probe in heads.items():
            head_key = f'layer_{layer_id}_head_{head_id}'
            
            try:
                X = np.array([act[head_key] for act in activations_list])
                y_pred = probe.predict(X)
                acc = accuracy_score(y, y_pred)
                layer_accuracies.append(acc)
                results[head_key] = acc
            except KeyError:
                continue
            except Exception as e:
                print(f"⚠ {head_key} 预测失败: {e}")
                continue
        
        if layer_accuracies:
            layer_key = f'layer_{layer_id}'
            results[layer_key] = np.mean(layer_accuracies)
    
    # 计算平均准确率（只算layer级别）
    layer_accs = [v for k, v in results.items() if k.startswith('layer_') and '_head_' not in k]
    avg_acc = np.mean(layer_accs) if layer_accs else 0.0
    
    # 打印结果
    print("\n" + "="*80)
    print("📊 测试结果")
    print("="*80)
    print(f"平均准确率: {avg_acc:.4f}")
    print(f"\n各层准确率 (前10层):")
    
    layer_results = [(k, v) for k, v in results.items() if k.startswith('layer_') and '_head_' not in k]
    layer_results.sort(key=lambda x: int(x[0].split('_')[1]))
    
    for layer_key, acc in layer_results[:10]:
        print(f"  {layer_key}: {acc:.4f}")
    
    if len(layer_results) > 10:
        print(f"  ...")
    
    # 保存结果
    output_file = f"baseline_result_{args.dimension}_{args.position}.json"
    output_data = {
        'model_path': args.model_path,
        'dimension': args.dimension,
        'position': args.position,
        'num_samples': len(texts),
        'layer_accuracies': results,
        'average_accuracy': avg_acc
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_file}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
