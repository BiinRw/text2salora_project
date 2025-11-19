"""
Multi-position Probe Accuracy Test with ABC Constraints

Differences from test_multi_position_probe_accuracy.py:
- Added subspace constraint C loading and application
- Delta_W = B @ A @ C, where C = I - V @ V^T
- Other configs (data loading, probe loading, position extraction) remain the same

Key configurations:
1. Data: Priority {dimension}_good/bad_pairs.json, fallback safe/harmful_pairs_large.json
2. Probes: {probe_dir}/{dimension}_{position}_probes/layer-X-head-Y.pkl
3. Position: assistant_last = len(tokens) - 1
4. Labels: good=1, bad=0
"""


import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import json
import argparse
import os
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple


class ABCConstraintLoader:
    """Load and apply ABC constraints (C = I - V @ V^T)"""
    
    def __init__(self, subspace_dir, dimension, device='cuda:0', constrained_layers=None):
        self.subspace_dir = Path(subspace_dir)
        self.dimension = dimension
        self.device = device
        self.subspaces = {}
        self.constrained_layers = constrained_layers  # None表示所有层, 或者(start, end)元组
        
    def load_subspaces(self):
        """Load subspace V matrices for all layers from per-layer files"""
        import re
        
        # 尝试格式1: {dimension}_subspaces.pt (单文件)
        single_file = self.subspace_dir / f"{self.dimension}_subspaces.pt"
        
        if single_file.exists():
            print(f"\n🔄 Loading subspace from single file: {single_file}")
            data = torch.load(single_file, map_location='cpu', weights_only=False)
            for layer_id, V in data.items():
                if isinstance(V, torch.Tensor):
                    self.subspaces[layer_id] = V.to(self.device)
            print(f"   ✅ Loaded {len(self.subspaces)} layer subspaces")
            return True
        
        # 格式2: {dimension}_layer{N}_subspace.pt (每层一个文件)
        print(f"\n🔄 Loading subspace from per-layer files...")
        print(f"   Directory: {self.subspace_dir}")
        print(f"   Pattern: {self.dimension}_layer*_subspace.pt")
        
        layer_files = sorted(self.subspace_dir.glob(f"{self.dimension}_layer*_subspace.pt"))
        
        if not layer_files:
            print(f"⚠️  Warning: No subspace files found!")
            print(f"   Running without ABC constraints")
            return False
        
        for layer_file in layer_files:
            # 提取层号
            match = re.search(r'layer(\d+)_subspace\.pt', layer_file.name)
            if not match:
                continue
            
            layer_id = int(match.group(1))
            
            # 加载数据
            data = torch.load(layer_file, map_location='cpu', weights_only=False)
            
            # 提取 V 矩阵
            if isinstance(data, dict) and 'V' in data:
                V = data['V']
            elif isinstance(data, torch.Tensor):
                V = data
            else:
                continue
            
            self.subspaces[layer_id] = V.to(self.device)
        
        print(f"   ✅ Loaded {len(self.subspaces)} layer subspaces")
        layer_ids = sorted(self.subspaces.keys())
        print(f"   📊 Layers: [{layer_ids[0]}...{layer_ids[-1]}]")
        return True

        
        # 格式2: {dimension}_layer{N}_subspace.pt (每层一个文件)
        print(f"\n🔄 Loading subspace from per-layer files...")
        print(f"   Directory: {self.subspace_dir}")
        print(f"   Pattern: {self.dimension}_layer*_subspace.pt")
        
        layer_files = sorted(self.subspace_dir.glob(f"{self.dimension}_layer*_subspace.pt"))
        
        if not layer_files:
            print(f"⚠️  Warning: No subspace files found!")
            print(f"   Running without ABC constraints")
            return False
        
        for layer_file in layer_files:
            # 提取层号
            match = re.search(r'layer(\d+)_subspace\.pt', layer_file.name)
            if not match:
                continue
            
            layer_id = int(match.group(1))
            
            # 加载数据
            data = torch.load(layer_file, map_location='cpu', weights_only=False)
            
            # 提取 V 矩阵
            if isinstance(data, dict) and 'V' in data:
                V = data['V']
            elif isinstance(data, torch.Tensor):
                V = data
            else:
                continue
            
            self.subspaces[layer_id] = V.to(self.device)
        
        print(f"   ✅ Loaded {len(self.subspaces)} layer subspaces")
        layer_ids = sorted(self.subspaces.keys())
        print(f"   📊 Layers: [{layer_ids[0]}...{layer_ids[-1]}]")
        return True

    
    def compute_constraint_matrix(self, layer_id, hidden_dim):
        """
        # 检查该层是否需要应用约束
        if self.constrained_layers is not None:
            start, end = self.constrained_layers
            if not (start <= layer_id <= end):
                # 该层不在约束范围内,返回单位矩阵(无约束)
                return torch.eye(hidden_dim, device=self.device)
        
        Compute constraint matrix C = I - V @ V^T
        
        Args:
            layer_id: Layer index
            hidden_dim: Hidden dimension size (e.g., 1536)
        
        Returns:
            C: Constraint matrix of shape (hidden_dim, hidden_dim)
        """
        if layer_id not in self.subspaces:
            return torch.eye(hidden_dim, device=self.device)
        
        V = self.subspaces[layer_id]  # Shape: (hidden_dim, subspace_rank)
        I = torch.eye(hidden_dim, device=self.device)
        
        # C = I - V @ V^T, shape: (hidden_dim, hidden_dim)
        C = I - torch.mm(V, V.t())
        
        return C
    
    def apply_constraint_to_lora(self, lora_A, lora_B, layer_id):
        """
        Apply ABC constraint: Delta_W = B @ A @ C
        
        Args:
            lora_A: LoRA A matrix, shape (lora_rank, hidden_dim)
            lora_B: LoRA B matrix, shape (hidden_dim, lora_rank)
            layer_id: Layer index
            
        Returns:
            delta_W: Constrained weight update (hidden_dim, hidden_dim)
        """
        # Get hidden dimension from lora_A
        hidden_dim = lora_A.size(1)
        
        # Compute constraint matrix C
        C = self.compute_constraint_matrix(layer_id, hidden_dim)
        
        # Apply constraint: Delta_W = B @ (A @ C)
        # A: (r, d), C: (d, d) -> A@C: (r, d)
        # B: (d, r), A@C: (r, d) -> B@(A@C): (d, d)
        A_constrained = torch.mm(lora_A, C)
        delta_W = torch.mm(lora_B, A_constrained)
        
        return delta_W


class MultiPositionActivationExtractor:
    """提取多个 token 位置的激活值 (复用训练时的逻辑)"""
    
    def __init__(self, model, tokenizer, device='cuda:0'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.activations = {}
        self.hooks = []
        
        # 获取模型配置
        config = model.config
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        print(f"\n📊 模型配置:")
        print(f"   层数: {self.num_layers}")
        print(f"   隐藏层维度: {self.hidden_size}")
        print(f"   注意力头数: {self.num_heads}")
        print(f"   每个头维度: {self.head_dim}")
    
    def _get_activation_hook(self, layer_id):
        """创建 hook 函数来捕获完整序列的激活值"""
        def hook(module, input, output):
            key = f"layer-{layer_id}"
            if key not in self.activations:
                self.activations[key] = []
            # 保存完整序列: (batch_size, seq_len, hidden_dim)
            self.activations[key].append(output.detach().cpu())
        return hook
    
    def register_hooks(self):
        """注册 hooks 到所有层的 Q 投影"""
        self.activations = {}
        self.hooks = []
        
        # 兼容 PeftModel 和基础模型
        if hasattr(self.model, 'get_base_model'):
            # PeftModel: 使用 get_base_model() 方法
            base_model = self.model.get_base_model()
            base_layers = base_model.model.layers
        else:
            # 基础模型: model.model.layers
            base_layers = self.model.model.layers
        
        for layer_id in range(self.num_layers):
            layer = base_layers[layer_id]
            hook = layer.self_attn.q_proj.register_forward_hook(
                self._get_activation_hook(layer_id)
            )
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """移除所有 hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def find_token_positions(self, text: str, inputs) -> Dict[str, int]:
        """
        定位关键 token 位置
        
        返回格式: {
            'user_last': int,
            'assistant_first': int,
            'assistant_last': int,
            'assistant_range': (start, end)
        }
        """
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        positions = {}
        
        # 寻找 assistant 标记的位置
        assistant_markers = ['assistant', '<|assistant|>', 'Assistant']
        assistant_start = -1
        
        for i, token in enumerate(tokens):
            for marker in assistant_markers:
                if marker.lower() in token.lower():
                    assistant_start = i
                    break
            if assistant_start != -1:
                break
        
        # 如果找不到助手标记,使用简单的分割策略
        if assistant_start == -1:
            seq_len = len(tokens)
            assistant_start = seq_len // 2
        
        # 计算各个位置
        positions['user_last'] = max(0, assistant_start - 1)
        positions['assistant_first'] = min(assistant_start + 1, len(tokens) - 1)
        positions['assistant_last'] = len(tokens) - 1
        positions['assistant_range'] = (assistant_start + 1, len(tokens))
        
        return positions
    
    def format_conversation(self, prompt, response):
        """格式化对话为模型输入格式"""
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return text
    
    def extract_from_pairs(self, pairs, max_samples=None, 
                          positions=['user_last', 'assistant_first', 'assistant_last', 'assistant_mean']):
        """
        从配对数据中提取多个位置的激活值
        
        Args:
            pairs: 配对数据列表
            max_samples: 最大样本数
            positions: 要提取的位置列表
        
        Returns:
            Dict[position_name, Dict[head_key, activations]]
        """
        self.activations = {}
        self.register_hooks()
        
        print(f"📥 提取 {len(pairs)} 个配对的激活值...")
        print(f"📍 提取位置: {', '.join(positions)}")
        self.model.eval()
        
        if max_samples:
            pairs = pairs[:max_samples]
        
        # 存储每个样本的位置信息
        position_indices = []
        
        with torch.no_grad():
            for pair in tqdm(pairs, desc="提取激活"):
                text = self.format_conversation(pair['prompt'], pair['response'])
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # 记录位置
                pos_info = self.find_token_positions(text, inputs)
                position_indices.append(pos_info)
                
                self.model(**inputs)
        
        self.remove_hooks()
        
        # 整理激活值: 按位置和注意力头组织
        print("🔄 整理激活值...")
        result = {pos: {} for pos in positions}
        
        for layer_id in tqdm(range(self.num_layers), desc="处理层"):
            key = f"layer-{layer_id}"
            if key not in self.activations:
                continue
            
            layer_acts_list = self.activations[key]
            
            for head_id in range(self.num_heads):
                start_idx = head_id * self.head_dim
                end_idx = (head_id + 1) * self.head_dim
                head_key = f"layer-{layer_id}-head-{head_id}"
                
                # 为每个位置提取激活
                for pos_name in positions:
                    acts_for_position = []
                    
                    for sample_idx, (act_tensor, pos_info) in enumerate(zip(layer_acts_list, position_indices)):
                        # act_tensor: (1, seq_len, hidden_dim)
                        act = act_tensor[0, :, start_idx:end_idx].numpy()  # (seq_len, head_dim)
                        
                        if pos_name == 'user_last':
                            token_idx = pos_info['user_last']
                            acts_for_position.append(act[token_idx])
                        
                        elif pos_name == 'assistant_first':
                            token_idx = pos_info['assistant_first']
                            acts_for_position.append(act[token_idx])
                        
                        elif pos_name == 'assistant_last':
                            token_idx = pos_info['assistant_last']
                            acts_for_position.append(act[token_idx])
                        
                        elif pos_name == 'assistant_mean':
                            start, end = pos_info['assistant_range']
                            mean_act = act[start:end].mean(axis=0)
                            acts_for_position.append(mean_act)
                    
                    result[pos_name][head_key] = np.array(acts_for_position)
        
        print(f"✅ 提取完成! 每个位置共 {len(result[positions[0]])} 个注意力头")
        return result


def load_model_with_abc(model_path, lora_path, subspace_dir, dimension, device='cuda:0', constrained_layers=None):
    """Load model and apply ABC constraints"""
    print(f"\n Loading model with ABC constraints...")
    print(f"   Base model: {model_path}")
    print(f"   LoRA: {lora_path}")
    print(f"   Subspace dir: {subspace_dir}")
    print(f"   Dimension: {dimension}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print(f"\n Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    
    # Load ABC constraints
    abc_loader = ABCConstraintLoader(subspace_dir, dimension, device, constrained_layers)
    has_constraints = abc_loader.load_subspaces()
    
    # 打印层约束信息
    if constrained_layers is not None:
        print(f"   🎯 Constrained layers: {constrained_layers[0]}-{constrained_layers[1]}")
    else:
        print(f"   🎯 Constrained layers: All layers (0-27)")
    
    # Manually load LoRA and apply ABC
    print(f"\n Loading LoRA weights and applying ABC constraints...")
    
    # 尝试加载 safetensors 或 bin 格式
    adapter_file_st = os.path.join(lora_path, 'adapter_model.safetensors')
    adapter_file_bin = os.path.join(lora_path, 'adapter_model.bin')
    
    if os.path.exists(adapter_file_st):
        print(f"   Loading adapter from: adapter_model.safetensors")
        from safetensors.torch import load_file
        lora_state_dict = load_file(adapter_file_st)
    elif os.path.exists(adapter_file_bin):
        print(f"   Loading adapter from: adapter_model.bin")
        lora_state_dict = torch.load(adapter_file_bin, map_location='cpu', weights_only=False)
    else:
        raise FileNotFoundError(f"No adapter found at {lora_path}. Expected adapter_model.safetensors or adapter_model.bin")
    
    with open(os.path.join(lora_path, 'adapter_config.json'), 'r') as f:
        lora_config = json.load(f)
    
    lora_r = lora_config['r']
    lora_alpha = lora_config['lora_alpha']
    scaling = lora_alpha / lora_r
    
    print(f"   LoRA config: r={lora_r}, alpha={lora_alpha}, scaling={scaling}")
    
    # Merge layer by layer
    merge_count = 0
    skip_count = 0
    base_layers = model.model.layers
    
    for layer_id in tqdm(range(len(base_layers)), desc="Merging LoRA+ABC"):
        layer = base_layers[layer_id]
        
        q_lora_A_key = f"base_model.model.model.layers.{layer_id}.self_attn.q_proj.lora_A.weight"
        q_lora_B_key = f"base_model.model.model.layers.{layer_id}.self_attn.q_proj.lora_B.weight"
        
        if q_lora_A_key in lora_state_dict and q_lora_B_key in lora_state_dict:
            lora_A = lora_state_dict[q_lora_A_key].to(device)
            lora_B = lora_state_dict[q_lora_B_key].to(device)
            
            if has_constraints:
                delta_W = abc_loader.apply_constraint_to_lora(lora_A, lora_B, layer_id)
            else:
                delta_W = torch.mm(lora_B, lora_A)
            
            with torch.no_grad():
                layer.self_attn.q_proj.weight.data += scaling * delta_W.to(torch.float16)
            
            merge_count += 1
        else:
            skip_count += 1
    
    print(f"\n LoRA+ABC merge complete: {merge_count} merged, {skip_count} skipped")
    
    model_type = "lora_with_abc" if has_constraints else "lora_only"
    return model, tokenizer, model_type



def load_position_probes(probe_dir, position, dimension):
    """
    加载指定位置的探针
    
    Args:
        probe_dir: 探针根目录
        position: 位置名称 (如 assistant_last)
        dimension: 维度名称 (如 safety)
        
    Returns:
        dict: {head_key: LogisticRegression模型}
    """
    # 构建探针文件路径: {dimension}_{position}_probes/
    probe_subdir = os.path.join(probe_dir, f"{dimension}_{position}_probes")
    
    if not os.path.exists(probe_subdir):
        raise FileNotFoundError(f"探针目录不存在: {probe_subdir}")
    
    print(f"\n📂 加载探针 [{position}]...")
    print(f"   探针目录: {probe_subdir}")
    
    # 加载所有探针文件
    probes = {}
    probe_files = sorted([f for f in os.listdir(probe_subdir) if f.endswith('.pkl')])
    
    for probe_file in probe_files:
        # 文件名格式: layer-{layer_id}-head-{head_id}.pkl
        probe_path = os.path.join(probe_subdir, probe_file)
        with open(probe_path, 'rb') as f:
            probe = pickle.load(f)
        
        # 提取 key: layer-{layer_id}-head-{head_id}
        key = probe_file.replace('.pkl', '')
        probes[key] = probe
    
    print(f"   ✅ 已加载 {len(probes)} 个探针")
    
    return probes


def load_test_data(test_data_dir, dimension):
    """加载测试数据"""
    good_file = os.path.join(test_data_dir, f"{dimension}_good_pairs.json")
    bad_file = os.path.join(test_data_dir, f"{dimension}_bad_pairs.json")
    
    # 尝试其他可能的文件名
    if not os.path.exists(good_file):
        good_file = os.path.join(test_data_dir, "safe_pairs_large.json")
    if not os.path.exists(bad_file):
        bad_file = os.path.join(test_data_dir, "harmful_pairs_large.json")
    
    print(f"\n📂 加载测试数据...")
    print(f"   好样本: {good_file}")
    print(f"   坏样本: {bad_file}")
    
    with open(good_file, 'r', encoding='utf-8') as f:
        good_samples = json.load(f)
    
    with open(bad_file, 'r', encoding='utf-8') as f:
        bad_samples = json.load(f)
    
    print(f"   ✅ 好样本数: {len(good_samples)}")
    print(f"   ✅ 坏样本数: {len(bad_samples)}")
    
    return good_samples, bad_samples


def evaluate_position_probes(probes, good_activations, bad_activations):
    """
    评估指定位置的探针性能
    
    Args:
        probes: {head_key: probe_model}
        good_activations: {head_key: np.array}
        bad_activations: {head_key: np.array}
    
    Returns:
        results: {head_key: metrics_dict}
    """
    results = {}
    
    for head_key in tqdm(probes.keys(), desc="评估探针"):
        if head_key not in good_activations or head_key not in bad_activations:
            continue
        
        probe = probes[head_key]
        
        # 准备测试数据
        X_good = good_activations[head_key]
        X_bad = bad_activations[head_key]
        
        X_test = np.vstack([X_good, X_bad])
        y_test = np.array([1] * len(X_good) + [0] * len(X_bad))
        
        # 预测
        y_pred = probe.predict(X_test)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary', zero_division=0
        )
        
        results[head_key] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_samples': len(y_test)
        }
    
    return results


def print_position_summary(position, results):
    """打印单个位置的统计摘要"""
    accuracies = [r['accuracy'] for r in results.values()]
    f1_scores = [r['f1'] for r in results.values()]
    
    print(f"\n{'='*80}")
    print(f"📍 位置: {position}")
    print(f"{'='*80}")
    print(f"探针数量: {len(results)}")
    print(f"平均准确率: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"平均 F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"最高准确率: {np.max(accuracies):.4f}")
    print(f"最低准确率: {np.min(accuracies):.4f}")
    print(f"准确率 >= 0.8: {sum(1 for a in accuracies if a >= 0.8)}")
    print(f"准确率 >= 0.9: {sum(1 for a in accuracies if a >= 0.9)}")


def save_results(output_dir, dimension, model_type, all_position_results):
    """保存测试结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存详细结果 (JSON)
    results_file = os.path.join(
        output_dir, 
        f"{dimension}_{model_type}_multi_position_test_{timestamp}.json"
    )
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_position_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 结果已保存:")
    print(f"   {results_file}")
    
    # 生成文本报告
    report_file = os.path.join(
        output_dir,
        f"{dimension}_{model_type}_multi_position_report_{timestamp}.txt"
    )
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"多位置探针测试报告\n")
        f.write("="*80 + "\n")
        f.write(f"维度: {dimension}\n")
        f.write(f"模型类型: {model_type}\n")
        f.write(f"测试时间: {timestamp}\n")
        f.write("="*80 + "\n\n")
        
        # 位置对比汇总
        f.write("📊 位置性能对比\n")
        f.write("-"*80 + "\n")
        
        position_summary = {}
        for pos_name, pos_results in all_position_results.items():
            accuracies = [r['accuracy'] for r in pos_results.values()]
            position_summary[pos_name] = {
                'mean_acc': np.mean(accuracies),
                'std_acc': np.std(accuracies),
                'max_acc': np.max(accuracies),
                'count_80': sum(1 for a in accuracies if a >= 0.8),
                'count_90': sum(1 for a in accuracies if a >= 0.9)
            }
        
        # 按平均准确率排序
        sorted_positions = sorted(
            position_summary.items(),
            key=lambda x: x[1]['mean_acc'],
            reverse=True
        )
        
        for pos_name, summary in sorted_positions:
            f.write(f"\n📍 {pos_name}\n")
            f.write(f"   平均准确率: {summary['mean_acc']:.4f} ± {summary['std_acc']:.4f}\n")
            f.write(f"   最高准确率: {summary['max_acc']:.4f}\n")
            f.write(f"   准确率>=0.8: {summary['count_80']}\n")
            f.write(f"   准确率>=0.9: {summary['count_90']}\n")
        
        # 详细的每层结果
        f.write("\n\n" + "="*80 + "\n")
        f.write("📊 详细的每层结果\n")
        f.write("="*80 + "\n")
        
        for pos_name, pos_results in all_position_results.items():
            f.write(f"\n\n{'='*80}\n")
            f.write(f"位置: {pos_name}\n")
            f.write(f"{'='*80}\n")
            
            # 按layer排序
            sorted_heads = sorted(pos_results.keys())
            for head_key in sorted_heads:
                metrics = pos_results[head_key]
                f.write(f"\n{head_key}:\n")
                f.write(f"  准确率: {metrics['accuracy']:.4f}\n")
                f.write(f"  精确率: {metrics['precision']:.4f}\n")
                f.write(f"  召回率: {metrics['recall']:.4f}\n")
                f.write(f"  F1分数: {metrics['f1']:.4f}\n")
    
    print(f"   {report_file}")


def main():
    parser = argparse.ArgumentParser(description='多位置探针准确度测试')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True, 
                       help='基础模型路径')
    parser.add_argument('--lora_path', type=str, required=True,
                       help='LoRA adapter path')
    parser.add_argument('--subspace_dir', type=str,
                       default='preference_subspace/saved_subspaces',
                       help='Subspace directory')
    parser.add_argument('--constrained_layers', type=str, default=None,
                       help='约束层范围,格式: "start,end" (如 "0,8" 或 "16,16"), None表示所有层')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='计算设备')
    
    # 探针参数
    parser.add_argument('--probe_dir', type=str, required=True,
                       help='探针根目录 (包含多个位置的探针)')
    parser.add_argument('--positions', type=str, nargs='+',
                       default=['assistant_last'],
                       choices=['user_last', 'assistant_first', 'assistant_last', 'assistant_mean'],
                       help='要测试的位置列表')
    
    # 数据参数
    parser.add_argument('--test_data', type=str, required=True,
                       help='测试数据目录')
    parser.add_argument('--dimension', type=str, required=True,
                       help='测试维度 (如 safety, helpfulness)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大测试样本数')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='results',
                       help='结果输出目录')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🧪 多位置探针准确度测试")
    print("="*80)
    print(f"📊 维度: {args.dimension}")
    print(f"📍 测试位置: {', '.join(args.positions)}")
    print(f"📁 探针目录: {args.probe_dir}")
    print(f"📁 测试数据: {args.test_data}")
    if args.max_samples:
        print(f"📦 测试样本数: {args.max_samples}")
    print("="*80)
    
    # 1. 加载模型
    # 解析层约束参数
    constrained_layers = None
    if args.constrained_layers:
        start, end = map(int, args.constrained_layers.split(','))
        constrained_layers = (start, end)
        print(f"🎯 将约束应用于层: {start}-{end}")
    else:
        print(f"🎯 将约束应用于所有层")
    
    model, tokenizer, model_type = load_model_with_abc(
        args.model_path,
        args.lora_path,
        args.subspace_dir,
        args.dimension,
        args.device,
        constrained_layers
    )
    
    # 2. 加载测试数据
    good_samples, bad_samples = load_test_data(args.test_data, args.dimension)
    
    # 3. 创建激活提取器
    extractor = MultiPositionActivationExtractor(model, tokenizer, args.device)
    
    # 4. 提取多位置激活
    print(f"\n{'='*80}")
    print("🔍 提取激活值 (所有位置)")
    print(f"{'='*80}")
    
    good_acts_multi = extractor.extract_from_pairs(
        good_samples,
        args.max_samples,
        args.positions
    )
    
    bad_acts_multi = extractor.extract_from_pairs(
        bad_samples,
        args.max_samples,
        args.positions
    )
    
    # 5. 为每个位置加载探针并测试
    all_position_results = {}
    
    for position in args.positions:
        print(f"\n{'='*80}")
        print(f"🧪 测试位置: {position}")
        print(f"{'='*80}")
        
        # 加载该位置的探针
        try:
            probes = load_position_probes(args.probe_dir, position, args.dimension)
        except FileNotFoundError as e:
            print(f"⚠️  跳过位置 {position}: {e}")
            continue
        
        # 评估该位置的探针
        position_results = evaluate_position_probes(
            probes,
            good_acts_multi[position],
            bad_acts_multi[position]
        )
        
        all_position_results[position] = position_results
        
        # 打印该位置的统计摘要
        print_position_summary(position, position_results)
    
    # 6. 保存结果
    if all_position_results:
        save_results(args.output_dir, args.dimension, model_type, all_position_results)
        
        print(f"\n{'='*80}")
        print("✅ 测试完成!")
        print(f"{'='*80}")
    else:
        print(f"\n⚠️  没有成功测试任何位置")


if __name__ == '__main__':
    main()
