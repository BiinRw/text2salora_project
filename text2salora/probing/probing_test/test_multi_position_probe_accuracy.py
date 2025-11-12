"""
多位置探针准确度测试脚本

功能:
1. 加载不同类型的模型(基模型/微调模型/LoRA模型)
2. 加载已训练的多位置探针
3. 在测试数据上评估每个位置、每层探针的准确度
4. 生成详细的多位置对比测试报告

位置说明:
- user_last: 用户问题的最后一个token
- assistant_first: 助手回答的第一个token  
- assistant_last: 助手回答的最后一个token (标准)
- assistant_mean: 助手回答的所有token平均

使用方法:
python test_multi_position_probe_accuracy.py \
    --model_path Qwen/Qwen2.5-1.5B-Instruct \
    --lora_path /path/to/lora/checkpoint \
    --probe_dir ../results_multi_position/safety \
    --test_data ../data/safety_paired \
    --dimension safety \
    --positions assistant_last assistant_first \
    --max_samples 100 \
    --output_dir results/multi_position_test
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


def load_model(model_path, lora_path=None, device='cuda:0'):
    """加载模型"""
    print(f"\n🔧 加载模型...")
    print(f"   基模型路径: {model_path}")
    if lora_path:
        print(f"   LoRA路径: {lora_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
        model_type = "lora"
        print(f"   ✅ LoRA适配器已加载")
    elif "checkpoint" in model_path or "finetuned" in model_path.lower():
        model_type = "finetuned"
    else:
        model_type = "base"
    
    print(f"   ✅ 模型类型: {model_type}")
    print(f"   ✅ 模型参数: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
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
    parser.add_argument('--lora_path', type=str, default=None,
                       help='LoRA适配器路径 (可选)')
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
    model, tokenizer, model_type = load_model(
        args.model_path,
        args.lora_path,
        args.device
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
