"""
探针准确度测试脚本

功能:
1. 加载不同类型的模型(基模型/微调模型/LoRA模型)
2. 加载已训练的探针
3. 在测试数据上评估每层探针的准确度
4. 生成详细的测试报告

使用方法:
# 测试基模型
python test_probe_accuracy.py \
    --model_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --probe_dir ../trained_probes_paired/helpfulness \
    --test_data ../data/helpsteer_merged_paired \
    --dimension helpfulness \
    --output_dir results/base_model

# 测试微调模型
python test_probe_accuracy.py \
    --model_path /path/to/finetuned/model \
    --probe_dir ../trained_probes_paired/helpfulness \
    --test_data ../data/helpsteer_merged_paired \
    --dimension helpfulness \
    --output_dir results/finetuned_model

# 测试LoRA模型
python test_probe_accuracy.py \
    --model_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --lora_path /path/to/lora/adapter \
    --probe_dir ../trained_probes_paired/helpfulness \
    --test_data ../data/helpsteer_merged_paired \
    --dimension helpfulness \
    --output_dir results/lora_model
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


class ActivationExtractor:
    """提取模型激活值的工具类"""
    
    def __init__(self, model, tokenizer, device='cuda:0'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # 自动检测模型层的正确路径
        # 不同模型和加载方式(基模型 vs LoRA)的层路径不同
        self.model_layers = self._get_model_layers()
        self.num_layers = model.config.num_hidden_layers
        
        self.activations = {}
        self.hooks = []
    
    def _get_model_layers(self):
        """自动检测模型层的正确访问路径
        
        不同情况下的层路径:
        - 基模型: model.model.layers
        - LoRA模型: model.model.model.layers 或 model.base_model.model.model.layers
        """
        # 尝试不同的路径
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
        
        raise RuntimeError("无法找到模型的层结构! 请检查模型类型。")
    
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
        """提取测试数据的激活值
        
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


def load_model(model_path, lora_path=None, device='cuda:0'):
    """加载模型(基模型/微调模型/LoRA模型)
    
    Args:
        model_path: 基模型路径
        lora_path: LoRA适配器路径(可选)
        device: 设备
        
    Returns:
        model, tokenizer, model_type
    """
    print(f"\n🔧 加载模型...")
    print(f"   基模型路径: {model_path}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device
    )
    
    model_type = "base"
    
    # 如果提供了LoRA路径,加载LoRA适配器
    if lora_path:
        print(f"   LoRA路径: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model_type = "lora"
        print(f"   ✅ LoRA适配器已加载")
    elif "checkpoint" in model_path or "finetuned" in model_path.lower():
        model_type = "finetuned"
    
    print(f"   ✅ 模型类型: {model_type}")
    print(f"   ✅ 模型参数: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    return model, tokenizer, model_type


def load_probes(probe_dir):
    """加载已训练的探针
    
    Args:
        probe_dir: 探针目录路径
        
    Returns:
        dict: {layer_id: LogisticRegression模型}
    """
    probe_file = os.path.join(probe_dir, 'linear_probes.pkl')
    
    if not os.path.exists(probe_file):
        raise FileNotFoundError(f"探针文件不存在: {probe_file}")
    
    print(f"\n📂 加载探针...")
    print(f"   探针文件: {probe_file}")
    
    with open(probe_file, 'rb') as f:
        probes = pickle.load(f)
    
    print(f"   ✅ 已加载 {len(probes)} 个层的探针")
    
    return probes


def load_test_data(test_data_dir, dimension):
    """加载测试数据
    
    Args:
        test_data_dir: 测试数据目录
        dimension: 维度名称(如 helpfulness, correctness, safety等)
        
    Returns:
        good_samples, bad_samples: 好样本和坏样本列表
    """
    # Safety维度使用不同的文件命名
    if dimension == 'safety':
        good_file = os.path.join(test_data_dir, "safe_pairs.json")
        bad_file = os.path.join(test_data_dir, "harmful_pairs.json")
    else:
        good_file = os.path.join(test_data_dir, f"{dimension}_good_pairs.json")
        bad_file = os.path.join(test_data_dir, f"{dimension}_bad_pairs.json")
    
    if not os.path.exists(good_file) or not os.path.exists(bad_file):
        raise FileNotFoundError(f"测试数据文件不存在: {good_file} 或 {bad_file}")
    
    print(f"
📊 加载测试数据...")
    print(f"   维度: {dimension}")
    if dimension == 'safety':
        print(f"   Safe数据: {good_file}")
        print(f"   Harmful数据: {bad_file}")
    else:
        print(f"   Good数据: {good_file}")
        print(f"   Bad数据: {bad_file}")
    
    with open(good_file, 'r') as f:
        good_samples = json.load(f)
    
    with open(bad_file, 'r') as f:
        bad_samples = json.load(f)
    
    if dimension == 'safety':
        print(f"   ✅ Safe样本: {len(good_samples)}")
        print(f"   ✅ Harmful样本: {len(bad_samples)}")
    else:
        print(f"   ✅ Good样本: {len(good_samples)}")
        print(f"   ✅ Bad样本: {len(bad_samples)}")
    
    return good_samples, bad_samples


def evaluate_probes(probes, good_activations, bad_activations):
    """评估每层探针的准确度
    
    Args:
        probes: 探针字典 {layer_id: LogisticRegression}
        good_activations: Good样本的激活值 {layer_id: numpy_array}
        bad_activations: Bad样本的激活值 {layer_id: numpy_array}
        
    Returns:
        dict: 每层的评估结果
    """
    print(f"\n🎯 评估探针准确度...")
    
    results = {}
    
    for layer_id, probe in tqdm(probes.items(), desc="评估层"):
        # 获取该层的激活值
        X_good = good_activations[layer_id]
        X_bad = bad_activations[layer_id]
        
        # 合并数据和标签
        X = np.vstack([X_good, X_bad])
        y = np.array([1] * len(X_good) + [0] * len(X_bad))
        
        # 预测
        y_pred = probe.predict(X)
        
        # 计算指标
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average='binary', zero_division=0
        )
        
        results[layer_id] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'n_samples': len(X),
            'n_good': len(X_good),
            'n_bad': len(X_bad)
        }
    
    return results


def print_results(results):
    """打印评估结果
    
    Args:
        results: 评估结果字典
    """
    print(f"\n" + "=" * 80)
    print(f"📊 探针准确度测试结果")
    print(f"=" * 80)
    
    # 计算统计信息
    accuracies = [r['accuracy'] for r in results.values()]
    f1_scores = [r['f1'] for r in results.values()]
    
    print(f"\n总体统计:")
    print(f"   层数: {len(results)}")
    print(f"   平均准确度: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"   最高准确度: {np.max(accuracies):.4f}")
    print(f"   最低准确度: {np.min(accuracies):.4f}")
    print(f"   平均F1分数: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    
    # 准确度分布
    acc_ge_80 = sum(1 for a in accuracies if a >= 0.80)
    acc_ge_85 = sum(1 for a in accuracies if a >= 0.85)
    acc_ge_90 = sum(1 for a in accuracies if a >= 0.90)
    
    print(f"\n准确度分布:")
    print(f"   >= 0.80: {acc_ge_80}/{len(results)} ({acc_ge_80/len(results)*100:.1f}%)")
    print(f"   >= 0.85: {acc_ge_85}/{len(results)} ({acc_ge_85/len(results)*100:.1f}%)")
    print(f"   >= 0.90: {acc_ge_90}/{len(results)} ({acc_ge_90/len(results)*100:.1f}%)")
    
    # Top 10 层
    print(f"\n�� Top 10 准确度最高的层:")
    sorted_results = sorted(
        results.items(), 
        key=lambda x: x[1]['accuracy'], 
        reverse=True
    )
    for i, (layer_id, metrics) in enumerate(sorted_results[:10], 1):
        print(f"   {i}. {layer_id}: "
              f"Acc={metrics['accuracy']:.4f}, "
              f"F1={metrics['f1']:.4f}, "
              f"Precision={metrics['precision']:.4f}, "
              f"Recall={metrics['recall']:.4f}")
    
    # Bottom 5 层
    print(f"\n⚠️ 准确度最低的5层:")
    for i, (layer_id, metrics) in enumerate(sorted_results[-5:], 1):
        print(f"   {i}. {layer_id}: "
              f"Acc={metrics['accuracy']:.4f}, "
              f"F1={metrics['f1']:.4f}")
    
    print(f"\n" + "=" * 80)


def save_results(results, output_dir, model_type, dimension):
    """保存评估结果
    
    Args:
        results: 评估结果字典
        output_dir: 输出目录
        model_type: 模型类型
        dimension: 维度名称
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备保存的数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_data = {
        'model_type': model_type,
        'dimension': dimension,
        'timestamp': timestamp,
        'layer_results': results,
        'summary': {
            'n_layers': len(results),
            'mean_accuracy': float(np.mean([r['accuracy'] for r in results.values()])),
            'std_accuracy': float(np.std([r['accuracy'] for r in results.values()])),
            'max_accuracy': float(np.max([r['accuracy'] for r in results.values()])),
            'min_accuracy': float(np.min([r['accuracy'] for r in results.values()])),
            'mean_f1': float(np.mean([r['f1'] for r in results.values()])),
            'layers_ge_80': sum(1 for r in results.values() if r['accuracy'] >= 0.80),
            'layers_ge_85': sum(1 for r in results.values() if r['accuracy'] >= 0.85),
            'layers_ge_90': sum(1 for r in results.values() if r['accuracy'] >= 0.90),
        }
    }
    
    # 保存详细结果
    results_file = os.path.join(
        output_dir, 
        f"probe_test_{model_type}_{dimension}_{timestamp}.json"
    )
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n💾 结果已保存到: {results_file}")
    
    # 保存简洁版本(仅准确度)
    accuracy_file = os.path.join(
        output_dir,
        f"accuracy_{model_type}_{dimension}_{timestamp}.txt"
    )
    with open(accuracy_file, 'w') as f:
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Dimension: {dimension}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"\nSummary:\n")
        f.write(f"  Mean Accuracy: {save_data['summary']['mean_accuracy']:.4f}\n")
        f.write(f"  Max Accuracy: {save_data['summary']['max_accuracy']:.4f}\n")
        f.write(f"  Layers >= 0.80: {save_data['summary']['layers_ge_80']}\n")
        f.write(f"  Layers >= 0.85: {save_data['summary']['layers_ge_85']}\n")
        f.write(f"  Layers >= 0.90: {save_data['summary']['layers_ge_90']}\n")
        f.write(f"\nPer-Layer Accuracy:\n")
        for layer_id in sorted(results.keys(), key=lambda x: int(x.split('-')[1])):
            acc = results[layer_id]['accuracy']
            f.write(f"  {layer_id}: {acc:.4f}\n")
    
    print(f"   简洁版本: {accuracy_file}")


def main():
    parser = argparse.ArgumentParser(description='测试探针在不同模型上的准确度')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='基模型路径')
    parser.add_argument('--lora_path', type=str, default=None,
                        help='LoRA适配器路径(可选)')
    
    # 探针和数据参数
    parser.add_argument('--probe_dir', type=str, required=True,
                        help='探针目录路径')
    parser.add_argument('--test_data', type=str, required=True,
                        help='测试数据目录路径')
    parser.add_argument('--dimension', type=str, required=True,
                        help='维度名称(如 helpfulness, correctness等)')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, required=True,
                        help='结果输出目录')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='设备(默认: cuda:0)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大测试样本数(用于快速测试)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🧪 探针准确度测试")
    print("=" * 80)
    
    # 1. 加载模型
    model, tokenizer, model_type = load_model(
        args.model_path, 
        args.lora_path, 
        args.device
    )
    
    # 2. 加载探针
    probes = load_probes(args.probe_dir)
    
    # 3. 加载测试数据
    good_samples, bad_samples = load_test_data(args.test_data, args.dimension)
    
    # 限制样本数量(用于快速测试)
    if args.max_samples:
        good_samples = good_samples[:args.max_samples]
        bad_samples = bad_samples[:args.max_samples]
        print(f"\n⚠️ 限制测试样本数: {args.max_samples}")
    
    # 4. 提取激活值
    extractor = ActivationExtractor(model, tokenizer, args.device)
    
    print(f"\n提取Good样本激活值...")
    good_activations = extractor.extract_activations(good_samples, args.max_samples)
    
    print(f"\n提取Bad样本激活值...")
    bad_activations = extractor.extract_activations(bad_samples, args.max_samples)
    
    # 5. 评估探针
    results = evaluate_probes(probes, good_activations, bad_activations)
    
    # 6. 打印结果
    print_results(results)
    
    # 7. 保存结果
    save_results(results, args.output_dir, model_type, args.dimension)
    
    print(f"\n✅ 测试完成!")


if __name__ == "__main__":
    main()
