"""
多位置探针训练脚本
支持同时训练多个 token 位置的探针，用于对比不同位置的效果

Token 位置策略:
1. user_last: 用户问题的最后一个 token (输入理解阶段)
2. assistant_first: 助手回答的第一个 token (决策起点)
3. assistant_last: 助手回答的最后一个 token (整体表征, 标准做法)
4. assistant_mean: 助手回答的所有 token 平均 (生成过程)
"""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import argparse
import os
import pickle
from datetime import datetime
from typing import Dict, List, Tuple


class MultiPositionActivationExtractor:
    """提取多个 token 位置的激活值"""
    
    def __init__(self, model, tokenizer, device='cuda:0'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads
        self.hidden_size = model.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        
        # 存储完整的激活值序列 (不只是最后一个 token)
        self.activations = {}
        self.hooks = []
        
        print(f"📊 模型配置:")
        print(f"   层数: {self.num_layers}")
        print(f"   注意力头数: {self.num_heads}")
        print(f"   隐藏层维度: {self.hidden_size}")
        print(f"   每个头维度: {self.head_dim}")
    
    def _get_activation_hook(self, layer_id):
        """创建 hook 函数来捕获完整序列的激活值"""
        def hook(module, input, output):
            key = f"layer-{layer_id}"
            if key not in self.activations:
                self.activations[key] = []
            # 保存完整序列 (batch, seq_len, hidden_dim)
            self.activations[key].append(output.detach().cpu())
        return hook
    
    def register_hooks(self):
        """注册 hooks 到所有层的 Q 投影"""
        print("🔧 注册激活值提取 hooks...")
        for layer_id in range(self.num_layers):
            layer = self.model.model.layers[layer_id]
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
            'user_last': int,      # 用户最后一个token位置
            'assistant_first': int, # 助手第一个token位置
            'assistant_last': int,  # 助手最后一个token位置
            'assistant_range': (start, end)  # 助手token范围
        }
        """
        # 获取完整 token 文本
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # 查找关键位置
        positions = {}
        
        # 寻找 <|im_start|>assistant 的位置
        assistant_markers = ['assistant', '<|assistant|>', 'Assistant']
        assistant_start = -1
        
        for i, token in enumerate(tokens):
            for marker in assistant_markers:
                if marker.lower() in token.lower():
                    assistant_start = i
                    break
            if assistant_start != -1:
                break
        
        # 如果找不到助手标记，使用简单的分割策略
        if assistant_start == -1:
            # 假设前半部分是用户，后半部分是助手
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
        
        try:
            text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
        except:
            text = f"User: {prompt}\nAssistant: {response}"
        
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
            
            # layer_acts: List[Tensor(1, seq_len, hidden_dim)]
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


def train_linear_probes_with_cv(acts_good, acts_bad, test_split=0.2, cv_folds=5,
                                  max_iter=2000, C=1.0):
    """为每个注意力头训练线性探针"""
    results = {}
    probes = {}
    detailed_results = {}
    
    print(f"\n🎯 开始训练线性探针")
    print(f"   测试集比例: {test_split}")
    print(f"   交叉验证折数: {cv_folds}")
    print(f"   最大迭代次数: {max_iter}")
    print(f"   正则化参数C: {C}")
    print()
    
    for key in tqdm(acts_good.keys(), desc="训练探针"):
        X_good = acts_good[key]  # label=1
        X_bad = acts_bad[key]    # label=0
        
        X = np.vstack([X_good, X_bad])
        y = np.hstack([
            np.ones(len(X_good)),
            np.zeros(len(X_bad))
        ])
        
        # 划分训练/测试集
        n_train = int((1 - test_split) * len(X))
        indices = np.random.permutation(len(X))
        train_idx, test_idx = indices[:n_train], indices[n_train:]
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 训练逻辑回归分类器
        clf = LogisticRegression(
            max_iter=max_iter,
            random_state=42,
            solver='lbfgs',
            C=C
        )
        clf.fit(X_train, y_train)
        
        # 测试集评估
        y_pred = clf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # 交叉验证
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv_folds)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results[key] = {
            'test_accuracy': float(test_accuracy),
            'cv_mean': float(cv_mean),
            'cv_std': float(cv_std),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        probes[key] = clf
        
        if test_accuracy >= 0.8:
            detailed_results[key] = {
                'accuracy': float(test_accuracy),
                'cv_mean': float(cv_mean),
                'coefficients': clf.coef_.tolist(),
                'intercept': float(clf.intercept_[0])
            }
    
    return results, probes, detailed_results


def load_paired_data(file_path):
    """加载配对数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def print_summary(all_results: Dict[str, Dict]):
    """打印各位置的统计摘要"""
    print("\n" + "="*80)
    print("📊 多位置探针训练结果汇总")
    print("="*80)
    
    for pos_name, results in all_results.items():
        test_accs = [r['test_accuracy'] for r in results.values()]
        cv_means = [r['cv_mean'] for r in results.values()]
        
        print(f"\n📍 位置: {pos_name}")
        print(f"   总注意力头数: {len(results)}")
        print(f"   平均测试准确率: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
        print(f"   最高测试准确率: {np.max(test_accs):.4f}")
        print(f"   平均CV准确率: {np.mean(cv_means):.4f}")
        print(f"   准确率 >= 0.8: {sum(1 for a in test_accs if a >= 0.8)} 个")
        print(f"   准确率 >= 0.9: {sum(1 for a in test_accs if a >= 0.9)} 个")
        
        # Top 3
        top_3 = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)[:3]
        print(f"   🏆 Top 3:")
        for i, (head, metrics) in enumerate(top_3, 1):
            print(f"      {i}. {head}: {metrics['test_accuracy']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='训练多位置探针')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, required=True, help='模型名称或路径')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    
    # 数据参数
    parser.add_argument('--good_pairs', type=str, required=True, help='好样本配对数据路径')
    parser.add_argument('--bad_pairs', type=str, required=True, help='坏样本配对数据路径')
    parser.add_argument('--max_samples', type=int, default=None, help='每个类别最大样本数')
    
    # 位置参数
    parser.add_argument('--positions', type=str, nargs='+', 
                       default=['user_last', 'assistant_first', 'assistant_last', 'assistant_mean'],
                       choices=['user_last', 'assistant_first', 'assistant_last', 'assistant_mean'],
                       help='要提取的token位置')
    
    # 训练参数
    parser.add_argument('--test_split', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--cv_folds', type=int, default=5, help='交叉验证折数')
    parser.add_argument('--max_iter', type=int, default=2000, help='逻辑回归最大迭代次数')
    parser.add_argument('--reg_C', type=float, default=1.0, help='正则化参数C')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--dimension', type=str, required=True, help='维度名称 (用于命名)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("🚀 多位置探针训练")
    print("="*80)
    print(f"模型: {args.model_name}")
    print(f"维度: {args.dimension}")
    print(f"位置: {', '.join(args.positions)}")
    print(f"设备: {args.device}")
    print("="*80)
    
    # 加载模型
    print("\n📦 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=args.device
    )
    model.eval()
    
    # 加载数据
    print("\n📂 加载数据...")
    good_pairs = load_paired_data(args.good_pairs)
    bad_pairs = load_paired_data(args.bad_pairs)
    print(f"   好样本: {len(good_pairs)} 对")
    print(f"   坏样本: {len(bad_pairs)} 对")
    
    # 提取激活值
    extractor = MultiPositionActivationExtractor(model, tokenizer, args.device)
    
    print("\n🔍 提取好样本激活值...")
    acts_good_multi = extractor.extract_from_pairs(good_pairs, args.max_samples, args.positions)
    
    print("\n🔍 提取坏样本激活值...")
    acts_bad_multi = extractor.extract_from_pairs(bad_pairs, args.max_samples, args.positions)
    
    # 为每个位置训练探针
    all_results = {}
    all_probes = {}
    
    for pos_name in args.positions:
        print(f"\n{'='*80}")
        print(f"📍 训练位置: {pos_name}")
        print(f"{'='*80}")
        
        acts_good = acts_good_multi[pos_name]
        acts_bad = acts_bad_multi[pos_name]
        
        results, probes, detailed = train_linear_probes_with_cv(
            acts_good, acts_bad,
            test_split=args.test_split,
            cv_folds=args.cv_folds,
            max_iter=args.max_iter,
            C=args.reg_C
        )
        
        all_results[pos_name] = results
        all_probes[pos_name] = probes
        
        # 保存结果
        result_file = os.path.join(args.output_dir, f"{args.dimension}_{pos_name}_results.json")
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 结果已保存: {result_file}")
        
        # 保存探针
        probe_dir = os.path.join(args.output_dir, f"{args.dimension}_{pos_name}_probes")
        os.makedirs(probe_dir, exist_ok=True)
        for head_key, probe in probes.items():
            probe_file = os.path.join(probe_dir, f"{head_key}.pkl")
            with open(probe_file, 'wb') as f:
                pickle.dump(probe, f)
        print(f"💾 探针已保存: {probe_dir}/")
    
    # 打印汇总
    print_summary(all_results)
    
    # 保存对比报告
    report_file = os.path.join(args.output_dir, f"{args.dimension}_position_comparison.txt")
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"📊 {args.dimension} 维度 - 多位置探针对比报告\n")
        f.write("="*80 + "\n\n")
        
        for pos_name, results in all_results.items():
            test_accs = [r['test_accuracy'] for r in results.values()]
            f.write(f"\n📍 {pos_name}\n")
            f.write(f"   平均准确率: {np.mean(test_accs):.4f}\n")
            f.write(f"   最高准确率: {np.max(test_accs):.4f}\n")
            f.write(f"   准确率>=0.8: {sum(1 for a in test_accs if a >= 0.8)}\n")
            f.write(f"   准确率>=0.9: {sum(1 for a in test_accs if a >= 0.9)}\n")
    
    print(f"\n📄 对比报告已保存: {report_file}")
    print("\n✅ 训练完成!")


if __name__ == '__main__':
    main()
