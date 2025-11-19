"""
主训练文件 v2 - 修复版
应用了所有发现的修复:
1. 梯度问题修复
2. GPU 选择
3. 命令行接口
4. 可选的正交约束
"""

import torch
import torch.nn as nn
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset
import numpy as np
from training_monitor import create_training_callbacks
from dataset_loader import load_ultrafeedback_data

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from protected_lora.orthogonal_constraints import (
    OrthogonalConstraint,
    collect_lora_AB_matrices
)
from protected_lora.peft_lora_patch import inject_hard_constraint_to_model


class SimpleDataset(Dataset):
    """简单文本数据集"""
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.encodings['input_ids'][idx]
        }


class OrthogonalLoRATrainer(Trainer):
    """支持正交约束的 Trainer"""
    
    def __init__(
        self,
        constraint_calculator: Optional[OrthogonalConstraint] = None,
        lambda_orth: float = 0.1,
        dimension_weights: Optional[Dict[str, float]] = None,
        use_orthogonal: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.constraint = constraint_calculator
        self.lambda_orth = lambda_orth
        self.dimension_weights = dimension_weights or {}
        self.use_orthogonal = use_orthogonal
        self.orth_loss_history = []
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """计算损失 = 任务损失 + 正交损失"""

        # 1. 计算任务损失
        outputs = model(**inputs)

        if isinstance(outputs, dict):
            task_loss = outputs.get('loss')
        else:
            task_loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss

        # 2. 如果不使用正交约束,直接返回
        if not self.use_orthogonal or self.constraint is None:
            return (task_loss, outputs) if return_outputs else task_loss

        # 3. 计算正交损失
        if model.training:
            lora_A, lora_B = collect_lora_AB_matrices(model)

            if len(lora_A) > 0:
                orth_loss, loss_details = self.constraint.compute_orthogonal_loss_efficient(
                    lora_A, lora_B,
                    lambda_orth=self.lambda_orth,
                    dimension_weights=self.dimension_weights
                )

                # 记录
                self.orth_loss_history.append({
                    'step': self.state.global_step,
                    'task_loss': task_loss.item(),
                    'orth_loss': orth_loss.item(),
                    'details': loss_details
                })

                # 总损失
                total_loss = task_loss + orth_loss
            else:
                total_loss = task_loss
        else:
            total_loss = task_loss

        return (total_loss, outputs) if return_outputs else total_loss

    def save_orth_loss_history(self, output_dir: str):
        """保存正交损失历史"""
        if len(self.orth_loss_history) > 0:
            output_path = Path(output_dir) / 'orth_loss_history.json'
            with open(output_path, 'w') as f:
                json.dump(self.orth_loss_history, f, indent=2)
            print(f"✅ 正交损失历史已保存: {output_path}")


def setup_lora_model(
    model_name: str,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: List[str] = None,
    device: str = 'cuda:0'
) -> tuple:
    """设置 LoRA 模型
    
    Returns:
        (model, tokenizer)
    """
    if target_modules is None:
        target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
    
    print(f"\n📥 加载模型: {model_name}")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ 模型加载完成")
    
    # 配置 LoRA
    print(f"\n🔧 应用 LoRA (rank={lora_rank}, alpha={lora_alpha})")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        init_lora_weights=True,
        bias='none'
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # ✅ 修复: 确保 LoRA 参数需要梯度
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
    
    return model, tokenizer


def load_subspace_constraint(
    subspace_dir: str,
    dimensions: List[str],
    device: str = 'cuda:0',
    subspace_rank: int = None  # 新增: 使用子空间的前 k 个向量，None=使用全部
) -> OrthogonalConstraint:
    """加载子空间约束。支持 .pt 和 .npy 文件格式。
    优先加载 fused 子空间文件（如 safety_fused_subspace.pt）。
    
    Args:
        subspace_dir: 子空间文件目录
        dimensions: 偏好维度列表
        device: 设备
        subspace_rank: 使用子空间的前 k 个向量（None=使用全部）
    """
    from pathlib import Path
    
    print(f"\n📊 加载偏好子空间: {dimensions}")
    print(f"   子空间目录: {subspace_dir}")
    
    # 加载子空间矩阵
    subspace_V = {}
    
    for dim in dimensions:
        # 尝试多种文件格式
        candidates = [
            Path(subspace_dir) / f"{dim}_fused_subspace.pt",  # PyTorch 格式（优先）
            Path(subspace_dir) / f"{dim}.pt",
            Path(subspace_dir) / f"{dim}_subspace.pt",
            Path(subspace_dir) / f"{dim}.npy",  # NumPy 格式
            Path(subspace_dir) / f"{dim}_V.npy",
            Path(subspace_dir) / f"{dim}_subspace.npy",
        ]
        
        loaded = False
        for p in candidates:
            if p.exists():
                print(f"   找到文件: {p.name}")
                try:
                    if p.suffix == '.pt':
                        # 加载 PyTorch 文件
                        V_tensor = torch.load(p, map_location=device)
                        if isinstance(V_tensor, dict):
                            # 如果是字典，尝试提取子空间矩阵
                            if 'subspace' in V_tensor:
                                V_tensor = V_tensor['subspace']
                            elif 'V' in V_tensor:
                                V_tensor = V_tensor['V']
                            else:
                                # 使用第一个值
                                V_tensor = list(V_tensor.values())[0]
                        V_tensor = V_tensor.to(device)
                    elif p.suffix == '.npy':
                        # 加载 NumPy 文件
                        arr = np.load(p)
                        if arr.ndim == 1:
                            arr = arr[:, None]
                        V_tensor = torch.from_numpy(arr).to(device)
                    else:
                        continue
                    
                    # 确保是 2D 张量
                    if V_tensor.ndim == 1:
                        V_tensor = V_tensor.unsqueeze(1)
                    
                    # 截断子空间（如果指定了 subspace_rank）
                    if subspace_rank is not None and V_tensor.shape[1] > subspace_rank:
                        original_rank = V_tensor.shape[1]
                        V_tensor = V_tensor[:, :subspace_rank]
                        print(f"   📊 截断子空间: {original_rank} → {subspace_rank}")
                    
                    subspace_V[dim] = V_tensor
                    print(f"   ✅ {dim}: shape={V_tensor.shape}")
                    loaded = True
                    break
                    
                except Exception as e:
                    print(f"   ⚠️ 加载 {p} 失败: {e}")
                    continue
        
        if not loaded:
            raise FileNotFoundError(
                f"无法找到子空间文件 for dimension '{dim}' in {subspace_dir}.\n"
                f"尝试的文件: {[str(c) for c in candidates]}"
            )
    
    # 创建正交约束
    # 创建 PreferenceSubspaceManager
    from utils.svd_utils import PreferenceSubspaceManager
    manager = PreferenceSubspaceManager(subspace_dir=subspace_dir, device=device)
    
    # 将加载的子空间矩阵放入 manager
    for dim in dimensions:
        manager.subspaces[dim] = {'fused': subspace_V[dim]}
    
    # 创建正交约束
    constraint = OrthogonalConstraint(
        subspace_manager=manager,
        dimensions=dimensions,
        device=device
    )
    
    print(f"✅ 子空间加载完成")
    return constraint

def train(
    model_name: str,
    train_texts: List[str] = None,
    eval_texts: List[str] = None,
    
    # 数据集配置
    dataset_type: str = 'demo',
    dataset_size: str = '100',
    data_format: str = 'instruction',
    max_samples: int = None,
    output_dir: str = './output/train',
    
    # GPU & 资源
    gpu_id: int = 1,
    
    # LoRA 配置
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: List[str] = None,
    
    # 训练配置
    num_epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation: int = 4,
    learning_rate: float = 1e-4,
    max_length: int = 512,
    use_gradient_checkpointing: bool = False,
    
    # 正交约束
    use_orthogonal: bool = False,
    use_hard_constraint: bool = False,  # 🔑 新增: 使用硬约束（SaLoRA 风格）
    subspace_dir: str = None,
    preference_dimensions: List[str] = None,
    lambda_orth: float = 0.1,
    dimension_weights: Dict[str, float] = None,
    subspace_rank: int = None,  # 新增: 使用子空间的前 k 个向量（None=使用全部）
    
    # 训练监控
    use_swanlab: bool = True,
    swanlab_project: str = 'protected-lora',
    experiment_name: str = None,
    print_interval: int = 10,
    enable_console_logging: bool = True
):
    """主训练函数
    
    Args:
        model_name: 模型名称或路径
        train_texts: 训练文本列表
        eval_texts: 验证文本列表
        output_dir: 输出目录
        gpu_id: GPU ID
        
        LoRA 配置:
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        target_modules: 目标模块
        
        训练配置:
        num_epochs: 训练轮数
        batch_size: batch size
        gradient_accumulation: 梯度累积步数
        learning_rate: 学习率
        max_length: 最大序列长度
        use_gradient_checkpointing: 是否使用梯度检查点
        
        正交约束:
        use_orthogonal: 是否使用正交约束
        subspace_dir: 子空间目录
        preference_dimensions: 偏好维度列表
        lambda_orth: 正交损失系数
        dimension_weights: 维度权重
    """
    
    print("="*80)


    print("🚀 开始训练 - 主训练文件 v2")
    print("="*80)
    
    # ✅ 修复: GPU 选择
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = 'cuda:0'
    
    print(f"\n📊 配置:")
    print(f"  模型: {model_name}")
    print(f"  GPU: {gpu_id}")
    if train_texts is not None:
        print(f"  训练样本: {len(train_texts)}")
    else:
        print(f"  数据集类型: {dataset_type}")
    print(f"  LoRA: rank={lora_rank}, alpha={lora_alpha}")
    print(f"  训练: epochs={num_epochs}, batch={batch_size}, lr={learning_rate}")
    if use_hard_constraint:
        print(f"  约束模式: 🔒 硬约束 (SaLoRA 风格，表征空间)")
    elif use_orthogonal:
        print(f"  约束模式: 📊 软约束 (参数空间，lambda={lambda_orth})")
    else:
        print(f"  约束模式: ❌ 无约束")
    
    # 1. 设置模型
    model, tokenizer = setup_lora_model(
        model_name=model_name,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        device=device
    )
    
    # 2. 准备数据
    print(f"\n📚 准备数据集")
    
    if dataset_type == 'ultrafeedback':
        # 使用 UltraFeedback 数据集
        print(f"  类型: UltraFeedback")
        print(f"  大小: {dataset_size}")
        print(f"  格式: {data_format}")
        
        from dataset_loader import load_ultrafeedback_data
        
        # 加载训练和验证数据
        train_dataset, eval_dataset = load_ultrafeedback_data(
            dataset_size=dataset_size,
            tokenizer=tokenizer,
            max_length=max_length,
            use_chosen_only=True,
            format_type=data_format,
            split="both"
        )
        
        # 限制样本数（用于快速测试）
        if max_samples and max_samples < len(train_dataset):
            print(f"  ⚠️  限制训练样本: {max_samples}/{len(train_dataset)}")
            train_dataset.data = train_dataset.data[:max_samples]
        
    else:
        # 使用简单文本数据
        if train_texts is None:
            raise ValueError("demo 模式需要提供 train_texts")
        train_dataset = SimpleDataset(train_texts, tokenizer, max_length)
        eval_dataset = SimpleDataset(eval_texts, tokenizer, max_length) if eval_texts else None
    print(f"✅ 训练集: {len(train_dataset)} 样本")
    if eval_dataset:
        print(f"✅ 验证集: {len(eval_dataset)} 样本")
    
    # 3. 加载约束 (软约束或硬约束)
    constraint = None
    
    if use_hard_constraint:
        # 🔒 硬约束模式 (SaLoRA 风格)
        if subspace_dir is None or preference_dimensions is None:
            print("⚠️  警告: use_hard_constraint=True 但未提供 subspace_dir 或 preference_dimensions")
            print("⚠️  将关闭硬约束")
            use_hard_constraint = False
        else:
            print(f"\n🔒 加载硬约束 (SaLoRA 风格)...")
            # 加载子空间
            constraint = load_subspace_constraint(
                subspace_dir=subspace_dir,
                dimensions=preference_dimensions,
                device=device,
                subspace_rank=subspace_rank
            )
            
            # 计算投影矩阵 C = V @ V^T
            C_combined = None
            for dim in preference_dimensions:
                V = constraint.manager.get_subspace(dim, layer_id=None)
                C = V @ V.T  # (hidden_dim, hidden_dim)
                if C_combined is None:
                    C_combined = C
                else:
                    C_combined = C_combined @ C  # 多个维度取交集
            
            # 注入硬约束到模型
            patched_count = inject_hard_constraint_to_model(
                model=model,
                lora_C=C_combined,
                verbose=True
            )
            
            print(f"✅ 硬约束注入完成: {patched_count} 个 LoRA 层")
            print(f"   约束公式: output = base(x) + (LoRA(x) @ C^T)")
            print(f"   C = V @ V^T (固定不训练)")
            
            # 硬约束模式下不需要软约束
            use_orthogonal = False
            constraint = None
    
    elif use_orthogonal:
        # 📊 软约束模式 (参数空间)
        if subspace_dir is None or preference_dimensions is None:
            print("⚠️  警告: use_orthogonal=True 但未提供 subspace_dir 或 preference_dimensions")
            print("⚠️  将关闭正交约束")
            use_orthogonal = False
        else:
            constraint = load_subspace_constraint(
                subspace_dir=subspace_dir,
                dimensions=preference_dimensions,
                device=device,
                subspace_rank=subspace_rank
            )
    
    # 4. 设置训练参数
    print(f"\n⚙️  设置训练参数")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        lr_scheduler_type='constant',  # 使用恒定学习率，避免约束失效
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        eval_strategy='steps' if eval_dataset else 'no',
        eval_steps=100 if eval_dataset else None,
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=use_gradient_checkpointing,  # ✅ 可配置
        report_to='none',
        remove_unused_columns=False
    )
    
    # 5. 创建 Trainer
    print(f"\n🏋️  创建 Trainer")

    # 准备 SwanLab 配置
    swanlab_config = {
        'model_name': model_name,
        'lora_rank': lora_rank,
        'lora_alpha': lora_alpha,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'use_orthogonal': use_orthogonal,
        'lambda_orth': lambda_orth if use_orthogonal else None,
        'gpu_id': gpu_id,
    }
    trainer = OrthogonalLoRATrainer(
        constraint_calculator=constraint,
        lambda_orth=lambda_orth,
        dimension_weights=dimension_weights,
        use_orthogonal=use_orthogonal,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    # 6. 训练

    # 添加训练监控回调
    callbacks = create_training_callbacks(
        trainer=trainer,
        use_swanlab=use_swanlab,
        swanlab_project=swanlab_project,
        swanlab_experiment=experiment_name,
        swanlab_config=swanlab_config,
        print_interval=print_interval,
        enable_console_logging=enable_console_logging,
        monitor_orth_loss=use_orthogonal,
    )
    
    for callback in callbacks:
        trainer.add_callback(callback)


    trainer.train()
    
    # 7. 保存
    print(f"\n💾 保存模型")
    trainer.save_model(output_dir)
    if use_orthogonal:
        trainer.save_orth_loss_history(output_dir)
    
    print(f"\n" + "="*80)
    print(f"✅ 训练完成!")
    print(f"📁 输出目录: {output_dir}")
    print("="*80)
    
    return trainer


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='主训练文件 v2')
    
    # 基础配置
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-1.5B-Instruct',
                        help='模型名称或路径')
    parser.add_argument('--output_dir', type=str, default='./output/train_v2',
                        help='输出目录')
    parser.add_argument('--gpu_id', type=int, default=1,
                        help='GPU ID')
    
    # LoRA 配置
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help='LoRA dropout')
    
    # 训练配置
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--gradient_accumulation', type=int, default=4,
                        help='梯度累积步数')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--max_length', type=int, default=512,
                        help='最大序列长度')
    parser.add_argument('--use_gradient_checkpointing', action='store_true',
                        help='使用梯度检查点')
    
    # 正交约束
    parser.add_argument('--use_orthogonal', action='store_true',
                        help='使用软约束（参数空间）')
    parser.add_argument('--use_hard_constraint', action='store_true',
                        help='🔑 使用硬约束（SaLoRA 风格，表征空间）')
    parser.add_argument('--subspace_dir', type=str,
                        default='../preference_subspace/output/qwen2.5_1.5b',
                        help='子空间目录')
    parser.add_argument('--preference_dimensions', type=str, nargs='+',
                        default=['safety', 'helpfulness'],
                        help='偏好维度')
    parser.add_argument('--lambda_orth', type=float, default=0.01,
                        help='正交损失系数')
    parser.add_argument('--subspace_rank', type=int, default=None,
                        help='使用子空间的前 k 个向量（None=使用全部）')
    
    # 数据
    parser.add_argument('--use_demo_data', action='store_true', default=True,
                        help='使用演示数据')

    # 训练监控参数
    
    # 数据集配置
    parser.add_argument('--dataset_type', type=str, default='demo',
                        choices=['demo', 'ultrafeedback'],
                        help='数据集类型')
    parser.add_argument('--dataset_size', type=str, default='100',
                        choices=['100', '1k', '3k', '1w', 'full'],
                        help='UltraFeedback 数据集大小')
    parser.add_argument('--data_format', type=str, default='instruction',
                        choices=['instruction', 'conversation'],
                        help='数据格式化类型')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='限制样本数量（用于快速测试）')

    parser.add_argument('--use_swanlab', type=lambda x: x.lower() == 'true', default=True,
                        help='使用 SwanLab 进行实验追踪 (true/false)')
    parser.add_argument('--swanlab_project', type=str, default='protected-lora',
                        help='SwanLab 项目名称')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='训练实验名称，用于 SwanLab 记录（可选，默认自动生成）')
    parser.add_argument('--print_interval', type=int, default=10,
                        help='终端打印间隔（步数）')
    parser.add_argument('--disable_console_log', action='store_true',
                        help='禁用终端详细日志')
    
    args = parser.parse_args()
    
    # 准备数据
    # 准备数据
    if args.dataset_type == 'demo':
        print("📚 使用演示数据")
        train_texts = [
            "What is machine learning? Machine learning is a subset of artificial intelligence.",
            "Explain neural networks. Neural networks are computing systems inspired by biology.",
            "What is deep learning? Deep learning uses neural networks with multiple layers.",
            "Describe NLP. Natural language processing helps computers understand human language.",
            "What is reinforcement learning? RL trains agents through rewards and penalties.",
            "Explain computer vision. CV enables computers to derive information from images.",
            "What is supervised learning? Supervised learning uses labeled training data.",
            "Describe unsupervised learning. Unsupervised learning finds patterns in unlabeled data.",
        ] * 20  # 160 样本
        eval_texts = train_texts[:10]
    
    elif args.dataset_type == 'ultrafeedback':
        print(f"📚 加载 UltraFeedback 数据集 (大小: {args.dataset_size})")
        
        # 这里先不实际加载，在 train() 函数中加载
        # 因为需要 tokenizer
        train_texts = None  # 标记为使用数据集
        eval_texts = None
    
    else:
        raise ValueError(f"不支持的数据集类型: {args.dataset_type}")
    
    # 训练
    trainer = train(
        model_name=args.model_name,
        train_texts=train_texts,
        eval_texts=eval_texts,
        
        # 数据集配置
        dataset_type=args.dataset_type,
        dataset_size=args.dataset_size,
        data_format=args.data_format,
        max_samples=args.max_samples,
        
        output_dir=args.output_dir,
        gpu_id=args.gpu_id,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_orthogonal=args.use_orthogonal,
        use_hard_constraint=args.use_hard_constraint,  # 🔑 新增
        subspace_dir=args.subspace_dir if (args.use_orthogonal or args.use_hard_constraint) else None,
        preference_dimensions=args.preference_dimensions if (args.use_orthogonal or args.use_hard_constraint) else None,
        lambda_orth=args.lambda_orth,
        subspace_rank=args.subspace_rank if (args.use_orthogonal or args.use_hard_constraint) else None,
        # 训练监控
        experiment_name=args.experiment_name,
        use_swanlab=args.use_swanlab,
        swanlab_project=args.swanlab_project,
        print_interval=args.print_interval,
        enable_console_logging=not args.disable_console_log
    )
    
    print(f"\n📊 训练统计:")
    print(f"  总步数: {trainer.state.global_step}")
    if trainer.state.log_history:
        print(f"  最终损失: {trainer.state.log_history[-1].get('loss', 'N/A')}")


if __name__ == '__main__':
    main()