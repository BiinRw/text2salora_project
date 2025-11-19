"""
显存优化版本的训练示例 (适配 24GB 显存)
主要优化:
1. 避免预计算完整投影矩阵 P = V @ V^T
2. 使用 8-bit 量化加载模型
3. 减小 batch size 和 sequence length
4. 仅约束 q_proj 和 v_proj
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import sys
from pathlib import Path
import gc

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))

from utils.svd_utils import PreferenceSubspaceManager
from protected_lora.train_lora_orthogonal import OrthogonalLoRATrainer


class MemoryEfficientOrthogonalConstraint:
    """显存优化的正交约束计算器
    
    不预计算 P = V @ V^T,而是在需要时动态计算 V @ (V^T @ x)
    """
    
    def __init__(
        self,
        subspace_manager: PreferenceSubspaceManager,
        dimensions: list,
        use_fused: bool = True,
        device: str = 'cuda:0'
    ):
        self.manager = subspace_manager
        self.dimensions = dimensions
        self.use_fused = use_fused
        self.device = device
        
        # 只存储 V 矩阵,不计算 P
        self.subspace_V = {}
        for dim in dimensions:
            V = self.manager.get_subspace(dim, layer_id=None if use_fused else 0)
            self.subspace_V[dim] = V
            print(f"   {dim}: V shape={V.shape} (不预计算 P,节省显存)")
    
    def compute_orthogonal_loss_efficient(
        self,
        lora_A: dict,
        lora_B: dict,
        lambda_orth: float = 0.1,
        dimension_weights: dict = None
    ):
        """显存优化的正交损失计算
        
        原始: BAP = B @ A @ (V @ V^T)
        优化: BAP = B @ A @ V @ V^T = (B @ A @ V) @ V^T
        
        进一步优化: 
        L = ||BAP||² = ||B @ A @ V @ V^T||²
          = trace((B @ A @ V @ V^T) @ (B @ A @ V @ V^T)^T)
          = trace((B @ A @ V) @ V^T @ V @ (B @ A @ V)^T)
          = trace((B @ A @ V) @ (B @ A @ V)^T)  (因为 V^T @ V = I)
          = ||B @ A @ V||²
        """
        if dimension_weights is None:
            dimension_weights = {dim: 1.0 for dim in self.dimensions}
        
        total_loss = 0.0
        loss_details = {}
        
        for dim in self.dimensions:
            dim_weight = dimension_weights.get(dim, 1.0)
            dim_loss = 0.0
            
            V = self.subspace_V[dim]  # (d, k)
            
            for layer_name in lora_A.keys():
                A = lora_A[layer_name]  # (rank, d)
                B = lora_B[layer_name]  # (out, rank)
                
                # 计算 B @ A @ V
                AV = A @ V  # (rank, d) @ (d, k) = (rank, k)
                BAV = B @ AV  # (out, rank) @ (rank, k) = (out, k)
                
                # ||BAV||²
                loss_term = torch.sum(BAV ** 2)
                dim_loss += loss_term
            
            dim_loss = dim_weight * dim_loss
            loss_details[dim] = dim_loss.item()
            total_loss += dim_loss
        
        total_loss = lambda_orth * total_loss
        
        return total_loss, loss_details


class MemoryEfficientTrainer(OrthogonalLoRATrainer):
    """显存优化的 Trainer"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """计算损失,定期清理显存"""
        
        # 1. 计算任务损失
        outputs = model(**inputs)
        
        if isinstance(outputs, dict):
            task_loss = outputs.get('loss')
        else:
            task_loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
        
        # 2. 计算正交损失
        if model.training:
            lora_A = {}
            lora_B = {}
            
            for name, module in model.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    lora_A[name] = module.lora_A['default'].weight
                    lora_B[name] = module.lora_B['default'].weight
            
            if len(lora_A) > 0:
                orth_loss, loss_details = self.constraint.compute_orthogonal_loss_efficient(
                    lora_A, lora_B,
                    lambda_orth=self.lambda_orth,
                    dimension_weights=self.dimension_weights
                )
                
                total_loss = task_loss + orth_loss
                
                # 记录
                if self.state.global_step % 10 == 0:
                    self.orth_loss_history.append({
                        'step': self.state.global_step,
                        'task_loss': task_loss.item(),
                        'orth_loss': orth_loss.item(),
                        'details': loss_details
                    })
            else:
                total_loss = task_loss
        else:
            total_loss = task_loss
        
        # 定期清理显存
        if self.state.global_step % 20 == 0:
            torch.cuda.empty_cache()
        
        return (total_loss, outputs) if return_outputs else total_loss


def main():
    # ═══════════════════════════════════════════════════════════════
    # 配置参数 (显存优化)
    # ═══════════════════════════════════════════════════════════════
    
    MODEL_PATH = '/var/models/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/f347a08eb880e0a3c87089c8c45043775c338c9c'
    SUBSPACE_DIR = '../preference_subspace/saved_subspaces'
    OUTPUT_DIR = './output/lora_orthogonal_24gb'
    
    # 偏好约束配置
    PREFERENCE_DIMENSIONS = ['safety', 'helpfulness']
    LAMBDA_ORTH = 0.1
    DIMENSION_WEIGHTS = {
        'safety': 1.0,
        'helpfulness': 0.5
    }
    
    # LoRA 配置 (显存优化)
    LORA_CONFIG = {
        'rank': 8,
        'alpha': 16,
        'dropout': 0.1,
        'target_modules': ['q_proj', 'v_proj']  # 只约束 QV,节省显存
    }
    
    print("=" * 70)
    print("🚀 显存优化训练: 正交约束 LoRA (24GB 显存)")
    print("=" * 70)
    
    # ═══════════════════════════════════════════════════════════════
    # 步骤 1: 加载偏好子空间
    # ═══════════════════════════════════════════════════════════════
    print("\n📦 步骤 1: 加载偏好子空间")
    
    manager = PreferenceSubspaceManager(
        subspace_dir=SUBSPACE_DIR,
        device='cuda:0'
    )
    
    manager.load_all_dimensions(
        dimensions=PREFERENCE_DIMENSIONS,
        use_fused=True
    )
    
    manager.print_info()
    
    # ═══════════════════════════════════════════════════════════════
    # 步骤 2: 创建显存优化的约束计算器
    # ═══════════════════════════════════════════════════════════════
    print("\n🔧 步骤 2: 创建显存优化的约束计算器")
    
    constraint = MemoryEfficientOrthogonalConstraint(
        subspace_manager=manager,
        dimensions=PREFERENCE_DIMENSIONS,
        use_fused=True,
        device='cuda:0'
    )
    
    print(f"✅ 约束计算器创建完成 (不预计算 P 矩阵)")
    
    # 清理显存
    gc.collect()
    torch.cuda.empty_cache()
    
    # ═══════════════════════════════════════════════════════════════
    # 步骤 3: 加载模型 (使用 float16)
    # ═══════════════════════════════════════════════════════════════
    print("\n🤖 步骤 3: 加载模型 (float16 优化)")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map='auto',  # 自动分配显存
        low_cpu_mem_usage=True
    )
    
    print(f"✅ 基座模型加载完成 (float16)")
    
    # ═══════════════════════════════════════════════════════════════
    # 步骤 4: 应用 LoRA
    # ═══════════════════════════════════════════════════════════════
    print("\n🔄 步骤 4: 应用 LoRA")
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_CONFIG['rank'],
        lora_alpha=LORA_CONFIG['alpha'],
        lora_dropout=LORA_CONFIG['dropout'],
        target_modules=LORA_CONFIG['target_modules'],
        bias='none'
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 清理显存
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"✅ LoRA 应用完成")
    print(f"📊 显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / 24 GB")
    
    # ═══════════════════════════════════════════════════════════════
    # 步骤 5: 准备训练数据 (小数据集)
    # ═══════════════════════════════════════════════════════════════
    print("\n📚 步骤 5: 准备训练数据")
    
    test_texts = [
        "What is artificial intelligence?",
        "Explain machine learning.",
        "How does a neural network work?",
        "What is deep learning?",
        "Describe natural language processing."
    ] * 4  # 20 个样本
    
    from torch.utils.data import Dataset
    
    class SimpleDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length=256):  # 减小 max_length
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
    
    train_dataset = SimpleDataset(test_texts, tokenizer, max_length=256)
    eval_dataset = SimpleDataset(test_texts[:3], tokenizer, max_length=256)
    
    print(f"✅ 数据集准备完成")
    print(f"   • 训练样本: {len(train_dataset)}")
    print(f"   • 验证样本: {len(eval_dataset)}")
    print(f"   • Max Length: 256 (节省显存)")
    
    # ═══════════════════════════════════════════════════════════════
    # 步骤 6: 设置训练参数 (显存优化)
    # ═══════════════════════════════════════════════════════════════
    print("\n⚙️  步骤 6: 设置训练参数 (显存优化)")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=1,  # 减小 batch size
        gradient_accumulation_steps=8,   # 增加梯度累积
        learning_rate=5e-5,
        warmup_steps=10,
        logging_steps=2,
        save_steps=20,
        eval_steps=20,
        evaluation_strategy='steps',
        save_total_limit=1,
        fp16=True,
        max_grad_norm=1.0,
        gradient_checkpointing=True,  # 启用梯度检查点
        optim='adamw_torch',
        report_to='none'
    )
    
    print(f"✅ 训练参数设置完成")
    print(f"   • Batch Size: 1")
    print(f"   • Gradient Accumulation: 8")
    print(f"   • Effective Batch Size: 8")
    print(f"   • Gradient Checkpointing: True")
    
    # ═══════════════════════════════════════════════════════════════
    # 步骤 7: 创建 Trainer 并训练
    # ═══════════════════════════════════════════════════════════════
    print("\n🏃 步骤 7: 创建 Trainer 并开始训练")
    
    trainer = MemoryEfficientTrainer(
        constraint_calculator=constraint,
        lambda_orth=LAMBDA_ORTH,
        dimension_weights=DIMENSION_WEIGHTS,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
    )
    
    print(f"✅ Trainer 创建完成")
    print(f"📊 训练前显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / 24 GB")
    print(f"\n{'=' * 70}")
    print("开始训练...")
    print(f"{'=' * 70}\n")
    
    # 开始训练
    try:
        trainer.train()
        
        print(f"\n💾 保存结果")
        trainer.save_model(OUTPUT_DIR)
        trainer.save_orth_loss_history(OUTPUT_DIR)
        
        print(f"\n{'=' * 70}")
        print(f"✅ 训练完成!")
        print(f"{'=' * 70}")
        print(f"模型保存路径: {OUTPUT_DIR}")
        print(f"正交损失历史: {OUTPUT_DIR}/orth_loss_history.json")
        print(f"📊 最终显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / 24 GB")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n❌ 显存不足错误: {e}")
            print(f"\n建议:")
            print(f"  1. 进一步减小 batch_size 或 max_length")
            print(f"  2. 减少 LoRA target_modules (只用 ['q_proj'])")
            print(f"  3. 减小子空间维度 (重新运行 compute_svd.py --top_k 32)")
            print(f"  4. 只约束一个偏好维度 (PREFERENCE_DIMENSIONS = ['safety'])")
        raise


if __name__ == '__main__':
    main()
