"""
完整的训练示例
展示如何使用正交约束训练 LoRA
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))

from utils.svd_utils import PreferenceSubspaceManager
from protected_lora.orthogonal_constraints import OrthogonalConstraint
from protected_lora.train_lora_orthogonal import OrthogonalLoRATrainer


def main():
    # ═══════════════════════════════════════════════════════════════
    # 配置参数
    # ═══════════════════════════════════════════════════════════════
    
    MODEL_PATH = 'Qwen/Qwen2.5-1.5B-Instruct'
    SUBSPACE_DIR = '../preference_subspace/saved_subspaces'
    OUTPUT_DIR = './output/lora_with_orthogonal_constraint'
    
    # 偏好约束配置
    PREFERENCE_DIMENSIONS = ['safety', 'helpfulness']  # 要保护的偏好
    LAMBDA_ORTH = 0.1  # 正交约束系数 (0.01~0.5)
    DIMENSION_WEIGHTS = {
        'safety': 1.0,      # safety 权重最高
        'helpfulness': 0.5  # helpfulness 权重较低
    }
    
    # LoRA 配置
    LORA_CONFIG = {
        'rank': 8,
        'alpha': 16,
        'dropout': 0.1,
        'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj']
    }
    
    print("=" * 70)
    print("🚀 开始训练: 正交约束 LoRA")
    print("=" * 70)
    
    # ═══════════════════════════════════════════════════════════════
    # 步骤 1: 加载偏好子空间
    # ═══════════════════════════════════════════════════════════════
    print("\n📦 步骤 1: 加载偏好子空间")
    
    manager = PreferenceSubspaceManager(
        subspace_dir=SUBSPACE_DIR,
        device='cuda:1'
    )
    
    manager.load_all_dimensions(
        dimensions=PREFERENCE_DIMENSIONS,
        use_fused=True  # 使用融合子空间
    )
    
    manager.print_info()
    
    # ═══════════════════════════════════════════════════════════════
    # 步骤 2: 创建正交约束计算器
    # ═══════════════════════════════════════════════════════════════
    print("\n🔧 步骤 2: 创建正交约束计算器")
    
    constraint = OrthogonalConstraint(
        subspace_manager=manager,
        dimensions=PREFERENCE_DIMENSIONS,
        use_fused=True,
        device='cuda:0'
    )
    
    print(f"✅ 正交约束计算器创建完成")
    print(f"   • 约束维度: {PREFERENCE_DIMENSIONS}")
    print(f"   • Lambda: {LAMBDA_ORTH}")
    print(f"   • 维度权重: {DIMENSION_WEIGHTS}")
    
    # ═══════════════════════════════════════════════════════════════
    # 步骤 3: 加载模型和 tokenizer
    # ═══════════════════════════════════════════════════════════════
    print("\n🤖 步骤 3: 加载模型和 tokenizer")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map='cuda:0'
    )
    
    print(f"✅ 基座模型加载完成")
    
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
    
    # ═══════════════════════════════════════════════════════════════
    # 步骤 5: 准备训练数据
    # ═══════════════════════════════════════════════════════════════
    print("\n📚 步骤 5: 准备训练数据")
    
    # 这里需要你自己准备数据
    # 示例: 使用 Hugging Face 数据集
    # dataset = load_dataset('your_dataset_name')
    
    # 临时示例: 创建一个小的测试数据集
    print("⚠️  警告: 使用测试数据集,实际训练需要替换为真实数据")
    
    test_texts = [
        "What is artificial intelligence?",
        "Explain machine learning.",
        "How does a neural network work?"
    ] * 10  # 重复以增加数据量
    
    def tokenize_function(examples):
        return tokenizer(
            examples,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
    
    # 创建简单数据集
    from torch.utils.data import Dataset
    
    class SimpleDataset(Dataset):
        def __init__(self, texts, tokenizer):
            self.encodings = tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=512,
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
    
    train_dataset = SimpleDataset(test_texts, tokenizer)
    eval_dataset = SimpleDataset(test_texts[:5], tokenizer)
    
    print(f"✅ 数据集准备完成")
    print(f"   • 训练样本: {len(train_dataset)}")
    print(f"   • 验证样本: {len(eval_dataset)}")
    
    # ═══════════════════════════════════════════════════════════════
    # 步骤 6: 设置训练参数
    # ═══════════════════════════════════════════════════════════════
    print("\n⚙️  步骤 6: 设置训练参数")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_steps=50,
        logging_steps=5,
        save_steps=50,
        eval_steps=50,
        evaluation_strategy='steps',
        save_total_limit=2,
        fp16=True,
        max_grad_norm=1.0,
        report_to='none'  # 不上报到 wandb 等
    )
    
    print(f"✅ 训练参数设置完成")
    
    # ═══════════════════════════════════════════════════════════════
    # 步骤 7: 创建 Trainer 并开始训练
    # ═══════════════════════════════════════════════════════════════
    print("\n🏃 步骤 7: 创建 Trainer 并开始训练")
    
    trainer = OrthogonalLoRATrainer(
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
    print(f"\n{'=' * 70}")
    print("开始训练...")
    print(f"{'=' * 70}\n")
    
    # 开始训练
    trainer.train()
    
    # ═══════════════════════════════════════════════════════════════
    # 步骤 8: 保存模型和历史
    # ═══════════════════════════════════════════════════════════════
    print("\n💾 步骤 8: 保存结果")
    
    trainer.save_model(OUTPUT_DIR)
    trainer.save_orth_loss_history(OUTPUT_DIR)
    
    print(f"\n{'=' * 70}")
    print(f"✅ 训练完成!")
    print(f"{'=' * 70}")
    print(f"模型保存路径: {OUTPUT_DIR}")
    print(f"正交损失历史: {OUTPUT_DIR}/orth_loss_history.json")
    print(f"\n下一步:")
    print(f"  1. 查看正交损失历史")
    print(f"  2. 使用 eval_preference_retention.py 评估偏好保留")
    print(f"  3. 对比有/无约束的效果差异")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    main()
