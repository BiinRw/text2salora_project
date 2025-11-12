"""
数据集加载模块
支持加载 UltraFeedback Binarized 偏好数据集
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from torch.utils.data import Dataset
import random


class PreferenceDataset(Dataset):
    """偏好数据集 - 加载 chosen/rejected 对"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        use_chosen_only: bool = True,  # 只用 chosen 做 SFT
        format_type: str = "instruction"  # instruction 或 conversation
    ):
        """
        Args:
            data_path: JSONL 数据文件路径
            tokenizer: tokenizer
            max_length: 最大序列长度
            use_chosen_only: 是否只使用 chosen 响应（用于 SFT）
            format_type: 格式化类型
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_chosen_only = use_chosen_only
        self.format_type = format_type
        
        # 加载数据
        self.data = self._load_jsonl(data_path)
        print(f"✅ 加载了 {len(self.data)} 条数据")
    
    def _load_jsonl(self, path: str) -> List[Dict]:
        """加载 JSONL 文件"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)
        return data
    
    def _format_instruction(self, prompt: str, response: str) -> str:
        """格式化为指令格式"""
        return f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
    
    def _format_conversation(self, prompt: str, response: str) -> str:
        """格式化为对话格式"""
        # Qwen 格式
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    
    def format_text(self, prompt: str, response: str) -> str:
        """格式化文本"""
        if self.format_type == "instruction":
            return self._format_instruction(prompt, response)
        elif self.format_type == "conversation":
            return self._format_conversation(prompt, response)
        else:
            return f"{prompt}\n{response}"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        
        # 选择响应
        if self.use_chosen_only:
            response = item['chosen']
        else:
            # 随机选择 chosen 或 rejected（用于对比学习）
            response = random.choice([item['chosen'], item['rejected']])
        
        # 格式化文本
        text = self.format_text(prompt, response)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0)
        }


class UltraFeedbackLoader:
    """UltraFeedback 数据集加载器"""
    
    def __init__(self, base_dir: str = "../datasets/ultrafeedback_binarized"):
        self.base_dir = Path(base_dir)
        if not self.base_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {self.base_dir}")
    
    def get_available_datasets(self) -> Dict[str, Path]:
        """获取可用的数据集"""
        datasets = {}
        
        # 训练集
        train_files = {
            'train_100': 'train_prefs_ultrafeedback_binarized_100.jsonl',
            'train_1k': 'train_prefs_ultrafeedback_binarized_1k.jsonl',
            'train_3k': 'train_prefs_ultrafeedback_binarized_3k.jsonl',
            'train_1w': 'train_prefs_ultrafeedback_binarized_1w.jsonl',
            'train_full': 'train_prefs_ultrafeedback_binarized.jsonl',
        }
        
        # 测试集
        test_files = {
            'test_100': 'test_prefs_ultrafeedback_binarized_100.jsonl',
            'test_full': 'test_prefs_ultrafeedback_binarized.jsonl',
        }
        
        all_files = {**train_files, **test_files}
        
        for name, filename in all_files.items():
            path = self.base_dir / filename
            if path.exists():
                datasets[name] = path
        
        return datasets
    
    def load_dataset(
        self,
        dataset_name: str,
        tokenizer,
        max_length: int = 512,
        use_chosen_only: bool = True,
        format_type: str = "instruction"
    ) -> PreferenceDataset:
        """加载指定的数据集
        
        Args:
            dataset_name: 数据集名称 (train_100, train_1k, test_100, 等)
            tokenizer: tokenizer
            max_length: 最大序列长度
            use_chosen_only: 是否只使用 chosen 响应
            format_type: 格式化类型
        """
        available = self.get_available_datasets()
        
        if dataset_name not in available:
            raise ValueError(
                f"数据集 '{dataset_name}' 不存在。可用的数据集:\n" + 
                "\n".join(f"  - {name}" for name in available.keys())
            )
        
        data_path = available[dataset_name]
        print(f"📚 加载数据集: {dataset_name}")
        print(f"   路径: {data_path}")
        
        dataset = PreferenceDataset(
            data_path=str(data_path),
            tokenizer=tokenizer,
            max_length=max_length,
            use_chosen_only=use_chosen_only,
            format_type=format_type
        )
        
        return dataset
    
    def print_sample(self, dataset_name: str, num_samples: int = 2):
        """打印数据样本（不需要tokenizer）"""
        available = self.get_available_datasets()
        if dataset_name not in available:
            print(f"❌ 数据集 '{dataset_name}' 不存在")
            return
        
        data_path = available[dataset_name]
        
        print(f"\n{'='*80}")
        print(f"数据集: {dataset_name}")
        print(f"路径: {data_path}")
        print(f"{'='*80}\n")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                
                item = json.loads(line.strip())
                print(f"样本 {i+1}:")
                print(f"  Prompt: {item['prompt'][:100]}...")
                print(f"  Chosen length: {len(item['chosen'])} chars")
                print(f"  Rejected length: {len(item['rejected'])} chars")
                print()


def load_ultrafeedback_data(
    dataset_size: str = "100",  # "100", "1k", "3k", "1w", "full"
    tokenizer = None,
    max_length: int = 512,
    use_chosen_only: bool = True,
    format_type: str = "instruction",
    split: str = "train"  # "train" or "test"
) -> Tuple[PreferenceDataset, Optional[PreferenceDataset]]:
    """便捷函数：加载 UltraFeedback 数据
    
    Args:
        dataset_size: 数据集大小 ("100", "1k", "3k", "1w", "full")
        tokenizer: tokenizer
        max_length: 最大序列长度
        use_chosen_only: 是否只使用 chosen 响应
        format_type: 格式化类型
        split: "train", "test", 或 "both"
    
    Returns:
        (train_dataset, test_dataset) 如果 split="both"
        train_dataset 如果 split="train"
        test_dataset 如果 split="test"
    """
    loader = UltraFeedbackLoader()
    
    train_dataset = None
    test_dataset = None
    
    if split in ["train", "both"]:
        train_name = f"train_{dataset_size}"
        train_dataset = loader.load_dataset(
            train_name, tokenizer, max_length, use_chosen_only, format_type
        )
    
    if split in ["test", "both"]:
        test_name = "test_100" if dataset_size in ["100", "1k"] else "test_full"
        test_dataset = loader.load_dataset(
            test_name, tokenizer, max_length, use_chosen_only, format_type
        )
    
    if split == "both":
        return train_dataset, test_dataset
    elif split == "train":
        return train_dataset
    else:
        return test_dataset


# 命令行测试
if __name__ == '__main__':
    print("UltraFeedback 数据集加载器测试\n")
    
    loader = UltraFeedbackLoader()
    
    print("📋 可用的数据集:")
    available = loader.get_available_datasets()
    for name, path in available.items():
        size = path.stat().st_size / (1024 * 1024)  # MB
        print(f"  - {name:20s} ({size:.1f} MB)")
    
    print("\n" + "="*80)
    print("查看样本数据:")
    print("="*80)
    
    # 打印一些样本
    loader.print_sample("train_100", num_samples=2)
