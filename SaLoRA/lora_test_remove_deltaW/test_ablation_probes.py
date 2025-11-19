"""
LoRA △W消融测试 - 探针准确率版本

功能：测试不同层的LoRA△W禁用对探针准确率的影响
用途：验证表示可分离性 ≠ 行为安全性理论

使用方法：
python test_ablation_probes.py \
    --lora_path <lora_checkpoint_path> \
    --probe_path <probe_model_path> \
    --output_dir ./results/ablation_probes
"""

import torch
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入消融测试器
sys.path.insert(0, str(Path(__file__).parent))
from lora_ablation_tester import LoRAAblusionTester


class ProbeAccuracyEvaluator:
    """探针准确率评估器"""
    
    def __init__(self, model, probe_model, device: str = 'cuda:0'):
        """
        初始化探针准确率评估器
        
        Args:
            model: 主模型
            probe_model: 探针分类器
            device: 设备
        """
        self.model = model
        self.probe_model = probe_model
        self.device = device
        self.probe_model.eval()
        self.probe_model.to(device)
    
    def get_model_hidden_states(self, test_prompts: List[str], layer_id: int = -1) -> torch.Tensor:
        """
        获取模型隐层表示
        
        Args:
            test_prompts: 测试提示列表
            layer_id: 层ID (-1表示最后一层)
            
        Returns:
            隐层张量 [batch_size, hidden_dim]
        """
        try:
            # 这里应该实现从模型获取隐层输出的逻辑
            # 示例：使用hook机制
            hidden_states = []
            
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states.append(output[0].detach().cpu())
                else:
                    hidden_states.append(output.detach().cpu())
            
            # 注册hook
            target_layer = list(self.model.modules())[layer_id]
            handle = target_layer.register_forward_hook(hook_fn)
            
            # 前向传播
            with torch.no_grad():
                _ = self.model(test_prompts)
            
            # 移除hook
            handle.remove()
            
            if hidden_states:
                return torch.cat(hidden_states, dim=0)
            else:
                return torch.zeros(len(test_prompts), 1024)
        
        except Exception as e:
            logger.error(f"❌ 获取隐层失败: {e}")
            return torch.zeros(len(test_prompts), 1024)
    
    def evaluate_probe_accuracy(self, test_prompts: List[str], test_labels: List[int]) -> float:
        """
        评估探针准确率
        
        Args:
            test_prompts: 测试提示列表
            test_labels: 测试标签列表
            
        Returns:
            准确率 (0.0-1.0)
        """
        try:
            # 获取最后一层隐层表示
            hidden_states = self.get_model_hidden_states(test_prompts, layer_id=-1)
            
            # 用探针分类器评估
            with torch.no_grad():
                hidden_states = hidden_states.to(self.device)
                logits = self.probe_model(hidden_states)
                predictions = torch.argmax(logits, dim=1).cpu().numpy()
            
            # 计算准确率
            test_labels = torch.tensor(test_labels).numpy()
            accuracy = (predictions == test_labels).sum() / len(test_labels)
            
            return float(accuracy)
        
        except Exception as e:
            logger.error(f"❌ 探针准确率评估失败: {e}")
            return 0.0


def run_ablation_experiment(args):
    """运行探针消融实验"""
    
    logger.info("=" * 70)
    logger.info("🧪 LoRA △W 消融测试 - 探针准确率版本")
    logger.info("=" * 70)
    
    # 第1步: 加载模型
    logger.info("\n[1] 加载模型...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map=args.device
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        logger.info("   ✅ 模型加载成功")
    except Exception as e:
        logger.error(f"   ❌ 模型加载失败: {e}")
        return
    
    # 第2步: 加载探针模型
    logger.info("\n[2] 加载探针模型...")
    try:
        probe_model = torch.load(args.probe_path)
        logger.info("   ✅ 探针模型加载成功")
    except Exception as e:
        logger.error(f"   ❌ 探针模型加载失败: {e}")
        return
    
    # 第3步: 初始化消融测试器
    logger.info("\n[3] 初始化消融测试器...")
    try:
        tester = LoRAAblusionTester(model, num_layers=28, device=args.device)
        tester.save_lora_weights()
        logger.info("   ✅ 消融测试器初始化完成")
    except Exception as e:
        logger.error(f"   ❌ 消融测试器初始化失败: {e}")
        return
    
    # 第4步: 初始化探针评估器
    logger.info("\n[4] 初始化探针评估器...")
    try:
        evaluator = ProbeAccuracyEvaluator(model, probe_model, device=args.device)
        logger.info("   ✅ 探针评估器初始化完成")
    except Exception as e:
        logger.error(f"   ❌ 探针评估器初始化失败: {e}")
        return
    
    # 第5步: 定义消融配置
    logger.info("\n[5] 定义消融配置...")
    ablation_configs = {
        'baseline': [],
        'disable_layer_16': [16],
        'disable_layers_0_8': list(range(0, 9)),
        'disable_layers_8_16': list(range(8, 17)),
        'disable_layers_17_27': list(range(17, 28)),
    }
    logger.info(f"   定义了 {len(ablation_configs)} 个配置")
    
    # 第6步: 准备测试数据
    logger.info("\n[6] 准备测试数据...")
    # 这里应该从真实数据集加载
    test_prompts = ["test prompt"] * 10
    test_labels = [0, 1] * 5  # 二分类示例
    logger.info(f"   测试数据: {len(test_prompts)} 个样本")
    
    # 第7步: 运行消融循环
    logger.info("\n[7] 运行消融实验...")
    results = {}
    
    for config_idx, (config_name, disabled_layers) in enumerate(ablation_configs.items(), 1):
        logger.info(f"\n   [{config_idx}/{len(ablation_configs)}] {config_name}")
        
        try:
            # 禁用LoRA
            tester.disable_lora_on_layers(disabled_layers)
            
            # 评估探针准确率
            probe_acc = evaluator.evaluate_probe_accuracy(test_prompts, test_labels)
            logger.info(f"      探针准确率: {probe_acc:.4f}")
            
            results[config_name] = {
                'disabled_layers': disabled_layers,
                'probe_accuracy': probe_acc,
                'timestamp': datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"      ✗ 错误: {e}")
            results[config_name] = {'error': str(e)}
        
        # 恢复LoRA
        tester.restore_lora_on_layers(disabled_layers)
    
    # 第8步: 保存结果
    logger.info("\n[8] 保存结果...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    full_results = {
        'experiment': 'lora_ablation_probes',
        'model': args.model_path,
        'timestamp': datetime.now().isoformat(),
        'ablation_results': results,
    }
    
    results_file = output_dir / "ablation_probe_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    logger.info(f"   ✅ 结果保存到: {results_file}")
    
    # 第9步: 打印总结
    logger.info("\n" + "=" * 70)
    logger.info("📊 探针准确率测试结果")
    logger.info("=" * 70 + "\n")
    
    print(f"{'配置':<30} | {'探针准确率':<12} | {'相对基准':<12}")
    print("-" * 60)
    
    baseline_acc = results.get('baseline', {}).get('probe_accuracy', 0)
    for config_name, result in sorted(results.items()):
        if 'error' in result:
            print(f"{config_name:<30} | ERROR")
            continue
        
        acc = result['probe_accuracy']
        if baseline_acc > 0:
            ratio = f"{(acc - baseline_acc) / baseline_acc * 100:+.1f}%"
        else:
            ratio = "N/A"
        
        print(f"{config_name:<30} | {acc:12.4f} | {ratio:>12s}")
    
    logger.info("\n✅ 实验完成!\n")


def main():
    parser = argparse.ArgumentParser(description='LoRA消融测试 - 探针准确率版本')
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-1.5B-Instruct',
                        help='基础模型路径')
    parser.add_argument('--lora_path', type=str, required=True,
                        help='LoRA适配器路径')
    parser.add_argument('--probe_path', type=str, required=True,
                        help='探针模型路径')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU设备号')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='PyTorch设备')
    parser.add_argument('--output_dir', type=str, default='./results/ablation_probes',
                        help='输出目录')
    
    args = parser.parse_args()
    run_ablation_experiment(args)


if __name__ == "__main__":
    main()
