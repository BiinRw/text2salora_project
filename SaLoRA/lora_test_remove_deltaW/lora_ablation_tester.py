"""
LoRA △W 层级消融测试框架

支持在推理时选择性禁用某些层的LoRA权重更新,用于验证:
1. 模型安全语义是否集中在特定层
2. 表征可分性(探针准确度) vs 模型行为安全性的关系
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from datetime import datetime
import logging

# 配置logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoRAAblusionTester:
    """
    LoRA权重消融测试器
    
    核心功能:
    - 保存原始模型权重和LoRA应用后的权重
    - 支持运行时禁用/恢复某些层的LoRA
    - 支持自动化的消融实验循环
    - 记录和分析结果
    """
    
    def __init__(
        self,
        model,
        num_layers: int = 28,
        device: str = 'cuda:0'
    ):
        """
        初始化消融测试器
        
        Args:
            model: 已加载LoRA的LLM模型
            num_layers: 模型层数 (Qwen2.5默认为28)
            device: 设备 ('cuda:0' 或 'cpu')
        """
        self.model = model
        self.num_layers = num_layers
        self.device = device
        
        # 权重备份存储
        self.original_weights: Dict[int, torch.Tensor] = {}  # 原始权重
        self.lora_weights: Dict[int, torch.Tensor] = {}      # LoRA应用后的权重
        
        # 实验结果
        self.ablation_results: Dict = {}
        
        logger.info(f"🔧 初始化LoRAAblusionTester (num_layers={num_layers})")
    
    def save_original_weights(self):
        """
        保存模型的原始权重 (未应用LoRA前)
        
        这应该在加载LoRA之前调用
        """
        logger.info("💾 保存原始权重...")
        
        for layer_id in tqdm(range(self.num_layers), desc="保存原始权重"):
            try:
                layer = self.model.model.layers[layer_id]
                # 保存Q投影层的权重 (其他层可类似扩展)
                self.original_weights[layer_id] = \
                    layer.self_attn.q_proj.weight.data.clone().detach()
            except Exception as e:
                logger.warning(f"⚠️ 保存第 {layer_id} 层权重失败: {e}")
                self.original_weights[layer_id] = None
        
        logger.info(f"✅ 已保存 {len(self.original_weights)} 层的原始权重")
    
    def save_lora_weights(self):
        """
        保存应用LoRA后的权重
        
        这应该在应用LoRA后调用
        """
        logger.info("💾 保存LoRA应用后的权重...")
        
        for layer_id in tqdm(range(self.num_layers), desc="保存LoRA权重"):
            try:
                layer = self.model.model.layers[layer_id]
                self.lora_weights[layer_id] = \
                    layer.self_attn.q_proj.weight.data.clone().detach()
            except Exception as e:
                logger.warning(f"⚠️ 保存第 {layer_id} 层LoRA权重失败: {e}")
                self.lora_weights[layer_id] = None
        
        logger.info(f"✅ 已保存 {len(self.lora_weights)} 层的LoRA权重")
    
    def disable_lora_on_layers(self, layer_ids: List[int]):
        """
        禁用指定层的LoRA (恢复到原始权重)
        
        Args:
            layer_ids: 要禁用的层ID列表, 如 [16] 或 [0, 8, 16, 27]
            
        Note:
            这将把指定层的权重恢复到 W_orig, 相当于 △W = 0
        """
        if not layer_ids:
            logger.info("ℹ️ 无需禁用任何层 (baseline配置)")
            return
        
        logger.info(f"🔇 禁用LoRA on layers: {layer_ids}")
        
        for layer_id in layer_ids:
            if layer_id >= self.num_layers:
                logger.warning(f"⚠️ 层ID {layer_id} 超出范围 (最大 {self.num_layers-1})")
                continue
            
            if self.original_weights.get(layer_id) is None:
                logger.warning(f"⚠️ 第 {layer_id} 层的原始权重未保存")
                continue
            
            try:
                layer = self.model.model.layers[layer_id]
                # 恢复到原始权重
                layer.self_attn.q_proj.weight.data = \
                    self.original_weights[layer_id].clone().to(self.device)
                logger.debug(f"   Layer {layer_id}: W = W_orig")
            except Exception as e:
                logger.error(f"❌ 禁用第 {layer_id} 层失败: {e}")
    
    def restore_lora_on_layers(self, layer_ids: List[int]):
        """
        恢复指定层的LoRA权重
        
        Args:
            layer_ids: 要恢复的层ID列表
            
        Note:
            这将把指定层的权重恢复到 W_lora, 相当于 △W = B @ A @ C
        """
        if not layer_ids:
            logger.info("ℹ️ 无需恢复任何层")
            return
        
        logger.info(f"🔊 恢复LoRA on layers: {layer_ids}")
        
        for layer_id in layer_ids:
            if layer_id >= self.num_layers:
                logger.warning(f"⚠️ 层ID {layer_id} 超出范围")
                continue
            
            if self.lora_weights.get(layer_id) is None:
                logger.warning(f"⚠️ 第 {layer_id} 层的LoRA权重未保存")
                continue
            
            try:
                layer = self.model.model.layers[layer_id]
                # 恢复LoRA权重
                layer.self_attn.q_proj.weight.data = \
                    self.lora_weights[layer_id].clone().to(self.device)
                logger.debug(f"   Layer {layer_id}: W = W_lora")
            except Exception as e:
                logger.error(f"❌ 恢复第 {layer_id} 层失败: {e}")
    
    def get_lora_delta_w(self, layer_id: int) -> Optional[torch.Tensor]:
        """
        计算某一层的△W (LoRA权重更新)
        
        Args:
            layer_id: 层ID
            
        Returns:
            △W = W_lora - W_orig, shape取决于q_proj权重
        """
        if (self.original_weights.get(layer_id) is None or 
            self.lora_weights.get(layer_id) is None):
            return None
        
        delta_w = self.lora_weights[layer_id] - self.original_weights[layer_id]
        return delta_w
    
    def get_lora_importance(self, layer_id: int) -> float:
        """
        估计某一层LoRA的重要性 (基于权重更新的大小)
        
        Args:
            layer_id: 层ID
            
        Returns:
            Frobenius范数 ||△W||_F
        """
        delta_w = self.get_lora_delta_w(layer_id)
        if delta_w is None:
            return 0.0
        
        return torch.norm(delta_w, p='fro').item()
    
    def run_inference(self, prompt: str, max_tokens: int = 100) -> str:
        """
        运行单次推理 (需要由调用者实现具体逻辑)
        
        Args:
            prompt: 输入提示词
            max_tokens: 最大生成token数
            
        Returns:
            模型生成的文本
            
        Note:
            这是一个占位符,具体实现取决于使用的推理引擎
        """
        raise NotImplementedError("需要由子类或调用者实现推理逻辑")
    
    def prepare_ablation_configs(self) -> Dict[str, List[int]]:
        """
        准备标准的消融配置
        
        Returns:
            配置字典: {'config_name': [layer_ids]}
        """
        configs = {
            'baseline': [],                          # 全层LoRA
            'disable_layer_16': [16],               # 只禁用16层
            'disable_early_layers_0_8': list(range(0, 9)),      # 禁用0-8
            'disable_mid_layers_8_16': list(range(8, 17)),      # 禁用8-16
            'disable_late_layers_17_27': list(range(17, 28)),   # 禁用17-27
        }
        return configs
    
    def run_ablation_test(
        self,
        ablation_configs: Optional[Dict[str, List[int]]] = None,
        save_results: bool = True,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """
        运行完整的消融实验循环
        
        Args:
            ablation_configs: 消融配置字典
            save_results: 是否保存结果到JSON
            output_dir: 输出目录
            
        Returns:
            实验结果字典
            
        Note:
            需要由调用者提供evaluate()方法来计算指标
        """
        if ablation_configs is None:
            ablation_configs = self.prepare_ablation_configs()
        
        logger.info(f"🧪 开始消融实验 ({len(ablation_configs)} 个配置)")
        
        self.ablation_results = {}
        
        for config_name, disabled_layers in ablation_configs.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"配置: {config_name}")
            logger.info(f"禁用层: {disabled_layers if disabled_layers else '[无] (baseline)'}")
            logger.info(f"{'='*60}")
            
            try:
                # 应用消融配置
                self.disable_lora_on_layers(disabled_layers)
                
                # 这里需要由调用者实现具体的评估逻辑
                # 例如: harmfulness_rate, probe_accuracy = self.evaluate()
                result = {
                    'disabled_layers': disabled_layers,
                    'timestamp': datetime.now().isoformat(),
                    # 需要填充: harmfulness_rate, probe_accuracy等指标
                }
                
                self.ablation_results[config_name] = result
                logger.info(f"✅ 配置 {config_name} 完成")
                
            except Exception as e:
                logger.error(f"❌ 配置 {config_name} 失败: {e}")
                self.ablation_results[config_name] = {'error': str(e)}
            
            finally:
                # 恢复LoRA权重
                self.restore_lora_on_layers(disabled_layers)
        
        # 保存结果
        if save_results and output_dir:
            self.save_results(output_dir)
        
        return self.ablation_results
    
    def save_results(self, output_dir: Path):
        """
        保存消融实验结果到JSON文件
        
        Args:
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / "ablation_results.json"
        
        # 转换torch张量为可序列化的格式
        results_to_save = {}
        for config, result in self.ablation_results.items():
            results_to_save[config] = {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in result.items()
            }
        
        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        logger.info(f"💾 结果已保存到: {results_file}")
    
    def analyze_results(self) -> Dict:
        """
        分析消融实验结果
        
        Returns:
            分析结果字典
        """
        logger.info("\n📊 消融实验结果分析:")
        logger.info("="*60)
        
        analysis = {}
        
        # 计算△W的大小 (重要性估计)
        logger.info("\n🔢 各层LoRA重要性 (||△W||_F):")
        importance_scores = {}
        for layer_id in range(self.num_layers):
            importance = self.get_lora_importance(layer_id)
            importance_scores[layer_id] = importance
            if layer_id % 5 == 0:
                logger.info(f"   Layer {layer_id:2d}: {importance:.4f}")
        
        analysis['layer_importance'] = importance_scores
        
        # 找出最重要的层
        top_k = 5
        top_layers = sorted(
            importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        logger.info(f"\n🏆 Top {top_k} 重要层:")
        for rank, (layer_id, score) in enumerate(top_layers, 1):
            logger.info(f"   {rank}. Layer {layer_id}: {score:.4f}")
        
        analysis['top_important_layers'] = [l for l, _ in top_layers]
        
        return analysis
    
    def print_summary(self):
        """打印消融实验的总结"""
        logger.info("\n" + "="*60)
        logger.info("🎯 消融实验总结")
        logger.info("="*60)
        logger.info(f"模型层数: {self.num_layers}")
        logger.info(f"原始权重备份: {len(self.original_weights)} 层")
        logger.info(f"LoRA权重备份: {len(self.lora_weights)} 层")
        logger.info(f"消融配置数: {len(self.ablation_results)}")
        logger.info("="*60 + "\n")


# 使用示例
if __name__ == "__main__":
    print("""
    LoRA消融测试框架使用示例:
    
    ```python
    # 1. 加载模型 (使用现有的load_model_with_abc函数)
    model, tokenizer, _ = load_model_with_abc(
        model_path, lora_path, subspace_dir, dimension
    )
    
    # 2. 创建消融测试器
    tester = LoRAAblusionTester(model, num_layers=28)
    
    # 3. 保存权重备份
    tester.save_original_weights()  # 注意: 应在应用LoRA前调用
    model = load_model_with_abc(...)  # 应用LoRA
    tester.save_lora_weights()  # 应用LoRA后调用
    
    # 4. 定义消融配置
    ablation_configs = {
        'baseline': [],
        'disable_layer_16': [16],
        'disable_layers_0_8': list(range(0, 9)),
    }
    
    # 5. 运行消融实验 (需要实现evaluate方法)
    results = tester.run_ablation_test(ablation_configs)
    
    # 6. 分析结果
    analysis = tester.analyze_results()
    ```
    """)
