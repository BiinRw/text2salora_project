"""
消融测试工具模块

功能：
- 配置加载和管理
- 结果保存和分析
- 日志记录
- 通用工具函数
"""

import json
import yaml
import torch
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: str = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径 (.yaml或.json)
        """
        self.config_file = config_file
        self.config = {}
        
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file: str) -> Dict:
        """加载配置文件"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            logger.warning(f"配置文件不存在: {config_file}")
            return {}
        
        try:
            if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
            elif config_path.suffix == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                logger.error(f"不支持的配置文件格式: {config_path.suffix}")
                return {}
            
            logger.info(f"✅ 配置加载成功: {config_file}")
            return self.config
        
        except Exception as e:
            logger.error(f"❌ 配置加载失败: {e}")
            return {}
    
    def get(self, key: str, default=None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        
        return value if value is not None else default
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self, output_file: str):
        """保存配置文件"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if output_path.suffix == '.yaml' or output_path.suffix == '.yml':
                with open(output_path, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(self.config, f, default_flow_style=False, allow_unicode=True)
            elif output_path.suffix == '.json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 配置保存成功: {output_file}")
        except Exception as e:
            logger.error(f"❌ 配置保存失败: {e}")


class ResultsManager:
    """结果管理器"""
    
    def __init__(self, output_dir: str = './results'):
        """
        初始化结果管理器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def add_result(self, key: str, value: Any):
        """添加结果"""
        self.results[key] = value
    
    def add_results_dict(self, results_dict: Dict):
        """添加结果字典"""
        self.results.update(results_dict)
    
    def save_results(self, filename: str = 'results.json', format: str = 'json'):
        """
        保存结果
        
        Args:
            filename: 文件名
            format: 文件格式 ('json' 或 'yaml')
        """
        output_file = self.output_dir / filename
        
        try:
            if format == 'json':
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, indent=2, ensure_ascii=False)
            elif format == 'yaml':
                with open(output_file, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(self.results, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"✅ 结果保存成功: {output_file}")
        except Exception as e:
            logger.error(f"❌ 结果保存失败: {e}")
    
    def save_csv_results(self, filename: str = 'results.csv'):
        """
        保存为CSV格式
        
        Args:
            filename: 文件名
        """
        import pandas as pd
        
        output_file = self.output_dir / filename
        
        try:
            df = pd.DataFrame(self.results)
            df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"✅ CSV结果保存成功: {output_file}")
        except Exception as e:
            logger.error(f"❌ CSV保存失败: {e}")
    
    def get_summary(self) -> Dict:
        """获取结果摘要"""
        return {
            'timestamp': datetime.now().isoformat(),
            'total_results': len(self.results),
            'results': self.results,
        }


class ExperimentLogger:
    """实验日志记录器"""
    
    def __init__(self, log_file: str = './experiment.log'):
        """
        初始化实验日志记录器
        
        Args:
            log_file: 日志文件路径
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 配置logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # 文件处理器
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # 添加处理器
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
    
    def log(self, message: str, level: str = 'info'):
        """记录日志"""
        if level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        elif level == 'debug':
            self.logger.debug(message)


class AblationAnalyzer:
    """消融分析器"""
    
    @staticmethod
    def compare_results(results_harmful: Dict, results_probes: Dict) -> Dict:
        """
        比较有害率和探针准确率结果
        
        Args:
            results_harmful: 有害率结果字典
            results_probes: 探针准确率结果字典
            
        Returns:
            对比分析结果
        """
        analysis = {}
        
        for config_name in results_harmful.keys():
            if config_name not in results_probes:
                continue
            
            harmful_rate = results_harmful[config_name].get('harmful_rate', None)
            probe_acc = results_probes[config_name].get('probe_accuracy', None)
            
            if harmful_rate is not None and probe_acc is not None:
                analysis[config_name] = {
                    'harmful_rate': harmful_rate,
                    'probe_accuracy': probe_acc,
                    'correlation': harmful_rate * probe_acc,  # 简单相关性度量
                }
        
        return analysis
    
    @staticmethod
    def get_layer_importance(analysis: Dict) -> Dict:
        """
        计算层的重要性排序
        
        Args:
            analysis: 对比分析结果
            
        Returns:
            层重要性排序
        """
        importance = {}
        
        for config_name, metrics in analysis.items():
            # 提取层信息
            if 'layers' in config_name:
                layers = config_name.split('_')[-1]
                importance[layers] = metrics.get('correlation', 0)
        
        # 按重要性排序
        sorted_importance = dict(sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return sorted_importance


def create_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    创建日志记录器
    
    Args:
        name: logger名称
        log_file: 日志文件路径
        
    Returns:
        logger实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_model_weights(model_path: str) -> Dict:
    """加载模型权重"""
    try:
        weights = torch.load(model_path, map_location='cpu')
        logger.info(f"✅ 权重加载成功: {model_path}")
        return weights
    except Exception as e:
        logger.error(f"❌ 权重加载失败: {e}")
        return {}


def save_model_weights(weights: Dict, output_path: str):
    """保存模型权重"""
    try:
        torch.save(weights, output_path)
        logger.info(f"✅ 权重保存成功: {output_path}")
    except Exception as e:
        logger.error(f"❌ 权重保存失败: {e}")


if __name__ == "__main__":
    # 测试工具模块
    print("✅ 工具模块已加载")
