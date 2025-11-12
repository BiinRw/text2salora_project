"""
正交约束核心实现
用于在 LoRA 训练时约束权重更新方向与偏好子空间正交
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from pathlib import Path
import sys

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))
from utils.svd_utils import PreferenceSubspaceManager, compute_projection_matrix


class OrthogonalConstraint:
    """正交约束计算器"""
    
    def __init__(
        self,
        subspace_manager: PreferenceSubspaceManager,
        dimensions: List[str],
        layer_ids: Optional[List[int]] = None,
        use_fused: bool = True,
        device: str = 'cuda:0'
    ):
        """
        Args:
            subspace_manager: 子空间管理器
            dimensions: 需要约束的偏好维度
            layer_ids: 对应的层ID (None=使用融合子空间)
            use_fused: 是否使用融合子空间
            device: 设备
        """
        self.manager = subspace_manager
        self.dimensions = dimensions
        self.layer_ids = layer_ids
        self.use_fused = use_fused
        self.device = device
        
        # 预计算投影矩阵 P = V @ V^T
        self._prepare_projection_matrices()
    
    def _prepare_projection_matrices(self):
        """预计算所有偏好维度的投影矩阵"""
        self.projection_matrices = {}  # {dimension: P or {layer_id: P}}
        
        print(f"\n📐 预计算投影矩阵...")
        
        for dim in self.dimensions:
            if self.use_fused:
                # 融合子空间: 一个投影矩阵
                V = self.manager.get_subspace(dim, layer_id=None)
                P = compute_projection_matrix(V)
                self.projection_matrices[dim] = P
                print(f"   {dim}: P shape={P.shape}")
            
            else:
                # 多层子空间: 每层一个投影矩阵
                self.projection_matrices[dim] = {}
                for layer_id in self.layer_ids:
                    V = self.manager.get_subspace(dim, layer_id=layer_id)
                    P = compute_projection_matrix(V)
                    self.projection_matrices[dim][layer_id] = P
                    print(f"   {dim} Layer {layer_id}: P shape={P.shape}")
    
    def compute_orthogonal_loss(
        self,
        lora_deltas: Dict[str, torch.Tensor],
        lambda_orth: float = 1.0,
        dimension_weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """计算正交损失
        
        正交损失定义: L_orth = ||ΔW @ P||²_F
        其中 ΔW = LoRA weight delta, P = V @ V^T
        
        最小化此损失 → ΔW 正交于子空间 V
        
        Args:
            lora_deltas: LoRA 权重更新 {layer_name: ΔW}
            lambda_orth: 正交损失系数
            dimension_weights: 各偏好维度权重 {dimension: weight}
            
        Returns:
            loss: 总正交损失
        """
        if dimension_weights is None:
            dimension_weights = {dim: 1.0 for dim in self.dimensions}
        
        total_loss = 0.0
        loss_details = {}
        
        for dim in self.dimensions:
            dim_weight = dimension_weights.get(dim, 1.0)
            dim_loss = 0.0
            
            for layer_name, delta_W in lora_deltas.items():
                # delta_W: (out_dim, in_dim) 或 (out_dim, rank) @ (rank, in_dim)
                
                # 获取对应的投影矩阵
                if self.use_fused:
                    P = self.projection_matrices[dim]
                else:
                    # 从 layer_name 提取 layer_id
                    layer_id = self._extract_layer_id(layer_name)
                    P = self.projection_matrices[dim].get(layer_id)
                    if P is None:
                        continue  # 该层未约束
                
                # 计算 ||ΔW @ P||²_F
                # 为了避免大矩阵乘法,改写为 trace((ΔW @ P) @ (ΔW @ P)^T)
                delta_P = delta_W @ P  # (out_dim, in_dim) @ (in_dim, in_dim) = (out_dim, in_dim)
                loss_term = torch.sum(delta_P ** 2)
                
                dim_loss += loss_term
            
            # 加权
            dim_loss = dim_weight * dim_loss
            loss_details[dim] = dim_loss.item()
            total_loss += dim_loss
        
        # 应用系数
        total_loss = lambda_orth * total_loss
        
        return total_loss, loss_details
    
    def compute_orthogonal_loss_efficient(
        self,
        lora_A: Dict[str, torch.Tensor],
        lora_B: Dict[str, torch.Tensor],
        lambda_orth: float = 1.0,
        dimension_weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """高效计算正交损失 (针对 LoRA 低秩分解)
        
        LoRA: ΔW = B @ A
        则 ΔW @ P = B @ A @ P
        ||ΔW @ P||²_F = ||B @ (A @ P)||²_F
        
        Args:
            lora_A: LoRA A矩阵 {layer_name: A (rank, in_dim)}
            lora_B: LoRA B矩阵 {layer_name: B (out_dim, rank)}
            lambda_orth: 正交损失系数
            dimension_weights: 各偏好维度权重
            
        Returns:
            loss: 总正交损失
            loss_details: 各维度损失详情
        """
        if dimension_weights is None:
            dimension_weights = {dim: 1.0 for dim in self.dimensions}
        
        total_loss = 0.0
        loss_details = {}
        
        for dim in self.dimensions:
            dim_weight = dimension_weights.get(dim, 1.0)
            dim_loss = 0.0
            
            for layer_name in lora_A.keys():
                A = lora_A[layer_name]  # (rank, in_dim)
                B = lora_B[layer_name]  # (out_dim, rank)
                
                # 获取投影矩阵
                if self.use_fused:
                    P = self.projection_matrices[dim]
                else:
                    layer_id = self._extract_layer_id(layer_name)
                    P = self.projection_matrices[dim].get(layer_id)
                    if P is None:
                        continue
                
                # 计算 B @ (A @ P)
                AP = A @ P  # (rank, in_dim) @ (in_dim, in_dim) = (rank, in_dim)
                BAP = B @ AP  # (out_dim, rank) @ (rank, in_dim) = (out_dim, in_dim)
                
                # ||BAP||²_F
                loss_term = torch.sum(BAP ** 2)
                dim_loss += loss_term
            
            dim_loss = dim_weight * dim_loss
            loss_details[dim] = dim_loss.item()
            total_loss += dim_loss
        
        total_loss = lambda_orth * total_loss
        
        return total_loss, loss_details
    
    def _extract_layer_id(self, layer_name: str) -> int:
        """从层名称提取层ID
        
        例如: 'model.layers.15.self_attn.q_proj' -> 15
        """
        parts = layer_name.split('.')
        for i, p in enumerate(parts):
            if p == 'layers' and i + 1 < len(parts):
                return int(parts[i + 1])
        raise ValueError(f"无法从 {layer_name} 提取层ID")


def collect_lora_deltas(model: nn.Module) -> Dict[str, torch.Tensor]:
    """从模型中收集 LoRA 权重更新 ΔW = B @ A
    
    Args:
        model: 包含 LoRA 层的模型
        
    Returns:
        {layer_name: ΔW}
    """
    lora_deltas = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # PEFT LoRA 结构
            A = module.lora_A['default'].weight  # (rank, in_dim)
            B = module.lora_B['default'].weight  # (out_dim, rank)
            
            # 计算 ΔW = B @ A
            delta_W = B @ A
            lora_deltas[name] = delta_W
    
    return lora_deltas


def collect_lora_AB_matrices(model: nn.Module) -> tuple:
    """从模型中收集 LoRA A 和 B 矩阵
    
    Args:
        model: 包含 LoRA 层的模型
        
    Returns:
        lora_A: {layer_name: A}
        lora_B: {layer_name: B}
    """
    lora_A = {}
    lora_B = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            A = module.lora_A['default'].weight  # (rank, in_dim)
            B = module.lora_B['default'].weight  # (out_dim, rank)
            
            lora_A[name] = A
            lora_B[name] = B
    
    return lora_A, lora_B


if __name__ == '__main__':
    # 测试代码
    from utils.svd_utils import PreferenceSubspaceManager
    
    # 加载子空间
    manager = PreferenceSubspaceManager(
        subspace_dir='../preference_subspace/saved_subspaces',
        device='cuda:0'
    )
    
    manager.load_all_dimensions(
        dimensions=['safety', 'helpfulness'],
        use_fused=True
    )
    
    # 创建约束计算器
    constraint = OrthogonalConstraint(
        subspace_manager=manager,
        dimensions=['safety', 'helpfulness'],
        use_fused=True,
        device='cuda:0'
    )
    
    # 模拟 LoRA 权重
    lora_A = {
        'model.layers.15.self_attn.q_proj': torch.randn(8, 1024, device='cuda:0')
    }
    lora_B = {
        'model.layers.15.self_attn.q_proj': torch.randn(1024, 8, device='cuda:0')
    }
    
    # 计算损失
    loss, details = constraint.compute_orthogonal_loss_efficient(
        lora_A, lora_B, lambda_orth=0.1
    )
    
    print(f"\n总损失: {loss.item():.6f}")
    print(f"详情: {details}")
