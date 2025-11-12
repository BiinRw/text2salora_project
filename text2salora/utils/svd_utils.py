"""
SVD 相关工具函数
加载和管理偏好子空间
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional, Union


class PreferenceSubspaceManager:
    """偏好子空间管理器"""
    
    def __init__(self, subspace_dir: str, device: str = 'cuda:0'):
        """
        Args:
            subspace_dir: 子空间文件目录
            device: 设备
        """
        self.subspace_dir = Path(subspace_dir)
        self.device = device
        self.subspaces = {}  # {dimension: {layer_id: V_tensor}}
        
    def load_dimension(
        self, 
        dimension: str, 
        layer_ids: Optional[List[int]] = None,
        use_fused: bool = False
    ):
        """加载某个偏好维度的子空间
        
        Args:
            dimension: 偏好维度名称
            layer_ids: 加载特定层 (None=加载所有层)
            use_fused: 是否使用融合的子空间
        """
        if use_fused:
            # 加载融合子空间
            fused_file = self.subspace_dir / f'{dimension}_fused_subspace.pt'
            if not fused_file.exists():
                raise FileNotFoundError(f"融合子空间文件不存在: {fused_file}")
            
            data = torch.load(fused_file, map_location=self.device)
            self.subspaces[dimension] = {
                'fused': data['V'].to(self.device)
            }
            print(f"✅ 加载 {dimension} 融合子空间: shape={data['V'].shape}")
        
        else:
            # 加载各层子空间
            self.subspaces[dimension] = {}
            
            # 自动检测所有层文件
            layer_files = sorted(self.subspace_dir.glob(f'{dimension}_layer*.pt'))
            
            for layer_file in layer_files:
                # 从文件名提取 layer_id
                layer_id = int(layer_file.stem.split('layer')[1].split('_')[0])
                
                # 如果指定了 layer_ids,只加载这些层
                if layer_ids is not None and layer_id not in layer_ids:
                    continue
                
                data = torch.load(layer_file, map_location=self.device)
                self.subspaces[dimension][layer_id] = data['V'].to(self.device)
                
                print(f"✅ 加载 {dimension} Layer {layer_id}: shape={data['V'].shape}")
    
    def load_all_dimensions(
        self,
        dimensions: List[str],
        layer_ids: Optional[List[int]] = None,
        use_fused: bool = False
    ):
        """加载所有偏好维度
        
        Args:
            dimensions: 偏好维度列表
            layer_ids: 加载特定层
            use_fused: 是否使用融合子空间
        """
        print(f"📦 加载偏好子空间...")
        for dim in dimensions:
            self.load_dimension(dim, layer_ids, use_fused)
    
    def get_subspace(
        self, 
        dimension: str, 
        layer_id: Optional[int] = None
    ) -> torch.Tensor:
        """获取子空间基向量
        
        Args:
            dimension: 偏好维度
            layer_id: 层ID (如果使用融合子空间则为None)
            
        Returns:
            V: 子空间基向量 (d, k)
        """
        if dimension not in self.subspaces:
            raise ValueError(f"维度 {dimension} 未加载")
        
        dim_subspaces = self.subspaces[dimension]
        
        if layer_id is None:
            # 返回融合子空间
            if 'fused' in dim_subspaces:
                return dim_subspaces['fused']
            else:
                raise ValueError(f"{dimension} 没有融合子空间")
        else:
            # 返回特定层子空间
            if layer_id not in dim_subspaces:
                raise ValueError(f"{dimension} Layer {layer_id} 未加载")
            return dim_subspaces[layer_id]
    
    def get_all_dimensions_subspace(
        self,
        layer_id: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """获取所有维度的子空间
        
        Args:
            layer_id: 层ID (None=融合子空间)
            
        Returns:
            {dimension: V_tensor}
        """
        result = {}
        for dim in self.subspaces.keys():
            result[dim] = self.get_subspace(dim, layer_id)
        return result
    
    def print_info(self):
        """打印子空间信息"""
        print("\n" + "=" * 70)
        print("偏好子空间信息")
        print("=" * 70)
        
        for dim, layers in self.subspaces.items():
            print(f"\n📊 {dim.capitalize()}:")
            for layer_id, V in layers.items():
                if layer_id == 'fused':
                    print(f"   Fused: {V.shape}")
                else:
                    print(f"   Layer {layer_id:2d}: {V.shape}")


def load_subspace_simple(
    subspace_file: str,
    device: str = 'cuda:0'
) -> torch.Tensor:
    """简单加载单个子空间文件
    
    Args:
        subspace_file: 子空间文件路径
        device: 设备
        
    Returns:
        V: 子空间基向量
    """
    data = torch.load(subspace_file, map_location=device)
    return data['V'].to(device)


def compute_projection_matrix(V: torch.Tensor) -> torch.Tensor:
    """计算投影矩阵 P = V @ V^T
    
    Args:
        V: 子空间基向量 (d, k)
        
    Returns:
        P: 投影矩阵 (d, d)
    """
    return V @ V.T


def compute_orthogonal_projection_matrix(V: torch.Tensor) -> torch.Tensor:
    """计算正交补空间投影矩阵 P_orth = I - V @ V^T
    
    Args:
        V: 子空间基向量 (d, k)
        
    Returns:
        P_orth: 正交补投影矩阵 (d, d)
    """
    d = V.shape[0]
    I = torch.eye(d, device=V.device, dtype=V.dtype)
    return I - V @ V.T


if __name__ == '__main__':
    # 示例用法
    manager = PreferenceSubspaceManager(
        subspace_dir='./preference_subspace/saved_subspaces',
        device='cuda:0'
    )
    
    # 加载所有维度的融合子空间
    manager.load_all_dimensions(
        dimensions=['safety', 'helpfulness', 'correctness', 'coherence'],
        use_fused=True
    )
    
    # 打印信息
    manager.print_info()
    
    # 获取 safety 子空间
    V_safety = manager.get_subspace('safety')
    print(f"\nSafety 子空间形状: {V_safety.shape}")
    
    # 计算投影矩阵
    P = compute_projection_matrix(V_safety)
    print(f"投影矩阵形状: {P.shape}")
