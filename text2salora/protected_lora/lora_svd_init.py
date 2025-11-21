"""
LoRA SVD 初始化模块

实现 SaLoRA 论文中的初始化方法：
1. SVD 分解原始权重
2. 投影到正交补空间（或子空间内）
3. 更新基础权重
"""

import torch
from typing import Optional
import sys
import re
from pathlib import Path

def parse_projection_from_module_name(module_name: str) -> Optional[str]:
    """
    从模块名解析投影类型
    
    Args:
        module_name: 如 "model.layers.0.self_attn.q_proj.lora_A.weight"
        
    Returns:
        projection_type: 如 "q_proj", "k_proj", 等,或 None
    """
    if 'q_proj' in module_name:
        return 'q_proj'
    elif 'k_proj' in module_name:
        return 'k_proj'
    elif 'v_proj' in module_name:
        return 'v_proj'
    elif 'o_proj' in module_name:
        return 'o_proj'
    elif 'up_proj' in module_name:
        return 'up_proj'
    elif 'down_proj' in module_name:
        return 'down_proj'
    elif 'gate_proj' in module_name:
        return 'gate_proj'
    else:
        return None


def load_projection_subspace(subspace_dir: str, dimension: str, projection: str, layer_id: int, device: str) -> Optional[torch.Tensor]:
    """
    加载指定投影层的子空间矩阵 C
    
    Args:
        subspace_dir: 子空间文件目录
        dimension: 偏好维度 (safety/helpfulness/etc.)
        projection: 投影类型 (q_proj/k_proj/etc.)
        layer_id: 层编号
        device: 设备
        
    Returns:
        C: 约束矩阵 (d_out, d_out) 或 None
    """
    subspace_dir = Path(subspace_dir)
    
    # 尝试加载 per-layer 文件: {dimension}_{projection}_layer{id}_subspace.pt
    subspace_file = subspace_dir / f"{dimension}_{projection}_layer{layer_id}_subspace.pt"
    
    # 如果不存在,尝试 fused 文件
    if not subspace_file.exists():
        subspace_file = subspace_dir / f"{dimension}_{projection}_fused_subspace.pt"
        
    if not subspace_file.exists():
        return None
    
    try:
        data = torch.load(subspace_file, map_location=device)
        V = data['V'].to(device)  # (d_out, k)
        
        # 计算约束矩阵: C = I - V @ V^T
        d_out = V.shape[0]
        I = torch.eye(d_out, device=device, dtype=V.dtype)
        C = I - torch.mm(V, V.t())
        
        return C
        
    except Exception as e:
        print(f"   ⚠️  加载子空间失败: {subspace_file} | {e}")
        return None

def initialize_lora_weights(
    model,
    constraint=None,
    rank: int = 16,
    method: str = 'random',
    niter: int = 30,
    verbose: bool = True
):
    """
    初始化 LoRA 权重
    
    Args:
        model: PEFT LoRA 模型
        constraint: OrthogonalConstraint 对象，包含子空间投影矩阵 C
        rank: LoRA 的秩
        method: 初始化方法
            - 'random': PEFT 默认随机初始化（不做任何处理）
            - 'svd': PiSSA 方法（SVD 分解，不投影）
            - 'svd_ortho': SVD + 投影到正交补空间（推荐）
            - 'svd_salora': SaLoRA 原始方法（SVD + 投影到子空间内）
        niter: SVD 迭代次数
        verbose: 是否打印详细信息
    
    Returns:
        初始化的模块数量
    """
    
    if method == 'random':
        if verbose:
            print("✅ 使用默认随机初始化")
        return 0
    
    if method in ['svd_ortho', 'svd_salora'] and constraint is None:
        raise ValueError(f"方法 '{method}' 需要提供 constraint 对象")
    
    if verbose:
        print(f"\n🔧 使用 {method} 方法初始化 LoRA...")
        print(f"   Rank: {rank}, SVD迭代次数: {niter}")
    
    initialized_count = 0
    
    for name, module in model.named_modules():
        # 检查是否是 LoRA 模块
        if not (hasattr(module, 'lora_A') and hasattr(module, 'lora_B')):
            continue
        
        # 检查是否有 base_layer
        if not hasattr(module, 'base_layer'):
            if verbose:
                print(f"⚠️ {name} 没有 base_layer，跳过")
            continue
        
        try:
            # 获取基础权重
            base_weight = module.base_layer.weight.data
            d_out, d_in = base_weight.shape
            
            if verbose:
                print(f"\n处理模块: {name}")
                print(f"  权重形状: {base_weight.shape}")
            
            # SVD 分解
            if verbose:
                print(f"  执行 SVD 分解 (rank={rank})...")
            
            U, S, V = torch.svd_lowrank(
                base_weight.float(), 
                q=min(rank, min(d_out, d_in)), 
                niter=niter
            )
            
            # 转换回原始数据类型
            U = U.to(base_weight.dtype)
            S = S.to(base_weight.dtype)
            V = V.to(base_weight.dtype)
            
            # 计算 sqrt(S)
            sqrt_S = torch.sqrt(S)
            
            # 初始化 B 和 A
            # B: (d_out, r)
            # A: (r, d_in)
            B_init = U @ torch.diag(sqrt_S)
            A_init = torch.diag(sqrt_S) @ V.T
            
            if verbose:
                print(f"  B_init 形状: {B_init.shape}, A_init 形状: {A_init.shape}")
            
            # 根据方法选择投影方式
            if method == 'svd_ortho':
                # 🆕 新版本: 从projection特定的子空间文件加载C矩阵
                if verbose:
                    print(f"  投影到正交补空间...")
                
                # 提取层编号
                layer_match = re.search(r'\.layers\.(\d+)\.', name)
                if not layer_match:
                    if verbose:
                        print(f"   ⚠️  无法提取层编号: {name}, 使用单位矩阵")
                    C = torch.eye(d_out, device=B_init.device, dtype=B_init.dtype)
                else:
                    layer_id = int(layer_match.group(1))
                    
                    # 解析投影类型
                    projection = parse_projection_from_module_name(name)
                    
                    if projection is None:
                        if verbose:
                            print(f"   ⚠️  无法识别投影类型: {name}, 使用单位矩阵")
                        C = torch.eye(d_out, device=B_init.device, dtype=B_init.dtype)
                    else:
                        # 从文件加载projection特定的C矩阵
                        # 需要从constraint对象获取subspace_dir和dimension
                        if hasattr(constraint, 'subspace_dir') and hasattr(constraint, 'dimension'):
                            C = load_projection_subspace(
                                constraint.subspace_dir,
                                constraint.dimension,
                                projection,
                                layer_id,
                                B_init.device
                            )
                            
                            if C is not None:
                                # 检查维度匹配
                                if C.shape[0] != d_out:
                                    if verbose:
                                        print(f"   ⚠️  C 形状 {C.shape} 与 d_out {d_out} 不匹配 ({projection} Layer {layer_id}), 使用单位矩阵")
                                    C = torch.eye(d_out, device=B_init.device, dtype=B_init.dtype)
                                elif verbose:
                                    print(f"   ✅ 加载子空间: {projection} Layer {layer_id} | C shape: {C.shape}")
                            else:
                                if verbose:
                                    print(f"   ℹ️  未找到子空间文件: {projection} Layer {layer_id}, 使用单位矩阵")
                                C = torch.eye(d_out, device=B_init.device, dtype=B_init.dtype)
                        else:
                            # 兼容旧版constraint对象(直接传P矩阵)
                            dim = constraint.dimensions[0] if hasattr(constraint, 'dimensions') and constraint.dimensions else "safety"
                            P_data = constraint.projection_matrices.get(dim) if hasattr(constraint, 'projection_matrices') else None
                            if isinstance(P_data, dict):
                                C = P_data.get(layer_id)
                                if C is None:
                                    C = torch.eye(d_out, device=B_init.device, dtype=B_init.dtype)
                            else:
                                C = P_data if P_data is not None else torch.eye(d_out, device=B_init.device, dtype=B_init.dtype)
                
                # 确保C在正确的设备和类型上
                if C is not None:
                    if not isinstance(C, torch.Tensor):
                        C = torch.tensor(C)
                    C = C.to(device=B_init.device, dtype=B_init.dtype)
                else:
                    C = torch.eye(d_out, device=B_init.device, dtype=B_init.dtype)
                
                # 确保 C 的形状与 B_init 匹配
                if C.shape[0] != d_out:
                    if verbose:
                        print(f"  ⚠️ C 形状 {C.shape} 与 d_out {d_out} 不匹配，跳过投影")
                    continue
                else:
                    I_minus_C = torch.eye(d_out, device=C.device, dtype=C.dtype) - C
                    # 投影到正交补空间: B' = (I - C) @ B
                    B_init = I_minus_C @ B_init
                    
                    if verbose:
                        print(f"  投影后 B_init 形状: {B_init.shape}")
            
            elif method == 'svd_salora':
                # 🆕 新版本: 从projection特定的子空间文件加载C矩阵
                if verbose:
                    print(f"  投影到子空间内 (SaLoRA 方法)...")
                
                # 提取层编号
                layer_match = re.search(r'\.layers\.(\d+)\.', name)
                if not layer_match:
                    if verbose:
                        print(f"   ⚠️  无法提取层编号: {name}, 使用单位矩阵")
                    C = torch.eye(d_out, device=B_init.device, dtype=B_init.dtype)
                else:
                    layer_id = int(layer_match.group(1))
                    
                    # 解析投影类型
                    projection = parse_projection_from_module_name(name)
                    
                    if projection is None:
                        if verbose:
                            print(f"   ⚠️  无法识别投影类型: {name}, 使用单位矩阵")
                        C = torch.eye(d_out, device=B_init.device, dtype=B_init.dtype)
                    else:
                        # 从文件加载projection特定的C矩阵
                        if hasattr(constraint, 'subspace_dir') and hasattr(constraint, 'dimension'):
                            C = load_projection_subspace(
                                constraint.subspace_dir,
                                constraint.dimension,
                                projection,
                                layer_id,
                                B_init.device
                            )
                            
                            if C is not None:
                                # 检查维度匹配
                                if C.shape[0] != d_out:
                                    if verbose:
                                        print(f"   ⚠️  C 形状 {C.shape} 与 d_out {d_out} 不匹配 ({projection} Layer {layer_id}), 使用单位矩阵")
                                    C = torch.eye(d_out, device=B_init.device, dtype=B_init.dtype)
                                elif verbose:
                                    print(f"   ✅ 加载子空间: {projection} Layer {layer_id} | C shape: {C.shape}")
                            else:
                                if verbose:
                                    print(f"   ℹ️  未找到子空间文件: {projection} Layer {layer_id}, 使用单位矩阵")
                                C = torch.eye(d_out, device=B_init.device, dtype=B_init.dtype)
                        else:
                            # 兼容旧版constraint对象
                            dim = constraint.dimensions[0] if hasattr(constraint, 'dimensions') and constraint.dimensions else "safety"
                            P_data = constraint.projection_matrices.get(dim) if hasattr(constraint, 'projection_matrices') else None
                            if isinstance(P_data, dict):
                                C = P_data.get(layer_id)
                                if C is None:
                                    C = torch.eye(d_out, device=B_init.device, dtype=B_init.dtype)
                            else:
                                C = P_data if P_data is not None else torch.eye(d_out, device=B_init.device, dtype=B_init.dtype)
                
                # 确保C在正确的设备和类型上
                if C is not None:
                    if not isinstance(C, torch.Tensor):
                        C = torch.tensor(C)
                    C = C.to(device=B_init.device, dtype=B_init.dtype)
                else:
                    C = torch.eye(d_out, device=B_init.device, dtype=B_init.dtype)
                
                if C.shape[0] != d_out:
                    if verbose:
                        print(f"  ⚠️ C 形状 {C.shape} 与 d_out {d_out} 不匹配，跳过投影")
                    continue
                else:
                    # 投影到子空间内: B' = C @ B
                    B_init = C @ B_init
                    
                    if verbose:
                        print(f"  投影后 B_init 形状: {B_init.shape}")
                    
                    # ✅ 关键步骤：更新 base_weight (与原版 SaLoRA 一致)
                    # 原理：使 LoRA 从 0 贡献开始
                    # 输出 = (W - B@A) @ x + B @ A @ x = W @ x
                    # 注意：因为 C @ B = B (B 在子空间内)，所以实际减去 B @ A
                    base_layer = module.base_layer
                    if hasattr(base_layer, 'weight'):
                        with torch.no_grad():
                            # 计算要减去的部分：B @ A
                            delta = (B_init @ A_init).to(base_layer.weight.dtype)
                            base_layer.weight.data.sub_(delta)
                            
                            if verbose:
                                print(f"  ✅ 已更新 base_weight: 减去 B@A (形状: {delta.shape})")
                                print(f"     确保初始化后模型输出与预训练模型一致")
            
            # 赋值给 LoRA 参数
            # 注意：PEFT 的 lora_B 和 lora_A 的 weight 需要转置
            # lora_B: Linear(r, d_out) -> weight: (d_out, r)
            # lora_A: Linear(d_in, r) -> weight: (r, d_in)
            
            if 'default' in module.lora_A:
                adapter_name = 'default'
            else:
                adapter_name = list(module.lora_A.keys())[0]
            
            # B_init: (d_out, r) -> lora_B.weight: (d_out, r)
            # 使用 with torch.no_grad() 避免创建计算图，同时保持 requires_grad
            with torch.no_grad():
                module.lora_B[adapter_name].weight.copy_(B_init.detach())
            # 确保 requires_grad=True
            module.lora_B[adapter_name].weight.requires_grad_(True)
            
            # A_init: (r, d_in) -> lora_A.weight: (r, d_in)
            with torch.no_grad():
                module.lora_A[adapter_name].weight.copy_(A_init.detach())
            # 确保 requires_grad=True
            module.lora_A[adapter_name].weight.requires_grad_(True)
            
            if verbose:
                print(f"  ✅ 已赋值 lora_A 和 lora_B")
            
            # 更新基础权重: W' = W - B @ A
            # 这样保证 W' + B @ A = W（初始输出不变）
            # 计算 BA 乘积并 detach，确保不影响计算图
            BA_product = (B_init @ A_init).detach()
            module.base_layer.weight.data.sub_(BA_product)
            
            if verbose:
                print(f"  ✅ 已更新基础权重")
                # 验证
                reconstructed = module.base_layer.weight.data + BA_product
                error = torch.norm(reconstructed - base_weight) / torch.norm(base_weight)
                print(f"  重构误差: {error.item():.6f}")
                
                # 验证梯度状态
                print(f"  📊 梯度状态:")
                print(f"     lora_B.requires_grad: {module.lora_B[adapter_name].weight.requires_grad}")
                print(f"     lora_A.requires_grad: {module.lora_A[adapter_name].weight.requires_grad}")
                print(f"     base_layer.requires_grad: {module.base_layer.weight.requires_grad}")
            
            initialized_count += 1
            
        except Exception as e:
            print(f"❌ 初始化 {name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if verbose:
        print(f"\n✅ LoRA 初始化完成！共初始化 {initialized_count} 个模块")
    
    return initialized_count




if __name__ == "__main__":
    # 测试代码
    print("LoRA SVD 初始化模块")
    print("支持的方法:")
    print("  - random: 默认随机初始化")
    print("  - svd: PiSSA 方法")
    print("  - svd_ortho: SVD + 正交补投影 (推荐)")
    print("  - svd_salora: SaLoRA 原始方法")