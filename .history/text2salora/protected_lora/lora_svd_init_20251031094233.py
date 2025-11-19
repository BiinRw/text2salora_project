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
                # 投影到正交补空间: B' = (I - C) @ B
                if verbose:
                    print(f"  投影到正交补空间...")
                
                # 获取投影矩阵 P = V @ V^T
                # OrthogonalConstraint.projection_matrices: {dimension: P} or {dimension: {layer_id: P}}
                dim = constraint.dimensions[0] if hasattr(constraint, 'dimensions') and constraint.dimensions else "safety"
                P_data = constraint.projection_matrices.get(dim) if hasattr(constraint, 'projection_matrices') else None
                
                # 处理分层和融合两种情况
                if isinstance(P_data, dict):
                    # 分层的情况：从模块名提取layer_id
                    import re
                    layer_match = re.search(r'\.layers\.(\d+)\.', name)
                    if layer_match:
                        layer_id = int(layer_match.group(1))
                        C = P_data.get(layer_id)
                    else:
                        C = None
                else:
                    # 融合的情况：直接使用
                    C = P_data
                
                # 如果没有找到投影矩阵，使用单位矩阵（等价于不投影）
                if C is None:
                    C = torch.eye(d_out, device=B_init.device, dtype=B_init.dtype)
                else:
                    C = C.to(device=B_init.device, dtype=B_init.dtype)
                
                # 确保 C 的形状与 B_init 匹配
                if C.shape[0] != d_out:
                    if verbose:
                        print(f"  ⚠️ C 形状 {C.shape} 与 d_out {d_out} 不匹配，跳过投影")
                else:
                    I_minus_C = torch.eye(d_out, device=C.device, dtype=C.dtype) - C
                    # SaLoRA 原始方法 B' = C @ B， 投影到安全子空间内部
                    B_init = I_minus_C @ B_init
                    
                    if verbose:
                        print(f"  投影后 B_init 形状: {B_init.shape}")
            
            elif method == 'svd_salora':
                # SaLoRA 原始方法：投影到子空间内: B' = C @ B
                if verbose:
                    print(f"  投影到子空间内 (SaLoRA 方法)...")
                
                # 获取投影矩阵 P = V @ V^T
                # OrthogonalConstraint.projection_matrices: {dimension: P} or {dimension: {layer_id: P}}
                dim = constraint.dimensions[0] if hasattr(constraint, 'dimensions') and constraint.dimensions else "safety"
                P_data = constraint.projection_matrices.get(dim) if hasattr(constraint, 'projection_matrices') else None
                
                # 处理分层和融合两种情况
                if isinstance(P_data, dict):
                    # 分层的情况：从模块名提取layer_id
                    import re
                    layer_match = re.search(r'\.layers\.(\d+)\.', name)
                    if layer_match:
                        layer_id = int(layer_match.group(1))
                        C = P_data.get(layer_id)
                    else:
                        C = None
                else:
                    # 融合的情况：直接使用
                    C = P_data
                
                # 如果没有找到投影矩阵，使用单位矩阵（等价于不投影）
                if C is None:
                    C = torch.eye(d_out, device=B_init.device, dtype=B_init.dtype)
                else:
                    C = C.to(device=B_init.device, dtype=B_init.dtype)
                
                if C.shape[0] != d_out:
                    if verbose:
                        print(f"  ⚠️ C 形状 {C.shape} 与 d_out {d_out} 不匹配，跳过投影")
                else:
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
