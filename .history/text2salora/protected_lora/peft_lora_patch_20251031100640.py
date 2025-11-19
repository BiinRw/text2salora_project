"""
Monkey Patch PEFT LoRA Layer 以支持 SaLoRA 风格的硬约束
直接修改 forward 方法，在 LoRA 输出后添加 @ C^T 投影
"""

import torch
import types
from typing import Optional


def patch_lora_linear_forward(lora_module, lora_C: Optional[torch.Tensor] = None):
    """
    Patch PEFT LoRA Linear 的 forward 方法，添加 SaLoRA 风格的投影
    
    修改逻辑:
    原始: output = base(x) + lora_B(lora_A(x)) * scaling
    修改: output = base(x) + (lora_C @ (lora_B(lora_A(x)) * scaling)
    
    Args:
        lora_module: PEFT LoRA Linear 模块
        lora_C: 投影矩阵 C (hidden_dim, hidden_dim)，会自动裁剪到 (out_dim, out_dim)
    """
    # 注入 lora_C 属性
    if lora_C is not None:
        # 🔑 关键修复: 根据该层的 out_dim 裁剪 C 矩阵
        # 获取该层的 out_dim
        out_dim = lora_module.base_layer.out_features
        hidden_dim = lora_C.shape[0]
        
        if out_dim == hidden_dim:
            # 维度匹配，直接使用
            # 注册为 buffer 而不是普通属性，确保不参与梯度计算
            #lora_module.register_buffer('lora_C', lora_C.clone(), persistent=False)
            lora_module.lora_C = lora_C.clone().detach()
            lora_module.lora_C.requires_grad_(False)
        elif out_dim < hidden_dim:
            # 需要裁剪：只使用前 out_dim 维度
            # C_small = C[:out_dim, :out_dim]
            #lora_module.register_buffer('lora_C', lora_C[:out_dim, :out_dim].clone(), persistent=False)
            lora_module.lora_C = lora_C[:out_dim, :out_dim].clone().detach()
            lora_module.lora_C.requires_grad_(False)
            print(f"   🔧 裁剪 C: {hidden_dim}x{hidden_dim} → {out_dim}x{out_dim}")
        else:
            # out_dim > hidden_dim，不应该发生
            print(f"   ⚠️  警告: out_dim ({out_dim}) > hidden_dim ({hidden_dim})，使用单位矩阵")
            #lora_module.register_buffer('lora_C', torch.eye(out_dim, dtype=lora_C.dtype, device=lora_C.device), persistent=False)
            # ✅ 直接赋值属性，不注册 buffer，更安全
            lora_module.lora_C = torch.eye(
                out_dim, dtype=lora_C.dtype, device=lora_C.device
            ).detach()
            lora_module.lora_C.requires_grad_(False)

        
    else:
        lora_module.lora_C = None
    
    # 保存原始 forward (如果还没保存)
    if not hasattr(lora_module, '_original_lora_forward'):
        lora_module._original_lora_forward = lora_module.forward
    
    # 定义新的 forward 方法
    def patched_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        修改后的 forward，支持 SaLoRA 投影
        """
        # 检查是否禁用 adapter
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
            return result
        
        # 检查是否已合并
        if self.merged:
            result = self.base_layer(x, *args, **kwargs)
            return result
        
        # 正常情况: base + lora
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        
        # 遍历所有活跃的 adapter
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            
            x_lora = x.to(lora_A.weight.dtype)
            
            # 计算 LoRA 输出
            if not self.use_dora[active_adapter]:
                # 标准 LoRA: lora_B(lora_A(dropout(x))) * scaling
                lora_output = lora_B(lora_A(dropout(x_lora))) * scaling
                
                # 🔑 关键修改: 如果有 lora_C，投影到正交补空间
                if hasattr(self, 'lora_C') and self.lora_C is not None:
                    out_dim = lora_output.size(-1)
                    lora_C = self.lora_C.detach().to(lora_output.device, dtype=lora_output.dtype)
                    C_block = lora_C

                    # ✅ 自动匹配 C 的尺寸到 out_dim
                    if C_block.shape[0] != out_dim or C_block.shape[1] != out_dim:
                        if C_block.shape[0] > out_dim and C_block.shape[1] > out_dim:
                            # 比当前层大 → 裁剪
                            C_block = C_block[:out_dim, :out_dim]
                        elif C_block.shape[0] < out_dim or C_block.shape[1] < out_dim:
                            # 比当前层小 → 扩展成 block 对角阵（重复C）
                            repeat_factor = math.ceil(out_dim / C_block.shape[0])
                            C_block = torch.block_diag(*([C_block] * repeat_factor))[:out_dim, :out_dim]
                        else:
                            # 完全不匹配 → 单位矩阵 fallback
                            C_block = torch.eye(out_dim, device=C_block.device, dtype=C_block.dtype)

                    # ✅ 左乘（正确方向）
                    lora_output = torch.matmul(C_block, lora_output.T).T


                result = result + lora_output
            else:
                # DoRA
                x_lora = dropout(x_lora)
                result = result + self._apply_dora(x_lora, lora_A, lora_B, scaling, active_adapter)
        
        result = result.to(torch_result_dtype)
        return result
    
    # 绑定新的 forward 方法
    lora_module.forward = types.MethodType(patched_forward, lora_module)


def inject_hard_constraint_to_model(
    model,
    lora_C: torch.Tensor,
    verbose: bool = True
) -> int:
    """
    为模型的所有 LoRA 层注入硬约束
    
    Args:
        model: PEFT 模型
        lora_C: 投影矩阵 C (shape: hidden_dim, hidden_dim)
        verbose: 是否打印详细信息
        
    Returns:
        patched_count: 修改的层数
    """
    patched_count = 0
    
    if verbose:
        print(f"\n🔧 注入 SaLoRA 硬约束到模型...")
        print(f"   投影矩阵 C shape: {lora_C.shape}")
    
    for name, module in model.named_modules():
        # 查找 PEFT LoRA Linear 层
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # Patch forward
            patch_lora_linear_forward(module, lora_C)
            patched_count += 1
            
            if verbose and patched_count <= 3:
                out_dim = module.base_layer.out_features
                print(f"   ✓ {name}: out_dim={out_dim}, 已注入 lora_C")
    
    if verbose:
        print(f"   📊 共修改 {patched_count} 个 LoRA 层")
        print(f"   🔒 约束: LoRA 输出 @ C^T (硬约束， 投影到正交补空间)")
    
    return patched_count


if __name__ == '__main__':
    print("测试 PEFT LoRA Patch...")
    
    # 模拟测试
    import torch.nn as nn
    
    class MockLoRALayer(nn.Module):
        def __init__(self, in_dim=1024, out_dim=1024):
            super().__init__()
            self.base_layer = nn.Linear(in_dim, out_dim)
            self.lora_A = {'default': nn.Linear(in_dim, 8, bias=False)}
            self.lora_B = {'default': nn.Linear(8, out_dim, bias=False)}
            self.lora_dropout = {'default': nn.Identity()}
            self.scaling = {'default': 1.0}
            self.active_adapters = ['default']
            self.disable_adapters = False
            self.merged = False
            self.use_dora = {'default': False}
        
        def forward(self, x):
            return self.base_layer(x) + self.lora_B['default'](
                self.lora_A['default'](x)
            ) * self.scaling['default']
    
    # 创建测试层
    layer = MockLoRALayer(in_dim=1024, out_dim=1024)
    
    # 创建投影矩阵
    C = torch.eye(1536)  # 更大的矩阵
    
    # Patch
    patch_lora_linear_forward(layer, C)
    
    # 测试
    x = torch.randn(2, 10, 1024)
    output = layer(x)
    
    print(f"\n✅ Patch 测试成功")
    print(f"   输入 shape: {x.shape}")
    print(f"   输出 shape: {output.shape}")
    print(f"   lora_C shape: {layer.lora_C.shape}")
    print(f"   lora_C 已注入: {hasattr(layer, 'lora_C')}")
