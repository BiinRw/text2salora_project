"""
偏好子空间 SVD 分解
对特征差分执行奇异值分解,提取偏好子空间基向量
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt


def load_feature_diff(feature_file: str) -> Dict[int, np.ndarray]:
    """加载特征差分文件
    
    Args:
        feature_file: 特征差分文件路径
        
    Returns:
        layer_diffs: {layer_id: diff_array}
    """
    print(f"📂 加载特征差分: {feature_file}")
    data = np.load(feature_file)
    
    layer_diffs = {}
    num_layers = int(data['num_layers'])
    
    for layer_id in range(num_layers):
        key = f'layer_{layer_id}'
        if key in data:
            layer_diffs[layer_id] = data[key]
    
    print(f"   ✅ 加载 {len(layer_diffs)} 层的特征差分")
    print(f"   样本数: {data['num_samples']}")
    print(f"   隐藏层维度: {data['hidden_size']}")
    
    return layer_diffs


def compute_svd_for_layer(
    diff: np.ndarray,
    top_k: int = 64,
    device: str = 'cuda:0'
) -> Dict:
    """对单层的特征差分执行 SVD 分解
    
    Args:
        diff: 特征差分矩阵 (N, d)
        top_k: 保留的奇异向量数量
        device: 计算设备
        
    Returns:
        subspace_dict: {U, S, V, explained_variance_ratio}
    """
    # 转换为 tensor
    diff_tensor = torch.from_numpy(diff).float().to(device)
    
    # SVD 分解
    U, S, V = torch.svd_lowrank(diff_tensor, q=top_k, niter=2)
    
    # 计算方差解释率
    total_variance = torch.sum(S ** 2)
    explained_variance = torch.cumsum(S ** 2, dim=0) / total_variance
    
    return {
        'U': U.cpu(),  # (N, top_k)
        'S': S.cpu(),  # (top_k,)
        'V': V.cpu(),  # (d, top_k) - 这是偏好子空间的基向量!
        'explained_variance_ratio': explained_variance.cpu(),
        'total_variance': total_variance.cpu().item()
    }


def compute_multi_layer_svd(
    layer_diffs: Dict[int, np.ndarray],
    top_k: int = 64,
    device: str = 'cuda:0',
    layer_selection: Optional[List[int]] = None
) -> Dict[int, Dict]:
    """对多层执行 SVD 分解
    
    Args:
        layer_diffs: {layer_id: diff_array}
        top_k: 保留的奇异向量数量
        device: 设备
        layer_selection: 选择特定层 (None=所有层)
        
    Returns:
        layer_subspaces: {layer_id: subspace_dict}
    """
    if layer_selection is None:
        layer_selection = sorted(layer_diffs.keys())
    
    print(f"\n🔬 对 {len(layer_selection)} 层执行 SVD 分解 (top_k={top_k})")
    
    layer_subspaces = {}
    
    for layer_id in layer_selection:
        if layer_id not in layer_diffs:
            print(f"   ⚠️  Layer {layer_id} 不存在,跳过")
            continue
        
        diff = layer_diffs[layer_id]
        print(f"\n   Layer {layer_id:2d}: shape={diff.shape}")
        
        subspace = compute_svd_for_layer(diff, top_k, device)
        layer_subspaces[layer_id] = subspace
        
        # 打印方差解释率
        ev_ratio = subspace['explained_variance_ratio']
        print(f"      Top 10 奇异值解释方差: {ev_ratio[9].item():.4f}")
        print(f"      Top 32 奇异值解释方差: {ev_ratio[31].item():.4f}")
        print(f"      Top 64 奇异值解释方差: {ev_ratio[-1].item():.4f}")
    
    return layer_subspaces


def fuse_multi_layer_subspace(
    layer_subspaces: Dict[int, Dict],
    method: str = 'weighted_avg',
    weights: Optional[Dict[int, float]] = None
) -> Dict:
    """融合多层子空间
    
    Args:
        layer_subspaces: {layer_id: subspace_dict}
        method: 融合方法 ('weighted_avg', 'concat', 'avg')
        weights: 层权重 {layer_id: weight} (用于 weighted_avg)
        
    Returns:
        fused_subspace: 融合后的子空间
    """
    if method == 'concat':
        # 拼接多层的 V 矩阵
        V_list = [subspace['V'] for subspace in layer_subspaces.values()]
        V_fused = torch.cat(V_list, dim=1)  # (d, num_layers * top_k)
        
        return {
            'V': V_fused,
            'method': 'concat',
            'num_layers': len(layer_subspaces)
        }
    
    elif method in ['avg', 'weighted_avg']:
        # 加权平均多层的 V 矩阵
        if weights is None:
            # 均等权重
            weights = {layer_id: 1.0 / len(layer_subspaces) 
                      for layer_id in layer_subspaces.keys()}
        else:
            # 归一化权重
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}
        
        V_fused = None
        for layer_id, subspace in layer_subspaces.items():
            V = subspace['V']
            w = weights.get(layer_id, 0.0)
            
            if V_fused is None:
                V_fused = w * V
            else:
                V_fused += w * V
        
        return {
            'V': V_fused,
            'method': method,
            'weights': weights,
            'num_layers': len(layer_subspaces)
        }
    
    else:
        raise ValueError(f"Unknown fusion method: {method}")


def save_subspaces(
    layer_subspaces: Dict[int, Dict],
    dimension: str,
    output_dir: str,
    fused_subspace: Optional[Dict] = None
):
    """保存子空间
    
    Args:
        layer_subspaces: {layer_id: subspace_dict}
        dimension: 偏好维度
        output_dir: 输出目录
        fused_subspace: 融合的子空间 (可选)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 保存各层子空间
    for layer_id, subspace in layer_subspaces.items():
        filename = output_dir / f'{dimension}_layer{layer_id}_subspace.pt'
        torch.save({
            'V': subspace['V'],
            'S': subspace['S'],
            'U': subspace['U'],
            'explained_variance_ratio': subspace['explained_variance_ratio'],
            'layer_id': layer_id,
            'dimension': dimension
        }, filename)
        print(f"   ✅ Layer {layer_id}: {filename}")
    
    # 2. 保存融合子空间
    if fused_subspace is not None:
        filename = output_dir / f'{dimension}_fused_subspace.pt'
        torch.save({
            'V': fused_subspace['V'],
            'method': fused_subspace['method'],
            'dimension': dimension,
            **{k: v for k, v in fused_subspace.items() 
               if k not in ['V', 'method', 'dimension']}
        }, filename)
        print(f"   ✅ Fused: {filename}")
    
    # 3. 保存元信息
    meta_info = {
        'dimension': dimension,
        'num_layers': len(layer_subspaces),
        'layer_ids': list(layer_subspaces.keys()),
        'top_k': layer_subspaces[list(layer_subspaces.keys())[0]]['V'].shape[1],
        'hidden_size': layer_subspaces[list(layer_subspaces.keys())[0]]['V'].shape[0]
    }
    
    if fused_subspace is not None:
        meta_info['fused_method'] = fused_subspace['method']
        meta_info['fused_shape'] = list(fused_subspace['V'].shape)
    
    meta_file = output_dir / f'{dimension}_meta.json'
    with open(meta_file, 'w') as f:
        json.dump(meta_info, f, indent=2)
    print(f"   ✅ Meta: {meta_file}")


def plot_singular_values(
    layer_subspaces: Dict[int, Dict],
    dimension: str,
    output_dir: str
):
    """可视化奇异值分布
    
    Args:
        layer_subspaces: {layer_id: subspace_dict}
        dimension: 偏好维度
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{dimension.capitalize()} Dimension - Singular Values Analysis', 
                 fontsize=16)
    
    # 1. 奇异值大小
    ax = axes[0, 0]
    for layer_id, subspace in sorted(layer_subspaces.items()):
        S = subspace['S'].numpy()
        ax.plot(S, label=f'Layer {layer_id}', alpha=0.7)
    ax.set_xlabel('Rank')
    ax.set_ylabel('Singular Value')
    ax.set_title('Singular Values by Layer')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. 方差解释率
    ax = axes[0, 1]
    for layer_id, subspace in sorted(layer_subspaces.items()):
        ev = subspace['explained_variance_ratio'].numpy()
        ax.plot(ev, label=f'Layer {layer_id}', alpha=0.7)
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('Explained Variance Ratio')
    ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90%')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. Log scale 奇异值
    ax = axes[1, 0]
    for layer_id, subspace in sorted(layer_subspaces.items()):
        S = subspace['S'].numpy()
        ax.semilogy(S, label=f'Layer {layer_id}', alpha=0.7)
    ax.set_xlabel('Rank')
    ax.set_ylabel('Singular Value (log scale)')
    ax.set_title('Singular Values (Log Scale)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. 各层前10个奇异值对比
    ax = axes[1, 1]
    layer_ids = sorted(layer_subspaces.keys())
    top_10_values = []
    for layer_id in layer_ids:
        S = subspace['S'].numpy()[:10]
        top_10_values.append(S)
    
    top_10_values = np.array(top_10_values)
    x = np.arange(len(layer_ids))
    width = 0.08
    
    for i in range(min(10, top_10_values.shape[1])):
        ax.bar(x + i * width, top_10_values[:, i], width, 
               label=f'SV {i+1}', alpha=0.7)
    
    ax.set_xlabel('Layer ID')
    ax.set_ylabel('Singular Value')
    ax.set_title('Top 10 Singular Values by Layer')
    ax.set_xticks(x + width * 4.5)
    ax.set_xticklabels(layer_ids)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_file = output_dir / f'{dimension}_singular_values.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 奇异值可视化已保存: {output_file}")
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='计算偏好子空间 SVD')
    parser.add_argument('--feature_file', type=str, required=True,
                        help='特征差分文件路径')
    parser.add_argument('--dimension', type=str, required=True,
                        help='偏好维度名称')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--top_k', type=int, default=64,
                        help='保留的奇异向量数量')
    parser.add_argument('--layers', type=str, default=None,
                        help='选择特定层,逗号分隔 (如: 15,16,17,18)')
    parser.add_argument('--fuse_method', type=str, default='weighted_avg',
                        choices=['weighted_avg', 'concat', 'avg', 'none'],
                        help='多层融合方法')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='计算设备')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"🔬 计算 {args.dimension} 维度的偏好子空间")
    print("=" * 80)
    
    # 1. 加载特征差分
    layer_diffs = load_feature_diff(args.feature_file)
    
    # 2. 选择层
    if args.layers:
        layer_selection = [int(x) for x in args.layers.split(',')]
        print(f"\n📌 选择层: {layer_selection}")
    else:
        layer_selection = None
        print(f"\n📌 使用所有层")
    
    # 3. 计算 SVD
    layer_subspaces = compute_multi_layer_svd(
        layer_diffs,
        top_k=args.top_k,
        device=args.device,
        layer_selection=layer_selection
    )
    
    # 4. 融合子空间
    fused_subspace = None
    if args.fuse_method != 'none':
        print(f"\n🔗 融合多层子空间 (方法: {args.fuse_method})")
        fused_subspace = fuse_multi_layer_subspace(
            layer_subspaces,
            method=args.fuse_method
        )
        print(f"   ✅ 融合后形状: {fused_subspace['V'].shape}")
    
    # 5. 保存
    print(f"\n💾 保存子空间:")
    save_subspaces(
        layer_subspaces,
        args.dimension,
        args.output_dir,
        fused_subspace
    )
    
    # 6. 可视化
    plot_singular_values(
        layer_subspaces,
        args.dimension,
        args.output_dir
    )
    
    print(f"\n✅ 完成!")
