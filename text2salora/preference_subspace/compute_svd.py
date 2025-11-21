"""
偏好子空间 SVD 分解 (支持投影层特定文件)
对特征差分执行奇异值分解,提取偏好子空间基向量
v2: 支持读取 {dimension}_{projection}_feature_diff.npz 格式文件
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
    
    # 检查是否有num_layers字段
    if 'num_layers' in data:
        num_layers = int(data['num_layers'])
    else:
        # 旧版本文件,尝试查找所有layer_*键
        layer_keys = [k for k in data.keys() if k.startswith('layer_')]
        num_layers = len(layer_keys)
    
    for layer_id in range(num_layers):
        key = f'layer_{layer_id}'
        if key in data:
            layer_diffs[layer_id] = data[key]
    
    print(f"   ✅ 加载 {len(layer_diffs)} 层的特征差分")
    if 'num_samples' in data:
        print(f"   样本数: {data['num_samples']}")
    if 'hidden_size' in data:
        print(f"   输出维度: {data['hidden_size']}")
    
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
        if len(ev_ratio) >= 64:
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
    
    elif method == 'weighted_avg':
        # 加权平均
        if weights is None:
            # 使用奇异值作为权重
            weights = {
                layer_id: subspace['S'][0].item()
                for layer_id, subspace in layer_subspaces.items()
            }
        
        # 归一化权重
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # 加权平均 V 矩阵
        V_fused = None
        for layer_id, subspace in layer_subspaces.items():
            if V_fused is None:
                V_fused = weights[layer_id] * subspace['V']
            else:
                V_fused += weights[layer_id] * subspace['V']
        
        return {
            'V': V_fused,
            'method': 'weighted_avg',
            'weights': weights
        }
    
    elif method == 'avg':
        # 简单平均
        V_list = [subspace['V'] for subspace in layer_subspaces.values()]
        V_fused = torch.stack(V_list).mean(dim=0)
        
        return {
            'V': V_fused,
            'method': 'avg',
            'num_layers': len(layer_subspaces)
        }
    
    else:
        raise ValueError(f"Unknown fusion method: {method}")


def save_subspaces(
    layer_subspaces: Dict[int, Dict],
    dimension: str,
    projection_type: str,  # 新增: 投影层类型
    output_dir: str,
    fused_subspace: Optional[Dict] = None
):
    """保存子空间到文件
    
    Args:
        layer_subspaces: {layer_id: subspace_dict}
        dimension: 偏好维度名称
        projection_type: 投影层类型
        output_dir: 输出目录
        fused_subspace: 融合后的子空间 (可选)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存各层子空间 (文件名包含projection)
    for layer_id, subspace in layer_subspaces.items():
        filename = output_dir / f'{dimension}_{projection_type}_layer{layer_id}_subspace.pt'
        torch.save({
            'V': subspace['V'],
            'S': subspace['S'],
            'U': subspace['U'],
            'explained_variance_ratio': subspace['explained_variance_ratio'],
            'layer_id': layer_id,
            'dimension': dimension,
            'projection': projection_type  # 新增: 保存投影类型信息
        }, filename)
        print(f"   ✅ 保存: {filename.name}")
    
    # 保存融合子空间
    if fused_subspace is not None:
        filename = output_dir / f'{dimension}_{projection_type}_fused_subspace.pt'
        torch.save({
            'V': fused_subspace['V'],
            'method': fused_subspace['method'],
            'dimension': dimension,
            'projection': projection_type,  # 新增
            **{k: v for k, v in fused_subspace.items() if k not in ['V', 'method']}
        }, filename)
        print(f"   ✅ 保存融合子空间: {filename.name}")
    
    # 保存元信息
    meta_file = output_dir / f'{dimension}_{projection_type}_meta.json'
    meta_info = {
        'dimension': dimension,
        'projection': projection_type,  # 新增
        'num_layers': len(layer_subspaces),
        'layer_ids': sorted(layer_subspaces.keys()),
        'subspace_rank': layer_subspaces[list(layer_subspaces.keys())[0]]['V'].shape[1],
        'fused': fused_subspace is not None,
        'fuse_method': fused_subspace['method'] if fused_subspace else None
    }
    with open(meta_file, 'w') as f:
        json.dump(meta_info, f, indent=2)
    print(f"   ✅ 保存元信息: {meta_file.name}")


def plot_singular_values(
    layer_subspaces: Dict[int, Dict],
    dimension: str,
    projection_type: str,  # 新增
    output_dir: str
):
    """绘制奇异值分布"""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # 1. 所有层的奇异值
    ax = axes[0]
    for layer_id, subspace in layer_subspaces.items():
        S = subspace['S'].numpy()
        ax.plot(S, alpha=0.5, label=f'Layer {layer_id}')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Singular Value')
    ax.set_title(f'{dimension.capitalize()} - {projection_type} - Singular Values')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 2. 方差解释率
    ax = axes[1]
    for layer_id, subspace in layer_subspaces.items():
        ev_ratio = subspace['explained_variance_ratio'].numpy()
        ax.plot(ev_ratio, alpha=0.5, label=f'Layer {layer_id}')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance Ratio')
    ax.set_title(f'{dimension.capitalize()} - {projection_type} - Explained Variance')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 3. 层平均奇异值
    ax = axes[2]
    layer_ids = sorted(layer_subspaces.keys())
    mean_sv = [layer_subspaces[lid]['S'].mean().item() for lid in layer_ids]
    ax.bar(layer_ids, mean_sv)
    ax.set_xlabel('Layer ID')
    ax.set_ylabel('Mean Singular Value')
    ax.set_title(f'{dimension.capitalize()} - {projection_type} - Mean SV per Layer')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Top-k 方差解释率热图
    ax = axes[3]
    k_values = [10, 32, 64]
    ev_matrix = []
    for lid in layer_ids:
        ev_ratio = layer_subspaces[lid]['explained_variance_ratio'].numpy()
        ev_at_k = [ev_ratio[k-1] if k <= len(ev_ratio) else ev_ratio[-1] for k in k_values]
        ev_matrix.append(ev_at_k)
    
    im = ax.imshow(np.array(ev_matrix).T, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(layer_ids)))
    ax.set_xticklabels(layer_ids)
    ax.set_yticks(range(len(k_values)))
    ax.set_yticklabels([f'Top-{k}' for k in k_values])
    ax.set_xlabel('Layer ID')
    ax.set_title(f'{dimension.capitalize()} - {projection_type} - Explained Variance at Top-k')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    output_file = output_dir / f'{dimension}_{projection_type}_singular_values.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✅ 保存可视化: {output_file.name}")
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='计算偏好子空间 SVD (支持投影层)')
    parser.add_argument('--feature_file', type=str, required=True,
                        help='特征差分文件路径 ({dimension}_{projection}_feature_diff.npz)')
    parser.add_argument('--dimension', type=str, required=True,
                        help='偏好维度名称')
    parser.add_argument('--projection', type=str, required=True,
                        help='投影层类型 (q_proj/k_proj/v_proj/o_proj/up_proj/down_proj)')
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
    print(f"🔬 计算 {args.dimension} 维度 - {args.projection} 投影的偏好子空间")
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
        args.projection,  # 新增
        args.output_dir,
        fused_subspace
    )
    
    # 6. 可视化
    print(f"\n📊 生成可视化:")
    plot_singular_values(
        layer_subspaces,
        args.dimension,
        args.projection,  # 新增
        args.output_dir
    )
    
    print("\n" + "=" * 80)
    print("✅ 完成!")
    print("=" * 80)