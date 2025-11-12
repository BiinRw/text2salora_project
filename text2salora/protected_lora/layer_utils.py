"""
层区间解析工具函数
"""

def parse_layer_ranges(layer_spec: str, total_layers: int) -> list:
    """
    解析层区间字符串
    
    Args:
        layer_spec: 层区间字符串，如 "all", "8-16", "8-16,20-24"
        total_layers: 模型总层数
    
    Returns:
        list: 应该应用约束的层索引列表
    
    Examples:
        >>> parse_layer_ranges("all", 28)
        [0, 1, 2, ..., 27]
        >>> parse_layer_ranges("8-16", 28)
        [8, 9, 10, 11, 12, 13, 14, 15, 16]
        >>> parse_layer_ranges("8-16,20-24", 28)
        [8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24]
    """
    if layer_spec.lower() == 'all':
        return list(range(total_layers))
    
    constrained_layers = []
    
    # 分割多个区间（用逗号分隔）
    ranges = layer_spec.split(',')
    
    for range_str in ranges:
        range_str = range_str.strip()
        
        # 解析单个区间 "start-end"
        if '-' in range_str:
            parts = range_str.split('-')
            if len(parts) != 2:
                raise ValueError(f"Invalid range format: {range_str}. Expected format: 'start-end'")
            
            try:
                start = int(parts[0].strip())
                end = int(parts[1].strip())
            except ValueError:
                raise ValueError(f"Invalid range: {range_str}. Start and end must be integers.")
            
            if start < 0 or end >= total_layers:
                raise ValueError(f"Range {range_str} is out of bounds. Model has {total_layers} layers (0-{total_layers-1}).")
            
            if start > end:
                raise ValueError(f"Invalid range: {range_str}. Start must be <= end.")
            
            # 添加区间内的所有层（包含 start 和 end）
            constrained_layers.extend(range(start, end + 1))
        else:
            # 单个层号
            try:
                layer_id = int(range_str)
            except ValueError:
                raise ValueError(f"Invalid layer specification: {range_str}. Must be an integer.")
            
            if layer_id < 0 or layer_id >= total_layers:
                raise ValueError(f"Layer {layer_id} is out of bounds. Model has {total_layers} layers (0-{total_layers-1}).")
            
            constrained_layers.append(layer_id)
    
    # 去重并排序
    constrained_layers = sorted(list(set(constrained_layers)))
    
    return constrained_layers


def should_constrain_layer(layer_name: str, constrained_layers: list) -> bool:
    """
    判断某一层是否应该应用约束
    
    Args:
        layer_name: 层的名字，如 "base_model.model.model.layers.8.self_attn.q_proj"
        constrained_layers: 应该约束的层索引列表
    
    Returns:
        bool: 是否应该约束这一层
    """
    # 从层名中提取层号
    # 格式: base_model.model.model.layers.{layer_id}.xxx
    import re
    
    # 匹配 .layers.数字. 或 .layers.数字（结尾）
    pattern = r'\.layers\.(\d+)'
    match = re.search(pattern, layer_name)
    
    if match:
        layer_id = int(match.group(1))
        return layer_id in constrained_layers
    
    # 如果无法提取层号，默认不约束（可能是其他类型的层）
    return False


if __name__ == '__main__':
    # 测试用例
    print("测试层区间解析:")
    
    print("\n1. 测试 'all':")
    result = parse_layer_ranges('all', 28)
    print(f"   结果: 共 {len(result)} 层")
    print(f"   前5层: {result[:5]}")
    print(f"   后5层: {result[-5:]}")
    
    print("\n2. 测试 '8-16':")
    result = parse_layer_ranges('8-16', 28)
    print(f"   结果: {result}")
    
    print("\n3. 测试 '8-16,20-24':")
    result = parse_layer_ranges('8-16,20-24', 28)
    print(f"   结果: {result}")
    
    print("\n4. 测试 '0-5,10,15,20-22':")
    result = parse_layer_ranges('0-5,10,15,20-22', 28)
    print(f"   结果: {result}")
    
    print("\n5. 测试 should_constrain_layer:")
    constrained = parse_layer_ranges('8-16', 28)
    test_names = [
        "base_model.model.model.layers.5.self_attn.q_proj",
        "base_model.model.model.layers.10.self_attn.q_proj",
        "base_model.model.model.layers.20.self_attn.v_proj",
    ]
    for name in test_names:
        result = should_constrain_layer(name, constrained)
        print(f"   {name}: {result}")
