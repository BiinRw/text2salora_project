"""
为test_multi_position_probe_accuracy_with_abc.py添加层约束支持
"""

# 读取原文件
with open('test_multi_position_probe_accuracy_with_abc.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. 修改ABCConstraintLoader.__init__,添加constrained_layers参数
old_init = """    def __init__(self, subspace_dir, dimension, device='cuda:0'):
        self.subspace_dir = Path(subspace_dir)
        self.dimension = dimension
        self.device = device
        self.subspaces = {}"""

new_init = """    def __init__(self, subspace_dir, dimension, device='cuda:0', constrained_layers=None):
        self.subspace_dir = Path(subspace_dir)
        self.dimension = dimension
        self.device = device
        self.subspaces = {}
        self.constrained_layers = constrained_layers  # None表示所有层, 或者(start, end)元组表示层范围"""

content = content.replace(old_init, new_init)

# 2. 修改compute_constraint_matrix,检查层是否需要约束
old_compute = """    def compute_constraint_matrix(self, layer_id, hidden_dim):
        """
        Compute constraint matrix C = I - V @ V^T
        
        Args:
            layer_id: Layer index
            hidden_dim: Hidden dimension size (e.g., 1536)
        
        Returns:
            C: Constraint matrix of shape (hidden_dim, hidden_dim)
        """
        if layer_id not in self.subspaces:
            return torch.eye(hidden_dim, device=self.device)
        
        V = self.subspaces[layer_id]  # Shape: (hidden_dim, subspace_rank)
        I = torch.eye(hidden_dim, device=self.device)
        
        # C = I - V @ V^T, shape: (hidden_dim, hidden_dim)
        C = I - torch.mm(V, V.t())
        
        return C"""

new_compute = """    def compute_constraint_matrix(self, layer_id, hidden_dim):
        """
        Compute constraint matrix C = I - V @ V^T
        
        Args:
            layer_id: Layer index
            hidden_dim: Hidden dimension size (e.g., 1536)
        
        Returns:
            C: Constraint matrix of shape (hidden_dim, hidden_dim)
        """
        # 检查该层是否需要应用约束
        if self.constrained_layers is not None:
            start, end = self.constrained_layers
            if not (start <= layer_id <= end):
                # 该层不在约束范围内,返回单位矩阵(无约束)
                return torch.eye(hidden_dim, device=self.device)
        
        if layer_id not in self.subspaces:
            return torch.eye(hidden_dim, device=self.device)
        
        V = self.subspaces[layer_id]  # Shape: (hidden_dim, subspace_rank)
        I = torch.eye(hidden_dim, device=self.device)
        
        # C = I - V @ V^T, shape: (hidden_dim, hidden_dim)
        C = I - torch.mm(V, V.t())
        
        return C"""

content = content.replace(old_compute, new_compute)

# 3. 修改apply_constraint_to_lora,添加日志
old_apply = """    def apply_constraint_to_lora(self, lora_A, lora_B, layer_id):
        """
        Apply ABC constraint: Delta_W = B @ A @ C
        
        Args:
            lora_A: LoRA A matrix, shape (lora_rank, hidden_dim)
            lora_B: LoRA B matrix, shape (hidden_dim, lora_rank)
            layer_id: Layer index
            
        Returns:
            delta_W: Constrained weight update (hidden_dim, hidden_dim)
        """
        # Get hidden dimension from lora_A
        hidden_dim = lora_A.size(1)
        
        # Compute constraint matrix C
        C = self.compute_constraint_matrix(layer_id, hidden_dim)
        
        # Apply constraint: Delta_W = B @ (A @ C)
        # A: (r, d), C: (d, d) -> A@C: (r, d)
        # B: (d, r), A@C: (r, d) -> B@(A@C): (d, d)
        A_constrained = torch.mm(lora_A, C)
        delta_W = torch.mm(lora_B, A_constrained)
        
        return delta_W"""

new_apply = """    def apply_constraint_to_lora(self, lora_A, lora_B, layer_id):
        """
        Apply ABC constraint: Delta_W = B @ A @ C
        
        Args:
            lora_A: LoRA A matrix, shape (lora_rank, hidden_dim)
            lora_B: LoRA B matrix, shape (hidden_dim, lora_rank)
            layer_id: Layer index
            
        Returns:
            delta_W: Constrained weight update (hidden_dim, hidden_dim)
        """
        # Get hidden dimension from lora_A
        hidden_dim = lora_A.size(1)
        
        # Compute constraint matrix C
        C = self.compute_constraint_matrix(layer_id, hidden_dim)
        
        # 检查是否实际应用了约束(C != I)
        is_identity = torch.allclose(C, torch.eye(hidden_dim, device=self.device), atol=1e-6)
        
        # Apply constraint: Delta_W = B @ (A @ C)
        # A: (r, d), C: (d, d) -> A@C: (r, d)
        # B: (d, r), A@C: (r, d) -> B@(A@C): (d, d)
        A_constrained = torch.mm(lora_A, C)
        delta_W = torch.mm(lora_B, A_constrained)
        
        return delta_W"""

content = content.replace(old_apply, new_apply)

# 4. 修改load_model_with_abc函数签名
old_load_sig = """def load_model_with_abc(model_path, lora_path, subspace_dir, dimension, device='cuda:0'):"""
new_load_sig = """def load_model_with_abc(model_path, lora_path, subspace_dir, dimension, device='cuda:0', constrained_layers=None):"""

content = content.replace(old_load_sig, new_load_sig)

# 5. 修改abc_loader初始化
old_abc_init = """    # Load ABC constraints
    abc_loader = ABCConstraintLoader(subspace_dir, dimension, device)
    has_constraints = abc_loader.load_subspaces()"""

new_abc_init = """    # Load ABC constraints
    abc_loader = ABCConstraintLoader(subspace_dir, dimension, device, constrained_layers)
    has_constraints = abc_loader.load_subspaces()
    
    # 打印层约束信息
    if constrained_layers is not None:
        print(f"   🎯 Constrained layers: {constrained_layers[0]}-{constrained_layers[1]}")
    else:
        print(f"   🎯 Constrained layers: All layers (0-27)")"""

content = content.replace(old_abc_init, new_abc_init)

# 6. 修改argparse添加--constrained_layers参数
old_argparse = """    parser.add_argument('--subspace_dir', type=str,
                       default='preference_subspace/saved_subspaces',
                       help='Subspace directory')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='计算设备')"""

new_argparse = """    parser.add_argument('--subspace_dir', type=str,
                       default='preference_subspace/saved_subspaces',
                       help='Subspace directory')
    parser.add_argument('--constrained_layers', type=str, default=None,
                       help='约束层范围,格式: "start,end" (如 "0,8" 或 "16,16"), None表示所有层')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='计算设备')"""

content = content.replace(old_argparse, new_argparse)

# 7. 修改main函数,解析constrained_layers参数
old_main_call = """    # 1. 加载模型
    model, tokenizer, model_type = load_model_with_abc(
        args.model_path,
        args.lora_path,
        args.subspace_dir,
        args.dimension,
        args.device
    )"""

new_main_call = """    # 解析层约束参数
    constrained_layers = None
    if args.constrained_layers:
        start, end = map(int, args.constrained_layers.split(','))
        constrained_layers = (start, end)
        print(f"🎯 将约束应用于层: {start}-{end}")
    else:
        print(f"🎯 将约束应用于所有层")
    
    # 1. 加载模型
    model, tokenizer, model_type = load_model_with_abc(
        args.model_path,
        args.lora_path,
        args.subspace_dir,
        args.dimension,
        args.device,
        constrained_layers
    )"""

content = content.replace(old_main_call, new_main_call)

# 保存修改后的文件
with open('test_multi_position_probe_accuracy_with_abc.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ 代码修改完成!")
print("\n修改内容:")
print("1. ✅ ABCConstraintLoader添加constrained_layers参数")
print("2. ✅ compute_constraint_matrix检查层范围")
print("3. ✅ load_model_with_abc添加constrained_layers参数")
print("4. ✅ argparse添加--constrained_layers参数")
print("5. ✅ main函数解析层约束参数")
