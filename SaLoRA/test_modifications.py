"""测试修改后的代码"""
import os

print("=" * 70)
print("�� 验证代码修改")
print("=" * 70)

# 读取文件
with open('lora_train_act.py', 'r') as f:
    content = f.read()

# 检查点1: 保存路径是否使用 f-string
if 'save_path = f"{args.name}_{rank}_{args.lr}_{args.wd}_lora_ABC.pt"' in content:
    print("✅ 修改1: 保存路径已修复")
else:
    print("❌ 修改1: 保存路径未修复")

# 检查点2: 数据集加载是否有错误处理
if 'cache_dir = os.path.expanduser' in content and 'download_mode="reuse_dataset_if_exists"' in content:
    print("✅ 修改2: 数据集加载已改进")
else:
    print("❌ 修改2: 数据集加载未改进")

# 检查点3: Qwen 格式是否正确
if '<|im_start|>user' in content and '<|im_start|>assistant' in content and '<|im_end|>' in content:
    print("✅ 修改3: Qwen2.5 对话格式已修复")
else:
    print("❌ 修改3: Qwen2.5 对话格式未修复")

# 检查点4: 训练输出目录是否使用变量
if 'training_output_dir = f"{args.name}_{rank}_{args.lr}_{args.wd}"' in content:
    print("✅ 修改4: 训练输出目录已改进")
else:
    print("❌ 修改4: 训练输出目录未改进")

# 检查点5: 是否有 Llama2 格式残留
if '[INST]' in content and '[/INST]' in content:
    # 检查是否在注释或字符串中
    lines_with_inst = [line for line in content.split('\n') if '[INST]' in line and 'print' not in line]
    if lines_with_inst:
        print("⚠️  警告: 仍有 Llama2 格式 [INST] 残留")
        for line in lines_with_inst[:3]:
            print(f"   {line.strip()}")
    else:
        print("✅ 修改5: Llama2 格式已清理")
else:
    print("✅ 修改5: Llama2 格式已清理")

print("=" * 70)
print("✅ 所有修改验证完成!")
print("=" * 70)
