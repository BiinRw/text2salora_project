#!/bin/bash

# 改进的探针训练脚本
# 主要改进:
# 1. 保存训练好的探针模型(.pkl)
# 2. 增加迭代次数从1000到2000
# 3. 添加交叉验证评估
# 4. 调整正则化参数C
# 5. 更详细的训练报告

MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
DEVICE="cuda:0"

DATA_DIR="data"
RESULTS_DIR="results_improved"
PROBES_DIR="trained_probes"

mkdir -p $RESULTS_DIR
mkdir -p $PROBES_DIR

echo "========================================="
echo "🚀 改进的线性探针训练"
echo "========================================="
echo "模型: $MODEL_NAME"
echo "设备: $DEVICE"
echo "数据目录: $DATA_DIR"
echo "结果目录: $RESULTS_DIR"
echo "探针目录: $PROBES_DIR"
echo "========================================="

# 训练安全性探针 (最重要的!)
echo ""
echo "📊 1/1 训练安全性探针..."
python train_probe_paired_improved.py \
    --model_name $MODEL_NAME \
    --device $DEVICE \
    --safe_pairs ${DATA_DIR}/safety_paired/safe_pairs.json \
    --harmful_pairs ${DATA_DIR}/safety_paired/harmful_pairs.json \
    --max_samples 500 \
    --test_split 0.2 \
    --cv_folds 5 \
    --max_iter 2000 \
    --reg_C 1.0 \
    --output_file ${RESULTS_DIR}/probe_safety.json \
    --probe_dir ${PROBES_DIR}/safety

echo ""
echo "========================================="
echo "✅ 训练完成!"
echo "========================================="
echo "📂 结果文件: $RESULTS_DIR/"
echo "📂 探针模型: $PROBES_DIR/"
echo "========================================="

# 生成训练报告
echo ""
echo "📝 生成训练报告..."
python << 'EOFREPORT'
import json
import os

results_dir = "results_improved"
output_file = os.path.join(results_dir, "training_report.txt")

with open(output_file, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("📊 改进探针训练报告\n")
    f.write("=" * 70 + "\n\n")
    
    # 安全性探针
    result_file = os.path.join(results_dir, "probe_safety.json")
    if os.path.exists(result_file):
        with open(result_file, 'r') as rf:
            results = json.load(rf)
        
        test_accs = [r['test_accuracy'] for r in results.values()]
        cv_means = [r['cv_mean'] for r in results.values()]
        
        f.write("🔒 安全性探针 (Safety)\n")
        f.write("-" * 70 + "\n")
        f.write(f"总注意力头数: {len(results)}\n")
        f.write(f"平均测试准确率: {sum(test_accs)/len(test_accs):.4f}\n")
        f.write(f"最高测试准确率: {max(test_accs):.4f}\n")
        f.write(f"平均CV准确率: {sum(cv_means)/len(cv_means):.4f}\n")
        f.write(f"最高CV准确率: {max(cv_means):.4f}\n")
        f.write(f"准确率 >= 0.8: {sum(1 for a in test_accs if a >= 0.8)}\n")
        f.write(f"准确率 >= 0.9: {sum(1 for a in test_accs if a >= 0.9)}\n")
        
        # Top 10
        f.write("\n🏆 Top 10 最佳注意力头:\n")
        top_10 = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)[:10]
        for i, (head, metrics) in enumerate(top_10, 1):
            f.write(f"  {i}. {head}: test={metrics['test_accuracy']:.4f}, cv={metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}\n")
        
        f.write("\n")
    
    f.write("=" * 70 + "\n")
    f.write("✅ 报告生成完成!\n")
    f.write(f"💾 探针模型已保存到: trained_probes/safety/linear_probes.pkl\n")
    f.write("=" * 70 + "\n")

print(f"✅ 报告已保存到: {output_file}")
EOFREPORT

echo ""
echo "📖 查看报告:"
cat ${RESULTS_DIR}/training_report.txt

