#!/bin/bash

# 修正版: 训练所有维度的探针
# 关键修改: 使用0/1标签的数据

MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
DEVICE="cuda:3"

DATA_DIR="data"
RESULTS_DIR="results_all_dimensions"
PROBES_DIR="trained_probes"

mkdir -p $RESULTS_DIR
mkdir -p $PROBES_DIR

echo "========================================="
echo "🚀 修正版: 训练所有维度探针"
echo "========================================="
echo "模型: $MODEL_NAME"
echo "设备: $DEVICE"
echo "数据: 0/1标签配对数据"
echo "========================================="

# 1. 训练安全性探针
echo ""
echo "📊 [1/6] 训练安全性探针..."
python train_probe_paired_improved.py \
    --model_name $MODEL_NAME \
    --device $DEVICE \
    --safe_pairs ${DATA_DIR}/safety_paired/safe_pairs_large.json \
    --harmful_pairs ${DATA_DIR}/safety_paired/harmful_pairs_large.json \
    --max_samples 5000 \
    --test_split 0.2 \
    --cv_folds 5 \
    --max_iter 2000 \
    --reg_C 1.0 \
    --output_file ${RESULTS_DIR}/probe_safety.json \
    --probe_dir ${PROBES_DIR}/safety

# 2-6. 训练HelpSteer各维度探针
dimensions=("helpfulness" "correctness" "coherence" "complexity" "verbosity")
dim_names=("有用性" "正确性" "连贯性" "复杂性" "冗长度")

for i in "${!dimensions[@]}"; do
    dim=${dimensions[$i]}
    name=${dim_names[$i]}
    idx=$((i+2))
    
    echo ""
    echo "📊 [$idx/6] 训练${name}探针 ($dim)..."
    python train_helpsteer_dimension.py \
        --model_name $MODEL_NAME \
        --device $DEVICE \
        --good_pairs ${DATA_DIR}/helpsteer_paired/${dim}_good_pairs.json \
        --bad_pairs ${DATA_DIR}/helpsteer_paired/${dim}_bad_pairs.json \
        --max_samples 5000 \
        --test_split 0.2 \
        --cv_folds 5 \
        --max_iter 2000 \
        --reg_C 1.0 \
        --output_file ${RESULTS_DIR}/probe_${dim}.json \
        --probe_dir ${PROBES_DIR}/${dim}
done

echo ""
echo "========================================="
echo "✅ 所有维度训练完成!"
echo "========================================="
echo "📂 结果目录: $RESULTS_DIR/"
echo "📂 探针目录: $PROBES_DIR/"
echo "========================================="

# 生成汇总报告
echo ""
echo "📝 生成汇总报告..."
python << 'EOFREPORT'
import json
import os
import numpy as np

results_dir = "results_all_dimensions"
output_file = os.path.join(results_dir, "training_summary.txt")

dimensions = [
    ('safety', '安全性'),
    ('helpfulness', '有用性'),
    ('correctness', '正确性'),
    ('coherence', '连贯性'),
    ('complexity', '复杂性'),
    ('verbosity', '冗长度')
]

with open(output_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("📊 所有维度探针训练汇总报告\n")
    f.write("=" * 80 + "\n\n")
    
    all_stats = []
    
    for dim_key, dim_name in dimensions:
        result_file = os.path.join(results_dir, f"probe_{dim_key}.json")
        
        if not os.path.exists(result_file):
            f.write(f"❌ {dim_name} ({dim_key}): 结果文件不存在\n\n")
            continue
        
        with open(result_file, 'r') as rf:
            results = json.load(rf)
        
        test_accs = [r['test_accuracy'] for r in results.values()]
        cv_means = [r['cv_mean'] for r in results.values()]
        
        stats = {
            'dimension': dim_name,
            'key': dim_key,
            'total_heads': len(results),
            'avg_test': np.mean(test_accs),
            'max_test': np.max(test_accs),
            'min_test': np.min(test_accs),
            'avg_cv': np.mean(cv_means),
            'max_cv': np.max(cv_means),
            'ge_80': sum(1 for a in test_accs if a >= 0.8),
            'ge_90': sum(1 for a in test_accs if a >= 0.9)
        }
        all_stats.append(stats)
        
        f.write(f"📌 {dim_name} ({dim_key})\n")
        f.write("-" * 80 + "\n")
        f.write(f"总注意力头数: {stats['total_heads']}\n")
        f.write(f"平均测试准确率: {stats['avg_test']:.4f}\n")
        f.write(f"最高测试准确率: {stats['max_test']:.4f}\n")
        f.write(f"最低测试准确率: {stats['min_test']:.4f}\n")
        f.write(f"平均CV准确率: {stats['avg_cv']:.4f}\n")
        f.write(f"最高CV准确率: {stats['max_cv']:.4f}\n")
        f.write(f"准确率 >= 0.8: {stats['ge_80']} 个\n")
        f.write(f"准确率 >= 0.9: {stats['ge_90']} 个\n")
        
        # Top 5
        f.write(f"\n🏆 Top 5 最佳注意力头:\n")
        top_5 = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)[:5]
        for i, (head, metrics) in enumerate(top_5, 1):
            f.write(f"  {i}. {head}: test={metrics['test_accuracy']:.4f}, ")
            f.write(f"cv={metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}\n")
        
        f.write("\n")
    
    # 总体统计
    if all_stats:
        f.write("=" * 80 + "\n")
        f.write("📊 总体统计\n")
        f.write("=" * 80 + "\n")
        
        avg_of_avgs = np.mean([s['avg_test'] for s in all_stats])
        best_dim = max(all_stats, key=lambda x: x['avg_test'])
        worst_dim = min(all_stats, key=lambda x: x['avg_test'])
        
        f.write(f"训练维度数: {len(all_stats)}\n")
        f.write(f"平均准确率: {avg_of_avgs:.4f}\n")
        f.write(f"最佳维度: {best_dim['dimension']} (avg={best_dim['avg_test']:.4f})\n")
        f.write(f"最差维度: {worst_dim['dimension']} (avg={worst_dim['avg_test']:.4f})\n")
        f.write(f"\n总>=0.8头数: {sum(s['ge_80'] for s in all_stats)}\n")
        f.write(f"总>=0.9头数: {sum(s['ge_90'] for s in all_stats)}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("✅ 训练完成!\n")
        f.write(f"💾 探针模型已保存到: trained_probes/\n")
        f.write("=" * 80 + "\n")

print(f"✅ 报告已保存到: {output_file}")
EOFREPORT

echo ""
echo "�� 查看汇总报告:"
cat ${RESULTS_DIR}/training_summary.txt

