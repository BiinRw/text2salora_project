#!/bin/bash

# 训练所有维度的多位置探针
# 支持同时训练: user_last, assistant_first, assistant_last, assistant_mean

MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
DEVICE="cuda:3"

DATA_DIR="data"
OUTPUT_DIR="results_multi_position"

mkdir -p $OUTPUT_DIR

echo "========================================="
echo "🚀 多位置探针训练 - 所有维度"
echo "========================================="
echo "模型: $MODEL_NAME"
echo "设备: $DEVICE"
echo "位置: user_last, assistant_first, assistant_last, assistant_mean"
echo "========================================="

# 定义要训练的位置
POSITIONS="user_last assistant_first assistant_last assistant_mean"

# 1. 训练安全性探针
echo ""
echo "📊 [1/6] 训练安全性多位置探针..."
python train_multi_position_probes.py \
    --model_name $MODEL_NAME \
    --device $DEVICE \
    --good_pairs ${DATA_DIR}/safety_paired/safe_pairs_large.json \
    --bad_pairs ${DATA_DIR}/safety_paired/harmful_pairs_large.json \
    --max_samples 1000 \
    --positions $POSITIONS \
    --test_split 0.2 \
    --cv_folds 5 \
    --max_iter 2000 \
    --reg_C 1.0 \
    --output_dir ${OUTPUT_DIR}/safety \
    --dimension safety

# 2-6. 训练HelpSteer各维度探针
dimensions=("helpfulness" "correctness" "coherence" "complexity" "verbosity")
dim_names=("有用性" "正确性" "连贯性" "复杂性" "冗长度")

for i in "${!dimensions[@]}"; do
    dim=${dimensions[$i]}
    name=${dim_names[$i]}
    idx=$((i+2))
    
    echo ""
    echo "📊 [$idx/6] 训练${name}多位置探针 ($dim)..."
    python train_multi_position_probes.py \
        --model_name $MODEL_NAME \
        --device $DEVICE \
        --good_pairs ${DATA_DIR}/helpsteer_paired/${dim}_good_pairs.json \
        --bad_pairs ${DATA_DIR}/helpsteer_paired/${dim}_bad_pairs.json \
        --max_samples 1000 \
        --positions $POSITIONS \
        --test_split 0.2 \
        --cv_folds 5 \
        --max_iter 2000 \
        --reg_C 1.0 \
        --output_dir ${OUTPUT_DIR}/${dim} \
        --dimension ${dim}
done

echo ""
echo "========================================="
echo "✅ 所有维度多位置训练完成!"
echo "========================================="
echo "📂 结果目录: $OUTPUT_DIR/"
echo ""

# 生成总体对比报告
echo "📝 生成总体对比报告..."
python << 'EOFREPORT'
import json
import os
import numpy as np

output_dir = "results_multi_position"
dimensions = [
    ('safety', '安全性'),
    ('helpfulness', '有用性'),
    ('correctness', '正确性'),
    ('coherence', '连贯性'),
    ('complexity', '复杂性'),
    ('verbosity', '冗长度')
]

positions = ['user_last', 'assistant_first', 'assistant_last', 'assistant_mean']
position_names = {
    'user_last': '用户末token',
    'assistant_first': '助手首token',
    'assistant_last': '助手末token',
    'assistant_mean': '助手平均'
}

summary_file = os.path.join(output_dir, "全维度位置对比报告.txt")

with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("�� 全维度多位置探针对比报告\n")
    f.write("="*80 + "\n\n")
    f.write(f"训练维度: {len(dimensions)} 个\n")
    f.write(f"训练位置: {len(positions)} 个\n")
    f.write(f"位置列表: {', '.join([position_names[p] for p in positions])}\n")
    f.write("\n" + "="*80 + "\n\n")
    
    # 为每个维度生成报告
    for dim_key, dim_name in dimensions:
        f.write(f"\n📌 {dim_name} ({dim_key})\n")
        f.write("-"*80 + "\n")
        
        dim_dir = os.path.join(output_dir, dim_key)
        if not os.path.exists(dim_dir):
            f.write("   ❌ 未找到训练结果\n")
            continue
        
        position_stats = {}
        
        for pos in positions:
            result_file = os.path.join(dim_dir, f"{dim_key}_{pos}_results.json")
            if not os.path.exists(result_file):
                continue
            
            with open(result_file, 'r') as rf:
                results = json.load(rf)
            
            test_accs = [r['test_accuracy'] for r in results.values()]
            cv_means = [r['cv_mean'] for r in results.values()]
            
            position_stats[pos] = {
                'avg_acc': np.mean(test_accs),
                'max_acc': np.max(test_accs),
                'ge_80': sum(1 for a in test_accs if a >= 0.8),
                'ge_90': sum(1 for a in test_accs if a >= 0.9),
                'avg_cv': np.mean(cv_means)
            }
        
        # 按平均准确率排序
        sorted_positions = sorted(position_stats.items(), 
                                 key=lambda x: x[1]['avg_acc'], 
                                 reverse=True)
        
        f.write(f"\n   位置性能排名:\n")
        for rank, (pos, stats) in enumerate(sorted_positions, 1):
            pos_name = position_names[pos]
            f.write(f"   {rank}. {pos_name:12s} ")
            f.write(f"平均: {stats['avg_acc']:.4f}  ")
            f.write(f"最高: {stats['max_acc']:.4f}  ")
            f.write(f">=0.8: {stats['ge_80']:3d}  ")
            f.write(f">=0.9: {stats['ge_90']:3d}\n")
        
        # 找出最佳位置
        if sorted_positions:
            best_pos, best_stats = sorted_positions[0]
            f.write(f"\n   🏆 最佳位置: {position_names[best_pos]} ")
            f.write(f"(平均准确率: {best_stats['avg_acc']:.4f})\n")
    
    # 总体统计
    f.write("\n" + "="*80 + "\n")
    f.write("📊 跨维度位置性能对比\n")
    f.write("="*80 + "\n")
    
    # 统计每个位置在各维度的平均表现
    position_overall = {pos: [] for pos in positions}
    
    for dim_key, dim_name in dimensions:
        dim_dir = os.path.join(output_dir, dim_key)
        if not os.path.exists(dim_dir):
            continue
        
        for pos in positions:
            result_file = os.path.join(dim_dir, f"{dim_key}_{pos}_results.json")
            if not os.path.exists(result_file):
                continue
            
            with open(result_file, 'r') as rf:
                results = json.load(rf)
            
            test_accs = [r['test_accuracy'] for r in results.values()]
            position_overall[pos].append(np.mean(test_accs))
    
    f.write("\n各位置跨所有维度的平均性能:\n")
    overall_ranking = []
    for pos in positions:
        if position_overall[pos]:
            avg_performance = np.mean(position_overall[pos])
            overall_ranking.append((pos, avg_performance))
    
    overall_ranking.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (pos, avg_perf) in enumerate(overall_ranking, 1):
        pos_name = position_names[pos]
        f.write(f"   {rank}. {pos_name:12s}: {avg_perf:.4f}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("✅ 报告生成完成\n")
    f.write(f"💾 各维度详细结果保存在: {output_dir}/*/\n")
    f.write("="*80 + "\n")

print(f"✅ 总体对比报告已保存: {summary_file}")
EOFREPORT

echo ""
echo "📄 查看报告:"
cat "${OUTPUT_DIR}/全维度位置对比报告.txt"

echo ""
echo "========================================="
echo "🎉 全部完成!"
echo "========================================="
