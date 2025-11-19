#!/bin/bash

# 训练所有维度的线性探针
# 包括: safety + 5个HelpSteer维度

MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
DEVICE="cuda:2"

RESULTS_DIR="results_all_dimensions"
PROBES_DIR="trained_probes"

mkdir -p $RESULTS_DIR
mkdir -p $PROBES_DIR

echo "========================================================================"
echo "🚀 训练所有维度的线性探针"
echo "========================================================================"
echo "模型: $MODEL_NAME"
echo "设备: $DEVICE"
echo "结果目录: $RESULTS_DIR"
echo "探针目录: $PROBES_DIR"
echo "========================================================================"

# 1. 训练安全性探针
echo ""
echo "📊 [1/6] 训练安全性探针 (Safety)..."
python train_probe_paired_improved.py \
    --model_name $MODEL_NAME \
    --device $DEVICE \
    --safe_pairs data/safety_paired/safe_pairs_large.json \
    --harmful_pairs data/safety_paired/harmful_pairs_large.json \
    --max_samples 5000 \
    --test_split 0.2 \
    --cv_folds 5 \
    --max_iter 2000 \
    --reg_C 1.0 \
    --output_file ${RESULTS_DIR}/probe_safety.json \
    --probe_dir ${PROBES_DIR}/safety

# 2. 训练有用性探针
echo ""
echo "📊 [2/6] 训练有用性探针 (Helpfulness)..."
python train_probe_paired_improved.py \
    --model_name $MODEL_NAME \
    --device $DEVICE \
    --safe_pairs data/helpsteer_paired/helpfulness_good_pairs.json \
    --harmful_pairs data/helpsteer_paired/helpfulness_bad_pairs.json \
    --max_samples 5000 \
    --test_split 0.2 \
    --cv_folds 5 \
    --max_iter 2000 \
    --reg_C 1.0 \
    --output_file ${RESULTS_DIR}/probe_helpfulness.json \
    --probe_dir ${PROBES_DIR}/helpfulness

# 3. 训练正确性探针
echo ""
echo "📊 [3/6] 训练正确性探针 (Correctness)..."
python train_probe_paired_improved.py \
    --model_name $MODEL_NAME \
    --device $DEVICE \
    --safe_pairs data/helpsteer_paired/correctness_good_pairs.json \
    --harmful_pairs data/helpsteer_paired/correctness_bad_pairs.json \
    --max_samples 5000 \
    --test_split 0.2 \
    --cv_folds 5 \
    --max_iter 2000 \
    --reg_C 1.0 \
    --output_file ${RESULTS_DIR}/probe_correctness.json \
    --probe_dir ${PROBES_DIR}/correctness

# 4. 训练连贯性探针
echo ""
echo "📊 [4/6] 训练连贯性探针 (Coherence)..."
python train_probe_paired_improved.py \
    --model_name $MODEL_NAME \
    --device $DEVICE \
    --safe_pairs data/helpsteer_paired/coherence_good_pairs.json \
    --harmful_pairs data/helpsteer_paired/coherence_bad_pairs.json \
    --max_samples 5000 \
    --test_split 0.2 \
    --cv_folds 5 \
    --max_iter 2000 \
    --reg_C 1.0 \
    --output_file ${RESULTS_DIR}/probe_coherence.json \
    --probe_dir ${PROBES_DIR}/coherence

# 5. 训练复杂性探针
echo ""
echo "📊 [5/6] 训练复杂性探针 (Complexity)..."
python train_probe_paired_improved.py \
    --model_name $MODEL_NAME \
    --device $DEVICE \
    --safe_pairs data/helpsteer_paired/complexity_good_pairs.json \
    --harmful_pairs data/helpsteer_paired/complexity_bad_pairs.json \
    --max_samples 5000 \
    --test_split 0.2 \
    --cv_folds 5 \
    --max_iter 2000 \
    --reg_C 1.0 \
    --output_file ${RESULTS_DIR}/probe_complexity.json \
    --probe_dir ${PROBES_DIR}/complexity

# 6. 训练冗长度探针
echo ""
echo "📊 [6/6] 训练冗长度探针 (Verbosity)..."
python train_probe_paired_improved.py \
    --model_name $MODEL_NAME \
    --device $DEVICE \
    --safe_pairs data/helpsteer_paired/verbosity_good_pairs.json \
    --harmful_pairs data/helpsteer_paired/verbosity_bad_pairs.json \
    --max_samples 5000 \
    --test_split 0.2 \
    --cv_folds 5 \
    --max_iter 2000 \
    --reg_C 1.0 \
    --output_file ${RESULTS_DIR}/probe_verbosity.json \
    --probe_dir ${PROBES_DIR}/verbosity

echo ""
echo "========================================================================"
echo "✅ 所有维度训练完成!"
echo "========================================================================"
echo "📂 结果文件: $RESULTS_DIR/"
echo "📂 探针模型: $PROBES_DIR/"
echo "========================================================================"

# 生成总结报告
echo ""
echo "📝 生成总结报告..."
python << 'EOFREPORT'
import json
import os

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
    f.write("📊 所有维度探针训练总结报告\n")
    f.write("=" * 80 + "\n\n")
    
    for dim_key, dim_name in dimensions:
        result_file = os.path.join(results_dir, f"probe_{dim_key}.json")
        
        if os.path.exists(result_file):
            with open(result_file, 'r') as rf:
                results = json.load(rf)
            
            test_accs = [r['test_accuracy'] for r in results.values()]
            cv_means = [r['cv_mean'] for r in results.values()]
            
            f.write(f"{'='*80}\n")
            f.write(f"📋 {dim_name} ({dim_key})\n")
            f.write(f"{'='*80}\n")
            f.write(f"总注意力头数: {len(results)}\n")
            f.write(f"平均测试准确率: {sum(test_accs)/len(test_accs):.4f}\n")
            f.write(f"最高测试准确率: {max(test_accs):.4f}\n")
            f.write(f"平均CV准确率: {sum(cv_means)/len(cv_means):.4f}\n")
            f.write(f"最高CV准确率: {max(cv_means):.4f}\n")
            f.write(f"准确率 >= 0.8: {sum(1 for a in test_accs if a >= 0.8)}\n")
            f.write(f"准确率 >= 0.9: {sum(1 for a in test_accs if a >= 0.9)}\n")
            
            # Top 5
            top_5 = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)[:5]
            f.write(f"\n🏆 Top 5 最佳注意力头:\n")
            for i, (head, metrics) in enumerate(top_5, 1):
                f.write(f"  {i}. {head}: test={metrics['test_accuracy']:.4f}, cv={metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}\n")
            
            f.write("\n")
    
    f.write("=" * 80 + "\n")
    f.write("✅ 所有维度探针训练完成!\n")
    f.write(f"💾 探针模型已保存到: trained_probes/\n")
    f.write("=" * 80 + "\n")

print(f"✅ 报告已保存到: {output_file}")
EOFREPORT

echo ""
echo "📖 查看报告:"
cat ${RESULTS_DIR}/training_summary.txt

