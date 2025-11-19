#!/bin/bash

# 快速测试脚本: 只训练一个维度验证修正是否有效

MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
DEVICE="cuda:3"

echo "========================================="
echo "🧪 快速测试: 训练helpfulness维度"
echo "========================================="
echo "目的: 验证0/1标签数据是否正常工作"
echo "========================================="

python train_helpsteer_dimension.py \
    --model_name $MODEL_NAME \
    --device $DEVICE \
    --good_pairs data/helpsteer_paired/helpfulness_good_pairs.json \
    --bad_pairs data/helpsteer_paired/helpfulness_bad_pairs.json \
    --max_samples 1000 \
    --test_split 0.2 \
    --cv_folds 3 \
    --max_iter 1000 \
    --reg_C 1.0 \
    --output_file results_test/probe_helpfulness_test.json \
    --probe_dir trained_probes_test/helpfulness

echo ""
echo "========================================="
echo "✅ 测试完成! 检查准确率是否正常"
echo "========================================="

