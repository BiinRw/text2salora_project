#!/bin/bash

# 快速测试多位置探针训练 (使用少量样本)

MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
DEVICE="cuda:3"

DATA_DIR="data"
OUTPUT_DIR="results_multi_position_test"

mkdir -p $OUTPUT_DIR

echo "========================================="
echo "🧪 测试多位置探针训练"
echo "========================================="
echo "模型: $MODEL_NAME"
echo "设备: $DEVICE"
echo "样本数: 100 (测试用)"
echo "========================================="

# 只测试一个维度 (safety)
echo ""
echo "📊 测试安全性多位置探针..."
python train_multi_position_probes.py \
    --model_name $MODEL_NAME \
    --device $DEVICE \
    --good_pairs ${DATA_DIR}/safety_paired/safe_pairs_large.json \
    --bad_pairs ${DATA_DIR}/safety_paired/harmful_pairs_large.json \
    --max_samples 100 \
    --positions user_last assistant_first assistant_last assistant_mean \
    --test_split 0.2 \
    --cv_folds 3 \
    --max_iter 1000 \
    --reg_C 1.0 \
    --output_dir ${OUTPUT_DIR}/safety \
    --dimension safety

echo ""
echo "========================================="
echo "✅ 测试完成!"
echo "========================================="
echo "📂 结果目录: $OUTPUT_DIR/safety/"
echo ""
echo "📄 查看对比报告:"
cat "${OUTPUT_DIR}/safety/safety_position_comparison.txt"
