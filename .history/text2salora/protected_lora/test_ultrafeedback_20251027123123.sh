#!/bin/bash
# UltraFeedback 数据集测试脚本
# 用于快速调试训练流程

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate text-to-salora

# 设置实验名称
EXPERIMENT_NAME="ultrafeedback-debug-$(date +%m%d-%H%M)"

echo "========================================"
echo "UltraFeedback 数据集训练测试"
echo "========================================"
echo "Conda 环境: text-to-salora"
echo "实验名称: $EXPERIMENT_NAME"
echo "数据集: ultrafeedback"
echo "数据大小: 100 条"
echo "限制样本: 20 条（快速测试）"
echo "训练轮数: 1 epoch"
echo "========================================"
echo ""

python train_v2_main.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --gpu_id 3 \
    \
    --dataset_type ultrafeedback \
    --dataset_size 1w \
    --data_format instruction \
    --max_samples 20 \
    \
    --lora_rank 8 \
    --lora_alpha 16 \
    --num_epochs 1 \
    --batch_size 2 \
    --gradient_accumulation 2 \
    --learning_rate 1e-4 \
    --max_length 512 \
    \
    --use_swanlab true \
    --swanlab_project "protected-lora-debug" \
    --print_interval 5 \
    \
    --output_dir "./output/test_ultrafeedback"

echo ""
echo "========================================"
echo "测试完成！"
echo "========================================"
