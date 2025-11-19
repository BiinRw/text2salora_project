#!/bin/bash
# 基线训练（无正交约束）- 1w 数据集
# 用于对比正交约束的效果

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate text-to-salora

# 设置实验名称
EXPERIMENT_NAME="baseline-1w-$(date +%m%d-%H%M)"

echo "========================================"
echo "基线训练（无正交约束）"
echo "========================================"
echo "Conda 环境: text-to-salora"
echo "实验名称: $EXPERIMENT_NAME"
echo "数据集: ultrafeedback (1w)"
echo "正交约束: ❌ 关闭"
echo "训练轮数: 3 epochs"
echo "========================================"
echo ""

python train_v2_main.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --gpu_id 3 \
    \
    --dataset_type ultrafeedback \
    --dataset_size 1w \
    --data_format instruction \
    \
    --lora_rank 8 \
    --lora_alpha 16 \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation 4 \
    --learning_rate 1e-4 \
    --max_length 512 \
    \
    --use_swanlab true \
    --swanlab_project "protected-lora" \
    --print_interval 50 \
    \
    --output_dir "./output/baseline_1w"

echo ""
echo "========================================"
echo "训练完成！"
echo "========================================"
