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
echo "LoRA 初始化: random (标准初始化)"
echo "训练轮数: 3 epochs"
echo "========================================"
echo ""

python train_v2_main.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --experiment_name "$EXPERIMENT_NAME" \
    --gpu_id 3 \
    \
    --dataset_type ultrafeedback \
    --dataset_size full \
    --data_format instruction \
    \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_dropout 0.01 \
    --target_modules q_proj k_proj v_proj o_proj up_proj down_proj \
    --lora_init_method random \
    \
    --num_epochs 1 \
    --batch_size 2 \
    --gradient_accumulation 4 \
    --learning_rate 1e-4 \
    --max_length 512 \
    --use_gradient_checkpointing \
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
