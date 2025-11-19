#!/bin/bash

# Safety LoRA 训练 - 使用 SVD + 正交补投影初始化
# 这是理论最优的初始化方法
# 
# 🆕 支持层区间约束:
#   --constrained_layers all            # 所有层 (默认)
#   --constrained_layers 8-16           # 仅约束 8-16 层
#   --constrained_layers 8-16,20-24     # 约束 8-16 和 20-24 层
#   --constrained_layers 0-8            # 仅约束前 9 层
#
# 用法:
#   bash train_safety_svd_ortho.sh              # 使用默认配置 (all)
#   bash train_safety_svd_ortho.sh 8-16         # 仅约束 8-16 层
#   bash train_safety_svd_ortho.sh 8-16,20-24   # 约束多个区间

# 从命令行参数获取层区间配置，默认为 'all'
CONSTRAINED_LAYERS="${1:-all}"

echo "🚀 开始训练: Safety LoRA (SVD + 正交补投影初始化)"
echo "🎯 约束层范围: $CONSTRAINED_LAYERS"

python train_v2_main.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_type ultrafeedback \
    --dataset_size full \
    --data_format instruction \
    --output_dir ./output/helpfulness-lora_wo_g_r16_a32-ep1-svd_rank16-salora_24_27-lr_5e-5 \
    --gpu_id 0 \
    \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --target_modules q_proj k_proj v_proj o_proj up_proj down_proj \
    --lora_init_method svd_salora \
    --use_hard_constraint \
    --svd_niter 30 \
    \
    --num_epochs 1 \
    --batch_size 1 \
    --gradient_accumulation 8 \
    --learning_rate 5e-5 \
    --max_length 512 \
    --use_gradient_checkpointing \
    \
    --subspace_dir ../preference_subspace/saved_subspaces \
    --preference_dimensions safety \
    --constrained_layers "$CONSTRAINED_LAYERS" \
    \
    --use_swanlab true \
    --swanlab_project protected-lora \
    --experiment_name "helpfulness-lora_wo_g_r16_a32-ep1-svd_rank16-salora_24_27-lr_5e-5" \
    --print_interval 10

echo "✅ 训练完成"
echo "🎯 约束层范围: $CONSTRAINED_LAYERS"
