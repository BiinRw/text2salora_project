#!/bin/bash

# Safety LoRA 训练 - 使用 SVD + 正交补投影初始化
# 这是理论最优的初始化方法

echo "🚀 开始训练: Safety LoRA (SVD + 正交补投影初始化)"

python train_v2_main.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_type ultrafeedback \
    --dataset_size full \
    --data_format instruction \
    --output_dir ./output/safety-lora_wo_g_u_d-ep1-svd_rank16_orth-lr_1e-4 \
    --gpu_id 3 \
    \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_dropout 0.01 \
    --target_modules q_proj k_proj v_proj o_proj \
    --lora_init_method svd_ortho \
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
    \
    --use_swanlab true \
    --swanlab_project protected-lora \
    --experiment_name "safety-lora_wo_g-ep1-svd_rank16_ortho-lr_1e-4" \
    --print_interval 10

echo "✅ 训练完成"
