#!/bin/bash
# 🔒 使用硬约束（SaLoRA 风格）训练 Safety LoRA

python train_v2_main.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --gpu_id 3 \
    \
    --dataset_type ultrafeedback \
    --dataset_size full \
    --data_format instruction \
    --max_samples 60000 \
    \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --target_modules q_proj v_proj o_proj up_proj down_proj \
    \
    --num_epochs 1 \
    --batch_size 1 \
    --gradient_accumulation 8 \
    --learning_rate 2e-4 \
    --max_length 512 \
    \
    --use_hard_constraint \
    --subspace_dir ../preference_subspace/saved_subspaces \
    --preference_dimensions safety \
    --subspace_rank 16 \
    \
    --output_dir ./output/safety_hard_constraint_full_rank16-lora_wo_gateproj_ep1 \
    --experiment_name safety_hard_constraint_ful_rank16-lora_wo_gateproj_ep1 \
    --use_swanlab True\
    --swanlab_project protected-lora

echo "✅ 硬约束训练完成"
