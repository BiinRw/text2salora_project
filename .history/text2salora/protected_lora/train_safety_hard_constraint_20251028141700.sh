#!/bin/bash
# 🔒 使用硬约束（SaLoRA 风格）训练 Safety LoRA

python train_v2_main.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --gpu_id 3 \
    \
    --dataset_type ultrafeedback \
    --dataset_size 1w \
    --data_format instruction \
    --max_samples 10000 \
    \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --target_modules q_proj,v_proj, v_proj, o_proj, gate_proj, up_proj, down_proj \
    \
    --num_epochs 1 \
    --batch_size 2 \
    --gradient_accumulation 4 \
    --learning_rate 1e-4 \
    --max_length 512 \
    \
    --use_hard_constraint \
    --subspace_dir ../preference_subspace/saved_subspaces \
    --preference_dimensions safety \
    --subspace_rank 16 \
    \
    --output_dir ./output/safety_hard_constraint_1w_rank16-lora_full \
    --experiment_name safety_hard_constraint_1w_rank16-lora_full \
    --use_swanlab True\
    --swanlab_project protected-lora

echo "✅ 硬约束训练完成"
