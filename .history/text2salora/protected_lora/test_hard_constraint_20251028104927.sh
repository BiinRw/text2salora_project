#!/bin/bash
# 快速测试硬约束功能

python train_v2_main.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --gpu_id 3 \
    \
    --dataset_type ultrafeedback \
    --dataset_size 1w \
    --data_format chosen_only \
    --max_samples 100 \
    \
    --lora_rank 8 \
    --lora_alpha 16 \
    \
    --num_epochs 1 \
    --batch_size 2 \
    --gradient_accumulation 2 \
    --learning_rate 1e-4 \
    --max_length 256 \
    \
    --use_hard_constraint \
    --subspace_dir ../preference_subspace/saved_subspaces \
    --preference_dimensions safety \
    --subspace_rank 16 \
    \
    --output_dir ./output/test_hard_constraint \
    --experiment_name test_hard_constraint \
    --use_swanlab

echo "✅ 测试完成"
