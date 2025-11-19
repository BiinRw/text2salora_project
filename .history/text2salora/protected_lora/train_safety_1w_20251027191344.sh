#!/bin/bash
# Safety 维度正交约束训练 - 1w 数据集
# 这是第一个完整的正交约束实验

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate text-to-salora

# 设置实验名称
EXPERIMENT_NAME="safety-orth-1w-$(date +%m%d-%H%M)-lambda1000"

echo "========================================"
echo "Safety 正交约束训练"
echo "========================================"
echo "Conda 环境: text-to-salora"
echo "实验名称: $EXPERIMENT_NAME"
echo "数据集: ultrafeedback (1w)"
echo "正交约束: safety 维度"
echo "子空间路径: ../preference_subspace/saved_subspaces"
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
    --use_orthogonal \
    --subspace_dir ../preference_subspace/saved_subspaces \
    --preference_dimensions safety \
    --lambda_orth 1000 \
    \
    --use_swanlab true \
    --swanlab_project "protected-lora" \
    --print_interval 50 \
    \
    --output_dir "./output/safety_orth_1w_lambda10000"

echo ""
echo "========================================"
echo "训练完成！"
echo "========================================"
echo "输出目录: ./output/safety_orth_1w"
echo "包含文件:"
echo "  - adapter_model.safetensors (LoRA 权重)"
echo "  - orth_loss_history.json (正交损失历史)"
echo ""
