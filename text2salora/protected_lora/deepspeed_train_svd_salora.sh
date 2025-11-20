#!/bin/bash

# Safety LoRA 训练 - 使用 SVD + 正交补投影初始化
#
# 🆕 支持层区间约束:
#   bash train_safety_svd_ortho.sh                 # 默认: 约束所有层, 使用 GPU 0,1
#   bash train_safety_svd_ortho.sh 8-16           # 仅约束 8-16 层, 使用 GPU 0,1
#   bash train_safety_svd_ortho.sh 8-16,20-24     # 约束 8-16 和 20-24 层, 使用 GPU 0,1
#   bash train_safety_svd_ortho.sh all 2,3        # 约束所有层, 使用 GPU 2,3
#   bash train_safety_svd_ortho.sh 8-16 0         # 仅约束 8-16 层, 只用 GPU 0 单卡

############################
# 1) 解析命令行参数
############################

# 第一个参数: 约束层范围, 默认 all
CONSTRAINED_LAYERS="${1:-all}"

# 第二个参数: 要使用的 GPU 列表, 默认 "0,1"（两张卡）
GPU_LIST="${2:-0,1}"

############################
# 2) 配置可见 GPU 和进程数
############################

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"

# 把 "0,1" 这样的字符串拆成数组，计算 GPU 个数
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "🚀 开始训练: Safety LoRA (SVD + 正交补投影初始化)"
echo "🎯 约束层范围: ${CONSTRAINED_LAYERS}"
echo "💻 使用的物理 GPU: ${GPU_LIST}"
echo "🧮 torchrun 进程数 (nproc_per_node): ${NUM_GPUS}"

############################
# 3) 启动训练 (torchrun + DeepSpeed)
############################

torchrun --nproc_per_node=${NUM_GPUS} train_v3_deepspeed.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_type ultrafeedback \
    --dataset_size full \
    --data_format instruction \
    --output_dir ./output/correctness-lora_wo_g_r16_a32-ep1-svd_rank16-salora_all-lr_5e-5 \
    --deepspeed ./ds_config/ds_zero3_config.json \
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
    --batch_size 2 \
    --gradient_accumulation 8 \
    --learning_rate 5e-5 \
    --max_length 512 \
    --use_gradient_checkpointing \
    \
    --subspace_dir ../preference_subspace/saved_subspaces \
    --preference_dimensions correctness \
    --constrained_layers "${CONSTRAINED_LAYERS}" \
    \
    --use_swanlab true \
    --swanlab_project protected-lora \
    --experiment_name "correctness-lora_wo_g_r16_a32-ep1-svd_rank16-salora_all-lr_5e-5" \
    --print_interval 10

echo "✅ 训练完成"
echo "🎯 最终约束层范围: ${CONSTRAINED_LAYERS}"
echo "💻 使用的 GPU: ${GPU_LIST}"
