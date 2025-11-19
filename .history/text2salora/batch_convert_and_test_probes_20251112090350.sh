#!/bin/bash
# 批量转换为 SaLoRA 格式并测试探针准确度

set -e

# 配置
LORA_PATH="safety-lora_wo_g_r16_a32-ep1-svd_rank16-salora_16-lr_5e-5"
LORA_DIR="./protected_lora/output/$LORA_PATH"
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
SUBSPACE_DIR="./preference_subspace/saved_subspaces"
DIMENSION="safety"
PROBE_PATH="./probing/trained_probes/safety/linear_probes.pkl"
TEST_DATA="./probing/data/safety_paired"
ABC_OUTPUT_DIR="./abc_checkpoints_salora"
RESULT_DIR="./probing/probing_test/results_with_constraints_ABC/$DIMENSION/$LORA_PATH"
DEVICE="cuda:3"
MAX_SAMPLES=500  # 使用全部测试数据

# 步数筛选
START_STEP=100
END_STEP=7600
STEP_INTERVAL=500

echo "========================================================================"
echo "🚀 批量转换 + 探针测试流程"
echo "========================================================================"
echo "LoRA 目录: $LORA_DIR"
echo "维度: $DIMENSION"
echo "设备: $DEVICE"
echo "步数范围: $START_STEP - $END_STEP (间隔 $STEP_INTERVAL)"
echo "========================================================================"

# 查找所有 checkpoints
CHECKPOINTS=$(find "$LORA_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V)

# 筛选步数
FILTERED_CHECKPOINTS=()
for CKPT in $CHECKPOINTS; do
    STEP=$(basename "$CKPT" | grep -oP '\d+')
    
    if [ "$STEP" -ge "$START_STEP" ] && [ "$STEP" -le "$END_STEP" ]; then
        if [ $(( ($STEP - $START_STEP) % $STEP_INTERVAL )) -eq 0 ]; then
            FILTERED_CHECKPOINTS+=("$CKPT")
        fi
    fi
done

TOTAL=${#FILTERED_CHECKPOINTS[@]}
echo "📝 找到 $TOTAL 个 checkpoints"

# 处理每个 checkpoint
CURRENT=0
for CKPT in "${FILTERED_CHECKPOINTS[@]}"; do
    CURRENT=$((CURRENT + 1))
    CKPT_NAME=$(basename "$CKPT")
    STEP=$(echo "$CKPT_NAME" | grep -oP '\d+')
    
    LORA_NAME=$(basename "$LORA_DIR")
    ABC_PATH="$ABC_OUTPUT_DIR/$LORA_NAME/${CKPT_NAME}_ABC.pt"
    
    echo ""
    echo "========================================================================"
    echo "[$CURRENT/$TOTAL] 处理 $CKPT_NAME"
    echo "========================================================================"
    
    # Step 1: 转换为 SaLoRA 格式
    if [ ! -f "$ABC_PATH" ]; then
        echo "🔄 转换为 SaLoRA 格式..."
        python convert_to_salora_format.py \
            --base_model "$BASE_MODEL" \
            --lora_path "$CKPT" \
            --subspace_dir "$SUBSPACE_DIR" \
            --dimension "$DIMENSION" \
            --output_path "$ABC_PATH" \
            --num_layers 28 \
            --device "$DEVICE"
        
        if [ $? -ne 0 ]; then
            echo "❌ 转换失败"
            continue
        fi
    else
        echo "⏭️  ABC.pt 已存在,跳过转换"
    fi
    
    # Step 2: 测试探针准确度
    echo ""
    echo "🧪 测试探针准确度..."
    cd probing/probing_test
    python test_probe_with_abc_simple.py \
        --model_path "$BASE_MODEL" \
        --abc_path "../../$ABC_PATH" \
        --probe_path "../trained_probes/$DIMENSION/linear_probes.pkl" \
        --test_data "../data/safety_paired" \
        --dimension "$DIMENSION" \
        --max_samples $MAX_SAMPLES \
        --output_dir "results/with_constraints" \
        --device "$DEVICE"
    
    if [ $? -eq 0 ]; then
        echo "✅ $CKPT_NAME 测试完成"
    else
        echo "❌ $CKPT_NAME 测试失败"
    fi
    
    cd ../..
done

echo ""
echo "========================================================================"
echo "✅ 批量处理完成!"
echo "========================================================================"
echo "ABC.pt 文件: $ABC_OUTPUT_DIR"
echo "测试结果: $RESULT_DIR"
echo "========================================================================"
