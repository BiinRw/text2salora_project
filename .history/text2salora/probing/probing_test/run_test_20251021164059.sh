#!/bin/bash

# 探针测试运行脚本
# 支持测试基模型、微调模型和LoRA模型

set -e

# 默认参数
MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
DIMENSION="helpfulness"
DEVICE="cuda:0"
MAX_SAMPLES=""

# 显示使用说明
show_help() {
    echo "用法: bash run_test.sh [选项]"
    echo ""
    echo "选项:"
    echo "  --model_path PATH      基模型路径 (默认: meta-llama/Meta-Llama-3.1-8B-Instruct)"
    echo "  --lora_path PATH       LoRA适配器路径 (可选)"
    echo "  --dimension DIM        测试维度 (默认: helpfulness)"
    echo "                         可选: helpfulness, correctness, coherence, verbosity, safety"
    echo "  --device DEVICE        设备 (默认: cuda:0)"
    echo "  --max_samples N        最大测试样本数 (可选,用于快速测试)"
    echo "  --help                 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  # 测试基模型 (使用配对数据)"
    echo "  bash run_test.sh --dimension helpfulness"
    echo ""
    echo "  # 测试基模型 (使用极端数据)"
    echo "  bash run_test.sh --dimension helpfulness --data_type ultra"
    echo ""
    echo "  # 测试LoRA模型"
    echo "  bash run_test.sh --dimension helpfulness --lora_path /path/to/lora"
    echo ""
    echo "  # 快速测试(仅100样本)"
    echo "  bash run_test.sh --dimension helpfulness --max_samples 100"
    exit 0
}

# 解析命令行参数
LORA_PATH=""
DATA_TYPE="paired"  # 默认使用配对数据

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --lora_path)
            LORA_PATH="$2"
            shift 2
            ;;
        --dimension)
            DIMENSION="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="--max_samples $2"
            shift 2
            ;;
        --data_type)
            DATA_TYPE="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            echo "未知选项: $1"
            show_help
            ;;
    esac
done

# 设置数据路径和探针路径
if [ "$DATA_TYPE" == "paired" ]; then
    TEST_DATA="../data/helpsteer_merged_paired"
    PROBE_DIR="../trained_probes_paired/${DIMENSION}"
    # Safety维度使用特殊路径
    if [ "$DIMENSION" == "safety" ]; then
        TEST_DATA="../data/safety_paired"
        PROBE_DIR="../trained_probes_large/${DIMENSION}"
    fi
    # Safety维度使用特殊路径
    if [ "$DIMENSION" == "safety" ]; then
        TEST_DATA="../data/safety_paired"
        PROBE_DIR="../trained_probes_large/${DIMENSION}"
    fi
    OUTPUT_DIR="results/paired/${DIMENSION}"
else
    TEST_DATA="../data/helpsteer_ultra_extreme"
    PROBE_DIR="../trained_probes_extreme/${DIMENSION}"
    # Safety维度使用特殊路径
    if [ "$DIMENSION" == "safety" ]; then
        TEST_DATA="../data/safety_paired"
        PROBE_DIR="../trained_probes_large/${DIMENSION}"
    fi
    OUTPUT_DIR="results/ultra/${DIMENSION}"
fi

# 如果使用LoRA,更新输出目录
if [ -n "$LORA_PATH" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_lora"
    LORA_ARG="--lora_path $LORA_PATH"
else
    LORA_ARG=""
fi

# 检查探针目录是否存在
if [ ! -d "$PROBE_DIR" ]; then
    echo "❌ 错误: 探针目录不存在: $PROBE_DIR"
    echo "请先训练探针:"
    echo "  cd .. && bash train_extreme_single.sh $DIMENSION --paired"
    exit 1
fi

# 检查测试数据是否存在
if [ ! -d "$TEST_DATA" ]; then
    echo "❌ 错误: 测试数据目录不存在: $TEST_DATA"
    exit 1
fi

# 打印配置信息
echo "========================================"
echo "🧪 探针准确度测试"
echo "========================================"
echo "模型路径: $MODEL_PATH"
[ -n "$LORA_PATH" ] && echo "LoRA路径: $LORA_PATH"
echo "测试维度: $DIMENSION"
echo "数据类型: $DATA_TYPE"
echo "探针目录: $PROBE_DIR"
echo "测试数据: $TEST_DATA"
echo "输出目录: $OUTPUT_DIR"
echo "设备: $DEVICE"
[ -n "$MAX_SAMPLES" ] && echo "样本限制: $MAX_SAMPLES"
echo "========================================"
echo ""

# 运行测试
python test_probe_accuracy.py \
    --model_path "$MODEL_PATH" \
    $LORA_ARG \
    --probe_dir "$PROBE_DIR" \
    --test_data "$TEST_DATA" \
    --dimension "$DIMENSION" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    $MAX_SAMPLES

echo ""
echo "✅ 测试完成!"
echo "结果保存在: $OUTPUT_DIR"
