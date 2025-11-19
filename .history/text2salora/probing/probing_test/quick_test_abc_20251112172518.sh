#!/bin/bash
#================================================================
# 快速测试单个 checkpoint 的完整 ABC 模块
#================================================================

set -e

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate text-to-salora

echo "✓ 已激活 conda 环境: text-to-salora"

# 默认参数
CHECKPOINT=${1:-"checkpoint-3000"}
DIMENSION=${2:-"safety"}
POSITION=${3:-"assistanct_last"}
DEVICE=${4:-"cuda:0"}

LORA_BASE_DIR="protected_lora/output/safety-lora_wo_g_r16_a32-ep1-svd_rank16-salora_16-lr_5e-5"
LORA_PATH="$LORA_BASE_DIR/$CHECKPOINT"

echo ""
echo "========================================================================"
echo "🚀 快速测试完整 ABC 模块"
echo "========================================================================"
echo "Checkpoint: $CHECKPOINT"
echo "维度: $DIMENSION"
echo "位置: $POSITION"
echo "设备: $DEVICE"
echo "========================================================================"

# 检查 checkpoint 是否存在
if [ ! -d "$LORA_PATH" ]; then
    echo "❌ 错误: Checkpoint 不存在: $LORA_PATH"
    echo ""
    echo "可用的 checkpoints:"
    ls -1 "$LORA_BASE_DIR" | grep "checkpoint-" | head -10
    exit 1
fi

# 确定数据路径
if [ "$DIMENSION" == "safety" ]; then
    DATA_PATH="probing/data/safety_paired"
else
    DATA_PATH="probing/data/helpsteer_merged_paired"
fi

# 输出文件
OUTPUT_FILE="probing/probing_test/quick_test_result.json"

# 运行测试
python probing/probing_test/test_probe_with_full_abc.py \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --lora_path "$LORA_PATH" \
    --subspace_dir preference_subspace/saved_subspaces \
    --probe_dir probing/trained_probes/multi_position-1103 \
    --dimension "$DIMENSION" \
    --position "$POSITION" \
    --data_path "$DATA_PATH" \
    --max_samples 100 \
    --output_file "$OUTPUT_FILE" \
    --device "$DEVICE"

echo ""
echo "========================================================================"
echo "✅ 测试完成!"
echo "========================================================================"
echo "结果文件: $OUTPUT_FILE"
echo ""
echo "查看结果:"
echo "  cat $OUTPUT_FILE | python -m json.tool"
echo ""
