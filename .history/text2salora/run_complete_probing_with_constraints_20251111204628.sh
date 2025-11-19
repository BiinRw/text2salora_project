#!/bin/bash
#================================================================
# 一键运行完整探针测试流程(带约束)
#================================================================

echo "========================================================================"
echo "🚀 批量探针测试(带约束)完整流程"
echo "========================================================================"

# ===== 配置区域 (请根据实际情况修改) =====
LORA_DIR="./protected_lora/output/safety-lora_wo_g_r16_a32-ep1-svd_rank16-salora_16-lr_5e-5"
DIMENSION="safety"
START_STEP=100
END_STEP=500
STEP_INTERVAL=100
MAX_SAMPLES=100
DEVICE="cuda:3"
# =========================================

# 检查 LoRA 目录
if [ ! -d "$LORA_DIR" ]; then
    echo "❌ 错误: LoRA 目录不存在: $LORA_DIR"
    echo "请修改脚本中的 LORA_DIR 变量"
    exit 1
fi

# Step 1: 批量转换
echo ""
echo "📝 Step 1/2: 批量转换 LoRA → ABC.pt"
echo "------------------------------------------------------------------------"

./batch_convert_checkpoints_to_abc.sh \
    --lora_output_dir "$LORA_DIR" \
    --dimension "$DIMENSION" \
    --start_step $START_STEP \
    --end_step $END_STEP \
    --step_interval $STEP_INTERVAL

if [ $? -ne 0 ]; then
    echo "❌ 转换失败,请检查日志"
    exit 1
fi

echo "✅ 转换完成"

# Step 2: 批量测试
echo ""
echo "🧪 Step 2/2: 批量测试探针准确度"
echo "------------------------------------------------------------------------"

LORA_NAME=$(basename "$LORA_DIR")
ABC_DIR="./abc_checkpoints/$LORA_NAME"

if [ ! -d "$ABC_DIR" ]; then
    echo "❌ 错误: ABC 目录不存在: $ABC_DIR"
    exit 1
fi

cd probing/probing_test

./batch_test_with_constraints.sh \
    --abc_dir "../../$ABC_DIR" \
    --dimension "$DIMENSION" \
    --max_samples $MAX_SAMPLES \
    --device "$DEVICE" \
    --start_step $START_STEP \
    --end_step $END_STEP \
    --step_interval $STEP_INTERVAL

if [ $? -ne 0 ]; then
    echo "❌ 测试失败,请检查日志"
    exit 1
fi

echo "✅ 测试完成"

cd ../..

# 显示结果位置
echo ""
echo "========================================================================"
echo "✅ 完整流程执行成功!"
echo "========================================================================"
echo "📂 ABC.pt 文件: $ABC_DIR"
echo "📊 测试结果: ./probing/probing_test/results/batch_test_with_constraints/$LORA_NAME"
echo ""
echo "查看结果:"
echo "  转换汇总: cat $ABC_DIR/conversion_summary.txt"
echo "  测试汇总: cat ./probing/probing_test/results/batch_test_with_constraints/$LORA_NAME/summary_report_constrained.txt"
echo "========================================================================"
