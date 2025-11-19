#!/bin/bash

#================================================================
# 批量测试多个LoRA checkpoints的多位置探针准确度
#================================================================
# 功能: 
# 1. 扫描output_dir下的所有checkpoints
# 2. 对每个checkpoint测试多个位置的探针准确度  
# 3. 生成汇总对比报告
#================================================================

# 🎨 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# 默认参数
MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR=""
DIMENSION="safety"
PROBE_DIR=""  # 将根据维度自动设置
TEST_DATA=""  # 将根据维度自动设置
MAX_SAMPLES=""
DEVICE="cuda:0"
RESULT_BASE_DIR="results/batch_multi_position_test"
POSITIONS="assistant_last assistant_first assistant_mean"
START_STEP=""
END_STEP=""
STEP_INTERVAL=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dimension)
            DIMENSION="$2"
            shift 2
            ;;
        --probe_dir)
            PROBE_DIR="$2"
            shift 2
            ;;
        --test_data)
            TEST_DATA="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --result_dir)
            RESULT_BASE_DIR="$2"
            shift 2
            ;;
        --positions)
            POSITIONS="$2"
            shift 2
            ;;
        --start_step)
            START_STEP="$2"
            shift 2
            ;;
        --end_step)
            END_STEP="$2"
            shift 2
            ;;
        --step_interval)
            STEP_INTERVAL="$2"
            shift 2
            ;;
        --help)
            echo "用法: $0 --output_dir PATH [选项]"
            echo ""
            echo "必需参数:"
            echo "  --output_dir PATH       LoRA训练输出目录 (包含多个checkpoints)"
            echo ""
            echo "可选参数:"
            echo "  --model_path PATH       基础模型路径 (默认: Qwen/Qwen2.5-1.5B-Instruct)"
            echo "  --dimension NAME        测试维度 (默认: safety)"
            echo "  --probe_dir PATH        探针目录 (默认: ../results_multi_position/safety)"
            echo "  --test_data PATH        测试数据目录 (默认: ../data/safety_paired)"
            echo "  --max_samples N         最大测试样本数 (可选)"
            echo "  --device DEVICE         GPU设备 (默认: cuda:0)"
            echo "  --result_dir PATH       结果输出目录 (默认: results/batch_multi_position_test)"
            echo "  --positions 'pos1 pos2' 测试位置列表 (默认: assistant_last assistant_first assistant_mean)"
            echo "  --start_step N          起始step (可选)"
            echo "  --end_step N            结束step (可选)"
            echo "  --step_interval N       step间隔 (可选)"
            echo "  --help                  显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0 --output_dir ../protected_lora/output/safety-lora \\"
            echo "     --dimension safety \\"
            echo "     --max_samples 100 \\"
            echo "     --start_step 100 \\"
            echo "     --end_step 1000 \\"
            echo "     --step_interval 100"
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知参数 $1${NC}"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$OUTPUT_DIR" ]; then
    echo -e "${RED}错误: 必须指定 --output_dir${NC}"
    echo "使用 --help 查看帮助"
    exit 1
fi

# 根据维度自动设置路径（如果未指定）
if [ -z "$PROBE_DIR" ]; then
    # 使用最新的多位置探针目录
    PROBE_DIR="../trained_probes/multi_position-1103/${DIMENSION}"
fi

if [ -z "$TEST_DATA" ]; then
    # 根据维度设置测试数据路径
    case "$DIMENSION" in
        safety)
            TEST_DATA="../data/safety_paired"
            ;;
        helpfulness)
            TEST_DATA="../data/helpsteer_merged_paired"
            ;;
        correctness)
            TEST_DATA="../data/helpsteer_merged_paired"
            ;;
        coherence)
            TEST_DATA="../data/helpsteer_merged_paired"
            ;;
        complexity)
            TEST_DATA="../data/helpsteer_merged_paired"
            ;;
        verbosity)
            TEST_DATA="../data/helpsteer_merged_paired"
            ;;
        *)
            echo -e "${RED}错误: 未知的维度 $DIMENSION${NC}"
            echo "支持的维度: safety, helpfulness, correctness, coherence, complexity, verbosity"
            exit 1
            ;;
    esac
fi

# 根据维度自动设置路径（如果未指定）
if [ -z "$PROBE_DIR" ]; then
    # 使用最新的多位置探针目录
    PROBE_DIR="../trained_probes/multi_position-1103/${DIMENSION}"
fi

if [ -z "$TEST_DATA" ]; then
    # 根据维度设置测试数据路径
    case "$DIMENSION" in
        safety)
            TEST_DATA="../data/safety_paired"
            ;;
        helpfulness)
            TEST_DATA="../data/helpsteer_merged_paired"
            ;;
        correctness)
            TEST_DATA="../data/helpsteer_merged_paired"
            ;;
        coherence)
            TEST_DATA="../data/helpsteer_merged_paired"
            ;;
        complexity)
            TEST_DATA="../data/helpsteer_merged_paired"
            ;;
        verbosity)
            TEST_DATA="../data/helpsteer_merged_paired"
            ;;
        *)
            echo -e "${RED}错误: 未知的维度 $DIMENSION${NC}"
            echo "支持的维度: safety, helpfulness, correctness, coherence, complexity, verbosity"
            exit 1
            ;;
    esac
fi

# 检查目录
if [ ! -d "$OUTPUT_DIR" ]; then
    echo -e "${RED}❌ 错误: 输出目录不存在: $OUTPUT_DIR${NC}"
    exit 1
fi

if [ ! -d "$PROBE_DIR" ]; then
    echo -e "${RED}❌ 错误: 探针目录不存在: $PROBE_DIR${NC}"
    exit 1
fi

if [ ! -d "$TEST_DATA" ]; then
    echo -e "${RED}❌ 错误: 测试数据目录不存在: $TEST_DATA${NC}"
    exit 1
fi

# 创建结果目录
# 提取模型名称（基础模型最后一部分）
MODEL_NAME=$(basename "$MODEL_PATH" | tr '/' '-')
# 提取LoRA名称（output_dir最后一部分）
LORA_NAME=$(basename "$OUTPUT_DIR")
# 组合目录名: 模型名+LoRA名
COMBINED_NAME="${MODEL_NAME}+${LORA_NAME}"

# 创建符合分析脚本要求的目录结构
# batch_multi_position_test/{combined_name}/{dimension}_lora/
RUN_DIR="$RESULT_BASE_DIR/$COMBINED_NAME"
mkdir -p "$RUN_DIR/${DIMENSION}_lora"
LOG_FILE="$RUN_DIR/batch_test.log"

# 打印配置
echo "" | tee "$LOG_FILE"
echo -e "${CYAN}================================================================================${NC}" | tee -a "$LOG_FILE"
echo -e "${CYAN}🧪 批量测试多位置探针准确度${NC}" | tee -a "$LOG_FILE"
echo -e "${CYAN}================================================================================${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}📊 基础模型:${NC} $MODEL_PATH" | tee -a "$LOG_FILE"
echo -e "${BLUE}📂 LoRA目录:${NC} $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo -e "${BLUE}📂 维度:${NC} $DIMENSION" | tee -a "$LOG_FILE"
echo -e "${BLUE}📍 测试位置:${NC} $POSITIONS" | tee -a "$LOG_FILE"
echo -e "${BLUE}📁 探针目录:${NC} $PROBE_DIR" | tee -a "$LOG_FILE"
echo -e "${BLUE}📁 测试数据:${NC} $TEST_DATA" | tee -a "$LOG_FILE"
if [ -n "$MAX_SAMPLES" ]; then
    echo -e "${BLUE}📦 测试样本:${NC} $MAX_SAMPLES" | tee -a "$LOG_FILE"
else
    echo -e "${BLUE}📦 测试样本:${NC} 全部" | tee -a "$LOG_FILE"
fi
echo -e "${BLUE}💻 设备:${NC} $DEVICE" | tee -a "$LOG_FILE"
echo -e "${BLUE}💾 结果目录:${NC} $RUN_DIR" | tee -a "$LOG_FILE"
echo -e "${CYAN}================================================================================${NC}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 查找所有checkpoints
echo -e "${YELLOW}🔍 扫描checkpoints...${NC}" | tee -a "$LOG_FILE"
CHECKPOINTS=($(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V))

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo -e "${RED}❌ 错误: 未找到任何checkpoints在 $OUTPUT_DIR${NC}" | tee -a "$LOG_FILE"
    exit 1
fi

echo -e "${GREEN}✅ 找到 ${#CHECKPOINTS[@]} 个checkpoints${NC}" | tee -a "$LOG_FILE"

# 过滤checkpoints (根据step范围)
FILTERED_CHECKPOINTS=()
for ckpt in "${CHECKPOINTS[@]}"; do
    STEP=$(basename "$ckpt" | sed 's/checkpoint-//')
    
    # 应用step过滤
    SKIP=0
    if [ -n "$START_STEP" ] && [ "$STEP" -lt "$START_STEP" ]; then
        SKIP=1
    fi
    if [ -n "$END_STEP" ] && [ "$STEP" -gt "$END_STEP" ]; then
        SKIP=1
    fi
    if [ -n "$STEP_INTERVAL" ]; then
        if [ $((STEP % STEP_INTERVAL)) -ne 0 ]; then
            SKIP=1
        fi
    fi
    
    if [ $SKIP -eq 0 ]; then
        FILTERED_CHECKPOINTS+=("$ckpt")
        echo -e "${BLUE}  ✓ checkpoint-$STEP${NC}" | tee -a "$LOG_FILE"
    fi
done

if [ ${#FILTERED_CHECKPOINTS[@]} -eq 0 ]; then
    echo -e "${RED}❌ 错误: 过滤后无可测试的checkpoints${NC}" | tee -a "$LOG_FILE"
    exit 1
fi

echo "" | tee -a "$LOG_FILE"
echo -e "${GREEN}📋 将测试 ${#FILTERED_CHECKPOINTS[@]} 个checkpoints${NC}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 批量测试
SUCCESS_COUNT=0
FAIL_COUNT=0
declare -A RESULTS_MAP

for i in "${!FILTERED_CHECKPOINTS[@]}"; do
    CKPT="${FILTERED_CHECKPOINTS[$i]}"
    CKPT_NAME=$(basename "$CKPT")
    STEP=$(echo "$CKPT_NAME" | sed 's/checkpoint-//')
    
    echo -e "${CYAN}================================================================================${NC}" | tee -a "$LOG_FILE"
    echo -e "${CYAN}🧪 [$((i+1))/${#FILTERED_CHECKPOINTS[@]}] 测试 $CKPT_NAME${NC}" | tee -a "$LOG_FILE"
    echo -e "${CYAN}================================================================================${NC}" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    # 为此checkpoint创建临时输出目录
    CKPT_OUTPUT_DIR="$RUN_DIR/temp_$CKPT_NAME"
    mkdir -p "$CKPT_OUTPUT_DIR"
    
    # 构建测试命令
    CMD="python test_multi_position_probe_accuracy.py \
        --model_path \"$MODEL_PATH\" \
        --lora_path \"$CKPT\" \
        --dimension \"$DIMENSION\" \
        --probe_dir \"$PROBE_DIR\" \
        --test_data \"$TEST_DATA\" \
        --device \"$DEVICE\" \
        --output_dir \"$CKPT_OUTPUT_DIR\" \
        --positions $POSITIONS"
    
    if [ -n "$MAX_SAMPLES" ]; then
        CMD="$CMD --max_samples $MAX_SAMPLES"
    fi
    
    # 执行测试
    echo "$CMD" >> "$LOG_FILE"
    eval $CMD 2>&1 | tee -a "$LOG_FILE"
    
    # 检查结果
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✅ $CKPT_NAME 测试成功${NC}" | tee -a "$LOG_FILE"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        
        # 提取并重命名结果文件为分析脚本兼容格式
        LATEST_JSON=$(ls -t "$CKPT_OUTPUT_DIR"/*_multi_position_test_*.json 2>/dev/null | head -1)
        if [ -f "$LATEST_JSON" ]; then
            # 重命名格式: {combined_name}_ckpt{step}_{dimension}_multi_position.json
            TARGET_JSON="$RUN_DIR/${DIMENSION}_lora/${COMBINED_NAME}_ckpt${STEP}_${DIMENSION}_multi_position.json"
            cp "$LATEST_JSON" "$TARGET_JSON"
            RESULTS_MAP["$STEP"]="$TARGET_JSON"
            echo -e "${GREEN}   💾 结果已保存: $(basename $TARGET_JSON)${NC}" | tee -a "$LOG_FILE"
        fi
        
        # 清理临时目录
        rm -rf "$CKPT_OUTPUT_DIR"
    else
        echo -e "${RED}❌ $CKPT_NAME 测试失败${NC}" | tee -a "$LOG_FILE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    
    echo "" | tee -a "$LOG_FILE"
done

# 生成汇总报告
echo -e "${CYAN}================================================================================${NC}" | tee -a "$LOG_FILE"
echo -e "${CYAN}📊 生成汇总报告${NC}" | tee -a "$LOG_FILE"
echo -e "${CYAN}================================================================================${NC}" | tee -a "$LOG_FILE"

SUMMARY_FILE="$RUN_DIR/summary_report.txt"

{
    echo "================================================================================"
    echo "批量多位置探针测试汇总报告"
    echo "================================================================================"
    echo "测试时间: $TIMESTAMP"
    echo "LoRA目录: $OUTPUT_DIR"
    echo "维度: $DIMENSION"
    echo "测试位置: $POSITIONS"
    echo "总checkpoint数: ${#FILTERED_CHECKPOINTS[@]}"
    echo "成功: $SUCCESS_COUNT"
    echo "失败: $FAIL_COUNT"
    echo "================================================================================"
    echo ""
    echo "📊 各Checkpoint位置准确率汇总"
    echo "--------------------------------------------------------------------------------"
    echo ""
    
    # 按step排序输出
    for STEP in $(printf '%s\n' "${!RESULTS_MAP[@]}" | sort -n); do
        JSON_FILE="${RESULTS_MAP[$STEP]}"
        echo "🔹 checkpoint-$STEP"
        
        # 提取每个位置的平均准确率
        for POS in $POSITIONS; do
            # 使用Python简单提取准确率
            AVG_ACC=$(python3 - <<EOF
import json
import sys
with open('$JSON_FILE', 'r') as f:
    data = json.load(f)
if '$POS' in data:
    accs = [v['accuracy'] for v in data['$POS'].values()]
    print(f"{sum(accs)/len(accs):.4f}")
else:
    print("N/A")
EOF
            )
            echo "   $POS: $AVG_ACC"
        done
        echo ""
    done
    
} > "$SUMMARY_FILE"

cat "$SUMMARY_FILE" | tee -a "$LOG_FILE"

echo -e "${GREEN}💾 汇总报告已保存: $SUMMARY_FILE${NC}" | tee -a "$LOG_FILE"
echo -e "${GREEN}📝 完整日志: $LOG_FILE${NC}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo -e "${CYAN}================================================================================${NC}" | tee -a "$LOG_FILE"
echo -e "${GREEN}✅ 批量测试完成!${NC}" | tee -a "$LOG_FILE"
echo -e "${CYAN}================================================================================${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}成功: $SUCCESS_COUNT / ${#FILTERED_CHECKPOINTS[@]}${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}失败: $FAIL_COUNT / ${#FILTERED_CHECKPOINTS[@]}${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}结果目录: $RUN_DIR${NC}" | tee -a "$LOG_FILE"
echo -e "${CYAN}================================================================================${NC}" | tee -a "$LOG_FILE"
