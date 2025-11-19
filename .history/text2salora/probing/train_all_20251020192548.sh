#!/bin/bash
#
# 线性探针训练脚本 - 完整配置
# 用法: bash train_all.sh [模式]
#   模式: test (快速测试), safety (仅安全性), helpsteer (仅HelpSteer), all (全部)
#

set -e  # 遇到错误立即退出

# ============================================================================
# 配置区域 - 根据需要修改
# ============================================================================

# 模型配置
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
DEVICE="cuda:2,3"  # 使用的GPU设备

# 数据配置
MAX_SAMPLES=500      # 每类最多使用多少样本
TEST_SPLIT=0.2       # 测试集比例
BATCH_SIZE=8         # 批次大小(如果显存不足,减小此值)

# 输出目录
RESULTS_DIR="results"
LOGS_DIR="logs"

# 数据路径
SAFE_PAIRS="data/safety_paired/safe_pairs.json"
HARMFUL_PAIRS="data/safety_paired/harmful_pairs.json"
HELPSTEER_DIR="data/helpsteer"

# ============================================================================
# 函数定义
# ============================================================================

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查文件是否存在
check_file() {
    if [ ! -f "$1" ]; then
        log_error "文件不存在: $1"
        return 1
    fi
    return 0
}

# 创建目录
setup_dirs() {
    mkdir -p "$RESULTS_DIR"
    mkdir -p "$LOGS_DIR"
    log_success "创建输出目录: $RESULTS_DIR, $LOGS_DIR"
}

# 检查数据文件
check_data() {
    log_info "检查数据文件..."
    
    local all_ok=true
    
    # 检查安全性数据
    if ! check_file "$SAFE_PAIRS"; then
        all_ok=false
    fi
    if ! check_file "$HARMFUL_PAIRS"; then
        all_ok=false
    fi
    
    # 检查HelpSteer数据
    for dim in helpfulness correctness coherence complexity verbosity; do
        local low_file="$HELPSTEER_DIR/$dim/${dim}_low.json"
        local high_file="$HELPSTEER_DIR/$dim/${dim}_high.json"
        if ! check_file "$low_file" || ! check_file "$high_file"; then
            all_ok=false
        fi
    done
    
    if [ "$all_ok" = true ]; then
        log_success "所有数据文件检查通过"
        return 0
    else
        log_error "部分数据文件缺失,请先运行 build_paired_data.py 和 process_real_data.py"
        return 1
    fi
}

# 显示配置信息
show_config() {
    echo "========================================================================"
    echo "  线性探针训练配置"
    echo "========================================================================"
    echo "模型:          $MODEL_NAME"
    echo "设备:          $DEVICE"
    echo "样本数:        $MAX_SAMPLES (每类)"
    echo "测试集比例:    $TEST_SPLIT"
    echo "批次大小:      $BATCH_SIZE"
    echo "结果目录:      $RESULTS_DIR"
    echo "日志目录:      $LOGS_DIR"
    echo "========================================================================"
    echo ""
}

# 训练安全性探针
train_safety() {
    log_info "开始训练安全性探针 (Prompt-Response配对)..."
    
    local output_file="$RESULTS_DIR/probe_safety.json"
    local log_file="$LOGS_DIR/train_safety_$(date +%Y%m%d_%H%M%S).log"
    
    python train_probe_paired.py \
        --model_name "$MODEL_NAME" \
        --device "$DEVICE" \
        --safe_pairs "$SAFE_PAIRS" \
        --harmful_pairs "$HARMFUL_PAIRS" \
        --max_samples "$MAX_SAMPLES" \
        --test_split "$TEST_SPLIT" \
        --output_file "$output_file" \
        2>&1 | tee "$log_file"
    
    if [ $? -eq 0 ]; then
        log_success "安全性探针训练完成: $output_file"
        log_info "日志保存到: $log_file"
        
        # 显示Top 5结果
        echo ""
        echo "Top 5 最敏感的 Attention Heads:"
        jq -r 'to_entries | sort_by(-.value) | .[0:5] | .[] | "\(.key): \(.value)"' "$output_file"
    else
        log_error "安全性探针训练失败"
        return 1
    fi
}

# 训练HelpSteer探针
train_helpsteer() {
    log_info "开始训练 HelpSteer 多维度探针..."
    
    local dimensions=("helpfulness" "correctness" "coherence" "complexity" "verbosity")
    
    for dim in "${dimensions[@]}"; do
        log_info "训练维度: $dim"
        
        local output_file="$RESULTS_DIR/probe_${dim}.json"
        local log_file="$LOGS_DIR/train_${dim}_$(date +%Y%m%d_%H%M%S).log"
        local low_file="$HELPSTEER_DIR/$dim/${dim}_low.json"
        local high_file="$HELPSTEER_DIR/$dim/${dim}_high.json"
        
        python train_linear_probe.py \
            --model_name "$MODEL_NAME" \
            --device "$DEVICE" \
            --data0_type json \
            --data0_path "$low_file" \
            --data0_text_field text \
            --data1_type json \
            --data1_path "$high_file" \
            --data1_text_field text \
            --max_samples "$MAX_SAMPLES" \
            --test_split "$TEST_SPLIT" \
            --output_file "$output_file" \
            2>&1 | tee "$log_file"
        
        if [ $? -eq 0 ]; then
            log_success "$dim 探针训练完成: $output_file"
        else
            log_error "$dim 探针训练失败"
        fi
        
        echo ""
    done
    
    log_success "HelpSteer 所有维度训练完成"
}

# 快速测试模式
test_mode() {
    log_warning "快速测试模式: 仅使用50个样本"
    
    local test_samples=50
    local output_file="$RESULTS_DIR/test_probe_safety.json"
    local log_file="$LOGS_DIR/test_$(date +%Y%m%d_%H%M%S).log"
    
    python train_probe_paired.py \
        --model_name "$MODEL_NAME" \
        --device "$DEVICE" \
        --safe_pairs "$SAFE_PAIRS" \
        --harmful_pairs "$HARMFUL_PAIRS" \
        --max_samples "$test_samples" \
        --test_split "$TEST_SPLIT" \
        --output_file "$output_file" \
        2>&1 | tee "$log_file"
    
    if [ $? -eq 0 ]; then
        log_success "测试完成: $output_file"
        echo ""
        echo "Top 5 结果:"
        jq -r 'to_entries | sort_by(-.value) | .[0:5] | .[] | "\(.key): \(.value)"' "$output_file"
    else
        log_error "测试失败"
        return 1
    fi
}

# 生成训练报告
generate_report() {
    log_info "生成训练报告..."
    
    local report_file="$RESULTS_DIR/training_report.txt"
    
    {
        echo "========================================================================"
        echo "训练报告 - $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================================================"
        echo ""
        echo "模型: $MODEL_NAME"
        echo "设备: $DEVICE"
        echo "样本数: $MAX_SAMPLES"
        echo ""
        echo "========================================================================"
        echo "结果文件:"
        echo "========================================================================"
        ls -lh "$RESULTS_DIR"/*.json 2>/dev/null || echo "无结果文件"
        echo ""
        
        # 显示每个维度的统计
        for result_file in "$RESULTS_DIR"/probe_*.json; do
            if [ -f "$result_file" ]; then
                local basename=$(basename "$result_file" .json)
                echo "--------------------------------------------------------------------"
                echo "维度: $basename"
                echo "--------------------------------------------------------------------"
                
                # 统计信息
                local avg=$(jq '[.[] | select(. != null)] | add / length' "$result_file")
                local max=$(jq '[.[] | select(. != null)] | max' "$result_file")
                local min=$(jq '[.[] | select(. != null)] | min' "$result_file")
                local high_acc=$(jq '[.[] | select(. >= 0.9)] | length' "$result_file")
                local total=$(jq 'length' "$result_file")
                
                echo "平均准确率: $avg"
                echo "最高准确率: $max"
                echo "最低准确率: $min"
                echo "高准确率(>=0.9)头数: $high_acc / $total"
                echo ""
                
                # Top 5
                echo "Top 5 Attention Heads:"
                jq -r 'to_entries | sort_by(-.value) | .[0:5] | .[] | "  \(.key): \(.value)"' "$result_file"
                echo ""
            fi
        done
        
        echo "========================================================================"
        echo "报告生成完成"
        echo "========================================================================"
    } | tee "$report_file"
    
    log_success "报告已保存: $report_file"
}

# ============================================================================
# 主程序
# ============================================================================

main() {
    local mode="${1:-all}"  # 默认模式: all
    
    echo ""
    echo "========================================================================"
    echo "  线性探针训练系统"
    echo "========================================================================"
    echo ""
    
    # 显示配置
    show_config
    
    # 设置目录
    setup_dirs
    
    # 检查数据
    if ! check_data; then
        exit 1
    fi
    
    echo ""
    log_info "训练模式: $mode"
    echo ""
    
    # 根据模式执行
    case "$mode" in
        test)
            test_mode
            ;;
        safety)
            train_safety
            ;;
        helpsteer)
            train_helpsteer
            ;;
        all)
            train_safety
            echo ""
            echo "========================================================================"
            echo ""
            train_helpsteer
            echo ""
            echo "========================================================================"
            echo ""
            generate_report
            ;;
        *)
            log_error "未知模式: $mode"
            echo "用法: bash train_all.sh [test|safety|helpsteer|all]"
            exit 1
            ;;
    esac
    
    echo ""
    echo "========================================================================"
    log_success "训练流程完成!"
    echo "========================================================================"
    echo ""
    echo "查看结果:"
    echo "  ls -lh $RESULTS_DIR/"
    echo ""
    echo "查看日志:"
    echo "  ls -lh $LOGS_DIR/"
    echo ""
}

# 运行主程序
main "$@"
