#!/bin/bash
# 偏好子空间提取流程 (支持分投影层提取)
# v2: 为每个投影层(q/k/v/o/up/down)分别提取子空间

set -e  # 遇到错误立即退出

# =============================================================================
# 配置参数
# =============================================================================

# 模型配置
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"

# 数据配置
DATA_DIR="../probing/data"

# 输出配置
OUTPUT_DIR="./preference_subspace"

# 计算配置
TOP_K=64
DEVICE="cuda:0"

# 偏好维度列表
DIMENSIONS=("safety" "helpfulness" "correctness" "coherence")

# 投影层列表 (新增)
PROJECTIONS=("q_proj" "k_proj" "v_proj" "o_proj" "up_proj" "down_proj")

# 测试模式 (可选,仅用于快速测试)
TEST_MODE=""
# TEST_MODE="--max_samples 10"

# =============================================================================
# 日志函数
# =============================================================================

log_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

log_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

log_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

# =============================================================================
# 打印配置
# =============================================================================

echo ""
echo "================================================================================"
echo "                   偏好子空间提取流程 (分投影层版本)"
echo "================================================================================"
echo ""
log_info "模型: $MODEL_NAME"
log_info "数据目录: $DATA_DIR"
log_info "输出目录: $OUTPUT_DIR"
log_info "Top-K: $TOP_K"
log_info "设备: $DEVICE"
log_info "偏好维度: ${DIMENSIONS[*]}"
log_info "投影层: ${PROJECTIONS[*]}"
if [ -n "$TEST_MODE" ]; then
    log_info "测试模式: $TEST_MODE"
fi
echo ""

# =============================================================================
# 步骤 1: 提取特征差分 (循环遍历 dimension × projection)
# =============================================================================
echo ""
echo "================================================================================"
echo "步骤 1/2: 提取特征差分 (所有维度 × 所有投影层)"
echo "================================================================================"
echo ""

for DIM in "${DIMENSIONS[@]}"; do
    for PROJ in "${PROJECTIONS[@]}"; do
        log_info "处理: $DIM - $PROJ"
        
        python extract_features.py \
            --model_name "$MODEL_NAME" \
            --data_dir "$DATA_DIR" \
            --dimension "$DIM" \
            --projection "$PROJ" \
            --output_dir "$OUTPUT_DIR" \
            --device "$DEVICE" \
            $TEST_MODE
        
        if [ $? -eq 0 ]; then
            log_success "$DIM - $PROJ 特征提取完成"
        else
            log_error "$DIM - $PROJ 特征提取失败"
            exit 1
        fi
        echo ""
    done
done

# =============================================================================
# 步骤 2: 计算 SVD 并保存子空间 (循环遍历 dimension × projection)
# =============================================================================
echo ""
echo "================================================================================"
echo "步骤 2/2: 计算 SVD 分解 (所有维度 × 所有投影层)"
echo "================================================================================"
echo ""

for DIM in "${DIMENSIONS[@]}"; do
    for PROJ in "${PROJECTIONS[@]}"; do
        log_info "处理: $DIM - $PROJ"
        
        FEATURE_FILE="$OUTPUT_DIR/${DIM}_${PROJ}_feature_diff.npz"
        
        if [ ! -f "$FEATURE_FILE" ]; then
            log_error "特征文件不存在: $FEATURE_FILE"
            continue
        fi
        
        python compute_svd.py \
            --feature_file "$FEATURE_FILE" \
            --dimension "$DIM" \
            --projection "$PROJ" \
            --output_dir "$OUTPUT_DIR" \
            --top_k "$TOP_K" \
            --fuse_method "weighted_avg" \
            --device "$DEVICE"
        
        if [ $? -eq 0 ]; then
            log_success "$DIM - $PROJ SVD 计算完成"
        else
            log_error "$DIM - $PROJ SVD 计算失败"
            exit 1
        fi
        echo ""
    done
done

# =============================================================================
# 完成
# =============================================================================
echo ""
echo "================================================================================"
echo "                              完成!"
echo "================================================================================"
echo ""
log_success "所有偏好子空间 (分投影层) 已提取完成"
log_info "输出目录: $OUTPUT_DIR"
echo ""
log_info "生成的文件格式:"
echo "  - {dimension}_{projection}_feature_diff.npz           特征差分"
echo "  - {dimension}_{projection}_layer{N}_subspace.pt       各层子空间"
echo "  - {dimension}_{projection}_fused_subspace.pt          融合子空间"
echo "  - {dimension}_{projection}_meta.json                  元信息"
echo "  - {dimension}_{projection}_singular_values.png        可视化"
echo ""
log_info "投影层说明:"
echo "  - q_proj, k_proj, v_proj: Query/Key/Value 投影 (输出维度 256)"
echo "  - o_proj: Output 投影 (输出维度 1536)"
echo "  - up_proj: MLP Up 投影 (输出维度 8960)"
echo "  - down_proj: MLP Down 投影 (输出维度 1536)"
echo ""
log_info "下一步: 修改 lora_svd_init.py 加载对应投影的子空间文件"
echo ""