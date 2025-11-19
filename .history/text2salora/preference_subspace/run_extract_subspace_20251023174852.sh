#!/bin/bash

# 偏好子空间提取完整流程脚本

# =============================================================================
# 配置参数
# =============================================================================

# 模型路径 (使用你训练探针时用的模型)
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"

# 数据目录
DATA_DIR="../probing/data"

# 输出目录
OUTPUT_DIR="./saved_subspaces"

# 设备
DEVICE="cuda:0"

# SVD 参数
TOP_K=64

# 是否在小数据集上测试 (设置为空字符串则使用全部数据)
TEST_MODE=""  # 设置为 "--max_samples 50" 进行快速测试

# =============================================================================
# 颜色输出
# =============================================================================
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

# =============================================================================
# 主流程
# =============================================================================

echo "================================================================================"
echo "                   偏好子空间提取 - 完整流程"
echo "================================================================================"
echo ""
log_info "模型: $MODEL_NAME"
log_info "数据目录: $DATA_DIR"
log_info "输出目录: $OUTPUT_DIR"
log_info "Top-K: $TOP_K"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 偏好维度列表
DIMENSIONS=("safety" "helpfulness" "correctness" "coherence")

# =============================================================================
# 步骤 1: 提取特征差分
# =============================================================================
echo ""
echo "================================================================================"
echo "步骤 1/2: 提取特征差分"
echo "================================================================================"
echo ""

for DIM in "${DIMENSIONS[@]}"; do
    log_info "处理维度: $DIM"
    
    python extract_features.py \
        --model_name "$MODEL_NAME" \
        --data_dir "$DATA_DIR" \
        --dimension "$DIM" \
        --output_dir "$OUTPUT_DIR" \
        --device "$DEVICE" \
        $TEST_MODE
    
    if [ $? -eq 0 ]; then
        log_success "$DIM 特征提取完成"
    else
        log_error "$DIM 特征提取失败"
        exit 1
    fi
    echo ""
done

# =============================================================================
# 步骤 2: 计算 SVD 并保存子空间
# =============================================================================
echo ""
echo "================================================================================"
echo "步骤 2/2: 计算 SVD 分解"
echo "================================================================================"
echo ""

for DIM in "${DIMENSIONS[@]}"; do
    log_info "处理维度: $DIM"
    
    FEATURE_FILE="$OUTPUT_DIR/${DIM}_feature_diff.npz"
    
    if [ ! -f "$FEATURE_FILE" ]; then
        log_error "特征文件不存在: $FEATURE_FILE"
        continue
    fi
    
    python compute_svd.py \
        --feature_file "$FEATURE_FILE" \
        --dimension "$DIM" \
        --output_dir "$OUTPUT_DIR" \
        --top_k "$TOP_K" \
        --fuse_method "weighted_avg" \
        --device "$DEVICE"
    
    if [ $? -eq 0 ]; then
        log_success "$DIM SVD 计算完成"
    else
        log_error "$DIM SVD 计算失败"
        exit 1
    fi
    echo ""
done

# =============================================================================
# 完成
# =============================================================================
echo ""
echo "================================================================================"
echo "                              完成!"
echo "================================================================================"
echo ""
log_success "所有偏好子空间已提取完成"
log_info "输出目录: $OUTPUT_DIR"
echo ""
log_info "生成的文件:"
echo "  - {dimension}_feature_diff.npz          特征差分"
echo "  - {dimension}_layer{N}_subspace.pt      各层子空间"
echo "  - {dimension}_fused_subspace.pt         融合子空间"
echo "  - {dimension}_meta.json                 元信息"
echo "  - {dimension}_singular_values.png       可视化"
echo ""
log_info "下一步: 使用这些子空间训练带正交约束的 LoRA"
echo ""

