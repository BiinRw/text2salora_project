#!/bin/bash

# 使用极端分数数据训练单个维度
# 用法: ./train_extreme_single.sh <dimension>

if [ $# -eq 0 ]; then
    echo "用法: $0 <dimension>"
    echo ""
    echo "可选维度:"
    echo "  - helpfulness   (941对)"
    echo "  - correctness   (886对)"
    echo "  - coherence     (365对)"
    echo "  - verbosity     (167对)"
    echo "  - complexity    (12对 - 不推荐,数据太少)"
    echo ""
    echo "示例: $0 helpfulness"
    exit 1
fi

DIMENSION=$1
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
DEVICE="cuda:0"

# 验证维度
VALID_DIMS=("helpfulness" "correctness" "coherence" "complexity" "verbosity")
if [[ ! " ${VALID_DIMS[@]} " =~ " ${DIMENSION} " ]]; then
    echo "❌ 错误: 无效的维度 '$DIMENSION'"
    exit 1
fi

# 检查数据文件
GOOD_FILE="data/helpsteer_ultra_extreme/${DIMENSION}_good_pairs.json"
BAD_FILE="data/helpsteer_ultra_extreme/${DIMENSION}_bad_pairs.json"

if [ ! -f "$GOOD_FILE" ]; then
    echo "❌ 错误: 数据文件不存在: $GOOD_FILE"
    echo "请先运行: python build_helpsteer_extreme_scores.py"
    exit 1
fi

echo "========================================================================"
echo "🎯 训练极端分数探针 - $DIMENSION"
echo "========================================================================"
echo "数据特点: 只使用4分(好) vs 0-1分(坏), 避免2-3分混淆"
echo "========================================================================"
echo "维度: $DIMENSION"
echo "模型: $MODEL_NAME"
echo "设备: $DEVICE"
echo "好样本: $GOOD_FILE"
echo "坏样本: $BAD_FILE"
echo "========================================================================"
echo ""

# 创建输出目录
mkdir -p results_extreme
mkdir -p trained_probes_extreme/$DIMENSION

# 根据数据量调整max_samples
# 读取实际样本数
SAMPLE_COUNT=$(python -c "import json; data=json.load(open('$GOOD_FILE')); print(len(data))")
echo "📊 可用样本数: $SAMPLE_COUNT 对"

# 使用所有可用数据
MAX_SAMPLES=$SAMPLE_COUNT

# 如果数据太少,给出警告
if [ $SAMPLE_COUNT -lt 100 ]; then
    echo "⚠️  警告: 数据量太少($SAMPLE_COUNT对),训练结果可能不可靠!"
    echo "   建议: 只使用helpfulness, correctness, coherence维度"
    read -p "   是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "🚀 开始训练 (使用全部$MAX_SAMPLES对数据)..."
echo ""

# 训练探针
python train_helpsteer_dimension.py \
    --model_name $MODEL_NAME \
    --device $DEVICE \
    --good_pairs $GOOD_FILE \
    --bad_pairs $BAD_FILE \
    --max_samples $MAX_SAMPLES \
    --test_split 0.2 \
    --cv_folds 5 \
    --max_iter 2000 \
    --reg_C 1.0 \
    --output_file results_extreme/probe_${DIMENSION}.json \
    --probe_dir trained_probes_extreme/$DIMENSION

echo ""
echo "========================================================================"
echo "✅ 训练完成!"
echo "========================================================================"
echo "📂 结果文件: results_extreme/probe_${DIMENSION}.json"
echo "📂 详细结果: results_extreme/probe_${DIMENSION}_detailed.json"
echo "📂 探针模型: trained_probes_extreme/$DIMENSION/linear_probes.pkl"
echo "========================================================================"

# 生成报告
echo ""
echo "📊 训练报告:"
python << EOFREPORT
import json
import numpy as np

result_file = "results_extreme/probe_${DIMENSION}.json"
if os.path.exists(result_file):
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    test_accs = [r['test_accuracy'] for r in results.values()]
    cv_means = [r['cv_mean'] for r in results.values()]
    
    print(f"总注意力头数: {len(results)}")
    print(f"平均测试准确率: {np.mean(test_accs):.4f}")
    print(f"最高测试准确率: {np.max(test_accs):.4f}")
    print(f"标准差: {np.std(test_accs):.4f}")
    print(f"准确率 >= 0.7: {sum(1 for a in test_accs if a >= 0.7)}")
    print(f"准确率 >= 0.8: {sum(1 for a in test_accs if a >= 0.8)}")
    print(f"准确率 >= 0.9: {sum(1 for a in test_accs if a >= 0.9)}")
    
    # Top 10
    print(f"\n🏆 Top 10 最佳注意力头:")
    top_10 = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)[:10]
    for i, (head, metrics) in enumerate(top_10, 1):
        print(f"  {i}. {head}: test={metrics['test_accuracy']:.4f}, cv={metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}")
else:
    print("❌ 未找到结果文件")
EOFREPORT

