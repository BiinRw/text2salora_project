#!/bin/bash

# 使用极端分数数据训练单个维度
# 用法: ./train_extreme_single.sh <dimension> [--merged|--paired]

if [ $# -eq 0 ]; then
    echo "用法: $0 <dimension> [--merged|--paired]"
    echo ""
    echo "可选维度:"
    echo "  - helpfulness"
    echo "  - correctness"
    echo "  - coherence"
    echo "  - verbosity"
    echo "  - complexity"
    echo "  - coding        (HelpSteer3, 编程质量)"
    echo ""
    echo "数据选项:"
    echo "  (默认)     原始 ultra-extreme 数据"
    echo "  --merged   整合数据,不保证配对 (数据量大)"
    echo "  --paired   配对整合数据,保证同prompt (质量高) ⭐推荐"
    echo ""
    echo "数据量对比:"
    echo "  维度          Ultra    Merged    Paired"
    echo "  helpfulness   385      16,485    1,295  ⭐"
    echo "  correctness   398      18,960    1,295  ⭐"
    echo "  coherence     80       30,599    154    ⭐"
    echo "  verbosity     59       928       153    ⭐"
    echo "  complexity    1        238       6"
    echo ""
    echo "示例:"
    echo "  $0 helpfulness --paired  # 使用配对数据 (推荐!)"
    echo "  $0 helpfulness --merged  # 使用整合数据 (数据量大)"
    echo "  $0 helpfulness           # 使用原始数据"
    echo "  $0 coding                # 使用HelpSteer3数据"
    exit 1
fi

DIMENSION=$1
USE_MERGED=false
USE_PAIRED=false

# 检查数据选项
if [ "$2" = "--merged" ]; then
    USE_MERGED=true
elif [ "$2" = "--paired" ]; then
    USE_PAIRED=true
fi

MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
DEVICE="cuda:1"

# 验证维度
VALID_DIMS=("helpfulness" "correctness" "coherence" "complexity" "verbosity" "coding")
if [[ ! " ${VALID_DIMS[@]} " =~ " ${DIMENSION} " ]]; then
    echo "❌ 错误: 无效的维度 '$DIMENSION'"
    exit 1
fi

# 根据维度和参数选择数据源
if [ "$DIMENSION" = "coding" ]; then
    # HelpSteer3 coding 数据
    GOOD_FILE="data/helpsteer3_coding/coding_good_pairs.json"
    BAD_FILE="data/helpsteer3_coding/coding_bad_pairs.json"
    OUTPUT_DIR="results_coding"
    PROBE_DIR="trained_probes_coding"
    DATA_DESC="HelpSteer3 编程质量对比数据 (score = ±3)"
elif [ "$USE_PAIRED" = true ]; then
    # HelpSteer + HelpSteer2 配对整合数据 (推荐!)
    GOOD_FILE="data/helpsteer_merged_paired/${DIMENSION}_good_pairs.json"
    BAD_FILE="data/helpsteer_merged_paired/${DIMENSION}_bad_pairs.json"
    OUTPUT_DIR="results_paired"
    PROBE_DIR="trained_probes_paired/$DIMENSION"
    DATA_DESC="HelpSteer+HelpSteer2配对数据 (同prompt 0分vs4分) ⭐推荐"
elif [ "$USE_MERGED" = true ]; then
    # HelpSteer + HelpSteer2 整合数据 (不保证配对)
    GOOD_FILE="data/helpsteer_merged_ultra/${DIMENSION}_good_pairs.json"
    BAD_FILE="data/helpsteer_merged_ultra/${DIMENSION}_bad_pairs.json"
    OUTPUT_DIR="results_merged"
    PROBE_DIR="trained_probes_merged/$DIMENSION"
    DATA_DESC="HelpSteer+HelpSteer2整合数据 (4分vs0分, 不保证配对)"
else
    # HelpSteer ultra extreme 数据 (原始)
    GOOD_FILE="data/helpsteer_ultra_extreme/${DIMENSION}_good_pairs.json"
    BAD_FILE="data/helpsteer_ultra_extreme/${DIMENSION}_bad_pairs.json"
    OUTPUT_DIR="results_ultra_extreme"
    PROBE_DIR="trained_probes_extreme/$DIMENSION"
    DATA_DESC="HelpSteer 原始超极端数据 (4分vs0分)"
fi

# 检查数据文件
if [ ! -f "$GOOD_FILE" ]; then
    echo "❌ 错误: 数据文件不存在: $GOOD_FILE"
    if [ "$DIMENSION" = "coding" ]; then
        echo "请先运行: python build_helpsteer3_coding_data.py"
    elif [ "$USE_PAIRED" = true ]; then
        echo "请先运行: python merge_helpsteer_paired.py"
    elif [ "$USE_MERGED" = true ]; then
        echo "请先运行: python merge_helpsteer_datasets.py"
    else
        echo "请先运行: python build_helpsteer_ultra_extreme.py"
    fi
    exit 1
fi

echo "========================================================================"
echo "🎯 训练极端分数探针 - $DIMENSION"
echo "========================================================================"
echo "数据特点: $DATA_DESC"
echo "========================================================================"
echo "维度: $DIMENSION"
echo "模型: $MODEL_NAME"
echo "设备: $DEVICE"
echo "好样本: $GOOD_FILE"
echo "坏样本: $BAD_FILE"
echo "========================================================================"
echo ""

# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p $PROBE_DIR

# 读取实际样本数
SAMPLE_COUNT=$(python -c "import json; data=json.load(open('$GOOD_FILE')); print(len(data))")
echo "📊 可用样本数: $SAMPLE_COUNT 对"

# 验证配对 (如果使用 paired 数据)
if [ "$USE_PAIRED" = true ]; then
    echo "🔍 验证配对..."
    python << EOFVERIFY
import json
with open('$GOOD_FILE', 'r') as f:
    good = json.load(f)
with open('$BAD_FILE', 'r') as f:
    bad = json.load(f)

assert len(good) == len(bad), "Good/Bad 数量不匹配!"

good_prompts = [item['prompt'] for item in good]
bad_prompts = [item['prompt'] for item in bad]
assert good_prompts == bad_prompts, "Prompt 不匹配!"

print("✅ 配对验证通过: 100% 匹配 (同一 prompt 的 0分 vs 4分)")
EOFVERIFY
    if [ $? -ne 0 ]; then
        echo "❌ 配对验证失败!"
        exit 1
    fi
fi

# 使用所有可用数据
MAX_SAMPLES=$SAMPLE_COUNT

# 如果数据太少,给出警告
if [ $SAMPLE_COUNT -lt 100 ]; then
    echo "⚠️  警告: 数据量太少($SAMPLE_COUNT对),训练结果可能不可靠!"
    echo "   建议: 使用 --paired 参数来使用配对整合数据集"
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
    --output_file ${OUTPUT_DIR}/probe_${DIMENSION}.json \
    --probe_dir $PROBE_DIR

echo ""
echo "========================================================================"
echo "✅ 训练完成!"
echo "========================================================================"
echo "📂 结果文件: ${OUTPUT_DIR}/probe_${DIMENSION}.json"
echo "📂 详细结果: ${OUTPUT_DIR}/probe_${DIMENSION}_detailed.json"
echo "📂 探针模型: ${PROBE_DIR}/linear_probes.pkl"
echo "========================================================================"

# 生成报告
echo ""
echo "📊 训练报告:"
python << EOFREPORT
import json
import numpy as np
import os

result_file = "${OUTPUT_DIR}/probe_${DIMENSION}.json"
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
