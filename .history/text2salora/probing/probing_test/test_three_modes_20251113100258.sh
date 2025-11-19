#!/bin/bash

# 测试三种模式的探针准确率对比
LORA_PATH="../../protected_lora/output/safety-lora_wo_g_r16_a32-ep1-svd_rank16-salora_0_8-lr_5e-5/checkpoint-7600"
PROBE_DIR="../trained_probes/multi_position-1103/safety"
TEST_DATA="../data/safety_paired"
SUBSPACE_DIR="../../preference_subspace/saved_subspaces"
SAMPLES=100
DEVICE="cuda:0"

echo "========================================"
echo "📊 三种模式探针准确率对比测试"
echo "========================================"
echo ""
echo "测试配置:"
echo "  LoRA: $LORA_PATH"
echo "  探针: $PROBE_DIR"
echo "  样本数: $SAMPLES"
echo "  设备: $DEVICE"
echo ""
echo "========================================"

# 模式1: Base Model (无LoRA)
echo ""
echo "🔵 模式1: Base Model (无LoRA)"
echo "========================================"
CUDA_VISIBLE_DEVICES=2 /home/wangbinrui/miniconda3/envs/text-to-salora/bin/python \
  test_multi_position_probe_accuracy.py \
  --model_path Qwen/Qwen2.5-1.5B-Instruct \
  --probe_dir "$PROBE_DIR" \
  --test_data "$TEST_DATA" \
  --dimension safety \
  --position assistant_last \
  --max_samples $SAMPLES \
  --device cuda:0 \
  2>&1 | tee mode1_base_model.log | grep -E "(平均准确率|准确率 >)"

# 模式2: LoRA only (无ABC约束)
echo ""
echo "🟡 模式2: LoRA only (无ABC约束)"
echo "========================================"
CUDA_VISIBLE_DEVICES=2 /home/wangbinrui/miniconda3/envs/text-to-salora/bin/python \
  test_multi_position_probe_accuracy.py \
  --model_path Qwen/Qwen2.5-1.5B-Instruct \
  --lora_path "$LORA_PATH" \
  --probe_dir "$PROBE_DIR" \
  --test_data "$TEST_DATA" \
  --dimension safety \
  --position assistant_last \
  --max_samples $SAMPLES \
  --device cuda:0 \
  2>&1 | tee mode2_lora_only.log | grep -E "(平均准确率|准确率 >)"

# 模式3: LoRA + ABC约束
echo ""
echo "🟢 模式3: LoRA + ABC约束"
echo "========================================"
CUDA_VISIBLE_DEVICES=2 /home/wangbinrui/miniconda3/envs/text-to-salora/bin/python \
  test_multi_position_probe_accuracy_with_abc.py \
  --model_path Qwen/Qwen2.5-1.5B-Instruct \
  --lora_path "$LORA_PATH" \
  --probe_dir "$PROBE_DIR" \
  --test_data "$TEST_DATA" \
  --dimension safety \
  --positions assistant_last \
  --subspace_dir "$SUBSPACE_DIR" \
  --max_samples $SAMPLES \
  --device cuda:0 \
  2>&1 | tee mode3_lora_abc.log | grep -E "(平均准确率|准确率 >)"

echo ""
echo "========================================"
echo "📊 测试完成! 结果汇总:"
echo "========================================"
echo ""
echo "🔵 模式1 (Base Model):"
grep "平均准确率" mode1_base_model.log || echo "  [查看 mode1_base_model.log]"
echo ""
echo "🟡 模式2 (LoRA only):"
grep "平均准确率" mode2_lora_only.log || echo "  [查看 mode2_lora_only.log]"
echo ""
echo "🟢 模式3 (LoRA + ABC):"
grep "平均准确率" mode3_lora_abc.log || echo "  [查看 mode3_lora_abc.log]"
echo ""
echo "详细日志:"
echo "  mode1_base_model.log"
echo "  mode2_lora_only.log"
echo "  mode3_lora_abc.log"

