#!/bin/bash
#
# full_training_phase3.sh - Complete 50K step training
# Cost: ~$235, Time: ~42 hours
#
# ONLY RUN THIS AFTER PHASE 1 AND 2 VALIDATION PASS!
#

set -e

echo "========================================================================"
echo "PHASE 3: Full Training (50,000 steps)"
echo "========================================================================"
echo ""
echo "⚠️  IMPORTANT: Have you completed and verified Phase 1 and 2?"
echo ""
echo "This will:"
echo "  - Run mid-training for ~35 hours"
echo "  - Run SFT for ~7 hours"
echo "  - Cost approximately $235"
echo ""
read -p "Type 'YES' to confirm you want to proceed: " CONFIRM

if [ "$CONFIRM" != "YES" ]; then
    echo "Training cancelled"
    exit 0
fi

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="remediation_${TIMESTAMP}"
LOG_DIR="logs/full_run_${TIMESTAMP}"

DEVICE_BATCH_SIZE_MID=2
DEVICE_BATCH_SIZE_SFT=1

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# Start logging
exec > >(tee -a "$LOG_DIR/training.log")
exec 2>&1

echo "Configuration:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Log directory: $LOG_DIR"
echo "  Start time: $(date)"
echo "  Mid-training batch size: $DEVICE_BATCH_SIZE_MID"
echo "  SFT batch size: $DEVICE_BATCH_SIZE_SFT"
echo ""
echo "========================================================================"
echo ""

START_TIME=$(date +%s)

###############################################################################
# PHASE 3A: MID-TRAINING
###############################################################################

echo "========================================================================"
echo "PHASE 3A: Mid-Training (50,000 steps)"
echo "========================================================================"
echo ""
echo "Teaching:"
echo "  - Algebra + tool use (40%)"
echo "  - Conversation retention (40%)"
echo "  - General knowledge (20%)"
echo ""
echo "Expected duration: ~35 hours"
echo "Starting at: $(date)"
echo ""

MID_START=$(date +%s)

torchrun \
  --standalone \
  --nproc_per_node=8 \
  -m scripts.mid_train \
  --base_checkpoint=base_checkpoints/d32 \
  --output_dir=$OUTPUT_DIR/mid \
  --device_batch_size=$DEVICE_BATCH_SIZE_MID \
  --max_steps=50000 \
  --log_interval=100 \
  --eval_interval=5000 \
  --save_interval=10000

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ ERROR: Mid-training failed"
    exit 1
fi

MID_END=$(date +%s)
MID_DURATION=$((MID_END - MID_START))
MID_HOURS=$((MID_DURATION / 3600))
MID_MINUTES=$(( (MID_DURATION % 3600) / 60 ))

echo ""
echo "✅ Mid-training complete"
echo "   Duration: ${MID_HOURS}h ${MID_MINUTES}m"
echo "   Checkpoint: $OUTPUT_DIR/mid/final"
echo ""

###############################################################################
# PHASE 3B: SUPERVISED FINE-TUNING
###############################################################################

echo "========================================================================"
echo "PHASE 3B: Supervised Fine-Tuning (10,000 steps)"
echo "========================================================================"
echo ""
echo "Polishing instruction-following ability"
echo "Expected duration: ~7 hours"
echo "Starting at: $(date)"
echo ""

SFT_START=$(date +%s)

torchrun \
  --standalone \
  --nproc_per_node=8 \
  -m scripts.chat_sft \
  --checkpoint=$OUTPUT_DIR/mid/final \
  --output_dir=$OUTPUT_DIR/sft \
  --device_batch_size=$DEVICE_BATCH_SIZE_SFT \
  --max_steps=10000 \
  --log_interval=50 \
  --eval_interval=2000 \
  --save_interval=5000

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ ERROR: SFT failed"
    exit 1
fi

SFT_END=$(date +%s)
SFT_DURATION=$((SFT_END - SFT_START))
SFT_HOURS=$((SFT_DURATION / 3600))
SFT_MINUTES=$(( (SFT_DURATION % 3600) / 60 ))

echo ""
echo "✅ SFT complete"
echo "   Duration: ${SFT_HOURS}h ${SFT_MINUTES}m"
echo "   Final checkpoint: $OUTPUT_DIR/sft/final"
echo ""

###############################################################################
# PHASE 3C: EVALUATION
###############################################################################

echo "========================================================================"
echo "PHASE 3C: Evaluation"
echo "========================================================================"
echo ""
echo "Testing on benchmarks..."
echo "Starting at: $(date)"
echo ""

EVAL_START=$(date +%s)

python3 -m scripts.chat_eval \
  --checkpoint=$OUTPUT_DIR/sft/final \
  --output_file=$OUTPUT_DIR/evaluation_report.md

if [ $? -ne 0 ]; then
    echo "⚠️  Evaluation had errors (non-critical)"
fi

EVAL_END=$(date +%s)
EVAL_DURATION=$((EVAL_END - EVAL_START))
EVAL_MINUTES=$((EVAL_DURATION / 60))

echo ""
echo "✅ Evaluation complete"
echo "   Duration: ${EVAL_MINUTES}m"
echo "   Report: $OUTPUT_DIR/evaluation_report.md"
echo ""

###############################################################################
# FINAL SUMMARY
###############################################################################

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(( (TOTAL_DURATION % 3600) / 60 ))

TOTAL_COST=$(echo "scale=2; 8 * 0.70 * $TOTAL_HOURS + 8 * 0.70 * $TOTAL_MINUTES / 60" | bc)

echo "========================================================================"
echo "TRAINING COMPLETE!"
echo "========================================================================"
echo ""
echo "Training Summary:"
echo "  Phase 3A (Mid-training): ${MID_HOURS}h ${MID_MINUTES}m"
echo "  Phase 3B (SFT): ${SFT_HOURS}h ${SFT_MINUTES}m"
echo "  Phase 3C (Evaluation): ${EVAL_MINUTES}m"
echo "  Total time: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
echo ""
echo "Estimated cost: \$${TOTAL_COST}"
echo ""
echo "Output files:"
echo "  - Final checkpoint: $OUTPUT_DIR/sft/final"
echo "  - Evaluation report: $OUTPUT_DIR/evaluation_report.md"
echo "  - Full logs: $LOG_DIR/training.log"
echo ""
echo "Next Steps:"
echo "  1. Review evaluation report: cat $OUTPUT_DIR/evaluation_report.md"
echo "  2. Test model manually: python3 -m scripts.chat_cli --checkpoint=$OUTPUT_DIR/sft/final"
echo "  3. Try algebra: 'Solve for x: 7x + 4 = 25'"
echo "  4. Try conversation: 'What is the capital of France?'"
echo ""
echo "Expected improvements:"
echo "  - GSM8K: 4.5% → 60-70%"
echo "  - Algebra: Should solve correctly with tool use"
echo "  - Conversation: Should maintain quality"
echo ""
echo "========================================================================"

# Create summary file
cat > $OUTPUT_DIR/training_summary.txt << SUMMARY
Nanochat D32 Remediation - Training Complete
==============================================

Completed: $(date)

Training Duration:
  - Mid-training: ${MID_HOURS}h ${MID_MINUTES}m
  - SFT: ${SFT_HOURS}h ${SFT_MINUTES}m
  - Total: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m

Estimated Cost: \$${TOTAL_COST}

Checkpoints:
  - Final model: $OUTPUT_DIR/sft/final
  - Mid-training: $OUTPUT_DIR/mid/final

Logs:
  - Training log: $LOG_DIR/training.log
  - Evaluation: $OUTPUT_DIR/evaluation_report.md

Task Mixture:
  - 40% AlgebraTutor (algebra + tool use)
  - 40% SmolTalk (conversation)
  - 10% MMLU (general knowledge)
  - 6% GSM8K (math reasoning)
  - 4% SpellingBee (tool use pattern)
SUMMARY

echo "Summary saved to: $OUTPUT_DIR/training_summary.txt"
echo ""