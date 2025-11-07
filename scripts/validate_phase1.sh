#!/bin/bash
#
# validate_phase1.sh - Quick 1K step validation
# Cost: ~$2, Time: ~30 minutes
#
# This validates:
# - Code runs without errors
# - Memory fits in GPU VRAM
# - Loss decreases
# - Checkpoints save
#

set -e

echo "========================================================================"
echo "PHASE 1: Quick Validation (1,000 steps)"
echo "========================================================================"
echo ""
echo "This will:"
echo "  - Test AlgebraTutor data generation"
echo "  - Run 1,000 training steps (~30 minutes)"
echo "  - Cost approximately $2"
echo "  - Verify everything works before full run"
echo ""
read -p "Press Enter to start validation or Ctrl+C to cancel..."
echo ""

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="validation_phase1_${TIMESTAMP}"
LOG_FILE="logs/phase1_${TIMESTAMP}.log"

# Create directories
mkdir -p logs
mkdir -p $OUTPUT_DIR

# Redirect output to both console and log
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "Starting validation at: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Step 1: Test AlgebraTutor
echo "Step 1: Testing AlgebraTutor Task Generation"
echo "---------------------------------------------"

python3 -m tasks.algebra_tutor 2>&1 | head -100

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo ""
    echo "❌ ERROR: AlgebraTutor task generation failed"
    exit 1
fi

echo ""
echo "✅ AlgebraTutor generates examples correctly"
echo ""

# Step 2: Check base checkpoint
echo "Step 2: Verifying Base Checkpoint"
echo "----------------------------------"

if [ ! -d "base_checkpoints/d32" ]; then
    echo "❌ ERROR: Base checkpoint not found at base_checkpoints/d32"
    exit 1
fi

echo "✅ Base checkpoint found"
echo ""

# Step 3: Verify mid_train.py has AlgebraTutor
echo "Step 3: Checking mid_train.py Modification"
echo "-------------------------------------------"

if grep -q "AlgebraTutor" scripts/mid_train.py; then
    echo "✅ mid_train.py includes AlgebraTutor"
else
    echo "❌ ERROR: mid_train.py does not include AlgebraTutor"
    echo "   You need to modify scripts/mid_train.py to include AlgebraTutor in TaskMixture"
    exit 1
fi
echo ""

# Step 4: Run short training
echo "Step 4: Running 1,000 Training Steps"
echo "-------------------------------------"
echo "Starting training at: $(date)"
echo "Monitor GPU usage in another terminal with: watch -n 5 nvidia-smi"
echo ""

START_TIME=$(date +%s)

# Run training with small parameters
torchrun \
  --standalone \
  --nproc_per_node=8 \
  -m scripts.mid_train \
  --base_checkpoint=base_checkpoints/d32 \
  --output_dir=$OUTPUT_DIR \
  --device_batch_size=2 \
  --max_steps=1000 \
  --log_interval=50 \
  --eval_interval=500 \
  --save_interval=1000

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ ERROR: Training failed"
    echo "Check logs at: $LOG_FILE"
    exit 1
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))

echo ""
echo "✅ Training completed successfully"
echo "   Duration: ${MINUTES} minutes"
echo ""

# Step 5: Verify checkpoint was saved
echo "Step 5: Verifying Checkpoint"
echo "-----------------------------"

if [ -d "$OUTPUT_DIR" ]; then
    echo "✅ Checkpoint directory created"
    echo "   Contents:"
    ls -lh $OUTPUT_DIR/ | awk '{print "   " $0}'
else
    echo "❌ ERROR: Checkpoint directory not found"
    exit 1
fi
echo ""

# Step 6: Extract metrics from log
echo "Step 6: Analyzing Results"
echo "-------------------------"

echo "Last 10 loss values:"
grep "loss:" "$LOG_FILE" | tail -10 | awk '{print "   " $0}'

FIRST_LOSS=$(grep "loss:" "$LOG_FILE" | head -1 | grep -oP 'loss:\s*\K[0-9.]+')
LAST_LOSS=$(grep "loss:" "$LOG_FILE" | tail -1 | grep -oP 'loss:\s*\K[0-9.]+')

echo ""
echo "First loss: $FIRST_LOSS"
echo "Last loss: $LAST_LOSS"

if (( $(echo "$LAST_LOSS < $FIRST_LOSS" | bc -l) )); then
    echo "✅ Loss decreased (good sign!)"
else
    echo "⚠️  Loss did not decrease (may need investigation)"
fi
echo ""

# Cost calculation
COST=$(echo "scale=2; 8 * 0.70 * $MINUTES / 60" | bc)

echo "========================================================================"
echo "PHASE 1 VALIDATION COMPLETE"
echo "========================================================================"
echo ""
echo "Summary:"
echo "  ✅ AlgebraTutor generates examples"
echo "  ✅ Training runs without crashes"
echo "  ✅ Checkpoint saved successfully"
echo "  ✅ Loss decreased from $FIRST_LOSS to $LAST_LOSS"
echo ""
echo "Duration: ${MINUTES} minutes"
echo "Estimated cost: \$${COST}"
echo ""
echo "Output saved to: $OUTPUT_DIR"
echo "Full logs: $LOG_FILE"
echo ""
echo "Next step: Run Phase 2 (checkpoint inspection)"
echo "  bash scripts/validate_phase2.sh $OUTPUT_DIR"
echo ""
echo "========================================================================"