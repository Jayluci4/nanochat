#!/bin/bash
# Quick validation script - runs 100 steps to verify memory optimizations work
# Costs: ~$0.10 on 8x L4 GPUs (2-3 minutes)

echo "========================================================"
echo "Memory Optimization Validation (100 steps)"
echo "========================================================"
echo ""

# Configuration
NPROC=1  # Start with single GPU for testing
MODEL_TAG="d32"
DEVICE_BATCH_SIZE=1
MAX_SEQ_LEN=1024
NUM_ITERATIONS=100

echo "Testing configuration:"
echo "  GPUs: $NPROC"
echo "  Model: $MODEL_TAG"
echo "  Sequence length: $MAX_SEQ_LEN"
echo "  Batch size: $DEVICE_BATCH_SIZE"
echo "  Steps: $NUM_ITERATIONS"
echo ""

# Run validation
python -m scripts.mid_train_optimized \
  --model_tag=$MODEL_TAG \
  --device_batch_size=$DEVICE_BATCH_SIZE \
  --max_seq_len=$MAX_SEQ_LEN \
  --num_iterations=$NUM_ITERATIONS \
  --use_lowrank_optim=True \
  --use_checkpointing=True \
  --use_fsdp=False \
  --monitor_memory=True

echo ""
echo "========================================================"
echo "Validation complete!"
echo ""
echo "Check for:"
echo "  [OK] No OOM errors"
echo "  [OK] Memory usage shown in logs"
echo "  [OK] Loss decreasing"
echo "  [OK] Training completed 100 steps"
echo ""
echo "If all passed, run full training with:"
echo "  bash scripts/train_d32_optimized.sh"
echo "========================================================"
