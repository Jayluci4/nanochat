#!/bin/bash
# Validation using the ORIGINAL mid_train.py + checkpointing
# This is proven code, just with one optimization added

echo "========================================================"
echo "Memory Optimization Validation"
echo "Using: Original mid_train.py + gradient checkpointing"
echo "========================================================"
echo ""

# Configuration
NPROC=8
MODEL_TAG="d32"
DEVICE_BATCH_SIZE=2  # Try 2 (checkpointing should save ~8GB)
MAX_SEQ_LEN=1024
NUM_ITERATIONS=100

echo "Testing configuration:"
echo "  GPUs: $NPROC"
echo "  Device batch size: $DEVICE_BATCH_SIZE"
echo "  Sequence length: $MAX_SEQ_LEN"
echo "  Steps: $NUM_ITERATIONS"
echo "  Optimization: Selective gradient checkpointing"
echo ""

# Run with original mid_train.py
torchrun \
  --standalone \
  --nproc_per_node=$NPROC \
  -m scripts.mid_train \
  --model_tag=$MODEL_TAG \
  --device_batch_size=$DEVICE_BATCH_SIZE \
  --max_seq_len=$MAX_SEQ_LEN \
  --num_iterations=$NUM_ITERATIONS

echo ""
echo "========================================================"
echo "Check results:"
echo "  [OK] No OOM errors?"
echo "  [OK] All 8 GPUs active in nvidia-smi?"
echo "  [OK] Loss decreasing (not NaN)?"
echo "  [OK] Memory per GPU < 20GB?"
echo ""
echo "If yes to all, increase device_batch_size to 4 and retry."
echo "If OOM, fallback to device_batch_size=1."
echo "========================================================"
