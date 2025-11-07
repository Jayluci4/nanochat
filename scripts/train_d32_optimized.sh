#!/bin/bash
# Full D32 algebra training with memory optimizations
# Estimated time: 34 hours on 8x L4 GPUs
# Estimated cost: ~$190 (vs $235 without optimizations)

set -e  # Exit on error

echo "========================================================"
echo "D32 Algebra Training - Memory Optimized"
echo "========================================================"
echo ""
echo "Optimizations enabled:"
echo "  [OK] Low-rank optimizer (saves ~12GB)"
echo "  [OK] Selective checkpointing (saves ~8GB)"
echo "  [OK] TF32 matmul (20% speedup)"
echo "  [OK] BF16 training"
echo ""
echo "Expected memory: ~10GB per GPU (vs 22GB baseline)"
echo "Expected speed: +25% faster"
echo ""

# Configuration
NPROC=8
MODEL_TAG="d32"
DEVICE_BATCH_SIZE=1
MAX_SEQ_LEN=1024
NUM_ITERATIONS=50000
LOWRANK_RANK=128

# Ask for confirmation
read -p "Start training? This will cost ~$190 and take 34 hours (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Starting training..."
echo ""

# Launch with torchrun
torchrun \
  --standalone \
  --nproc_per_node=$NPROC \
  scripts/mid_train_optimized.py \
  --model_tag=$MODEL_TAG \
  --device_batch_size=$DEVICE_BATCH_SIZE \
  --max_seq_len=$MAX_SEQ_LEN \
  --num_iterations=$NUM_ITERATIONS \
  --use_lowrank_optim=True \
  --use_checkpointing=True \
  --lowrank_rank=$LOWRANK_RANK \
  --eval_every=500 \
  --monitor_memory=False

echo ""
echo "========================================================"
echo "Training complete!"
echo "========================================================"
echo ""
echo "Check final checkpoint and run evaluation:"
echo "  python -m scripts.chat_eval -i mid"
echo ""
