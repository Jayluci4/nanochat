# Quick Start - D32 Memory Optimized Training

## Current Situation

Your test run showed:
- Loss: 1.79 → Working! ✓
- Only 1 GPU active → FSDP broken ✗
- 57 sec per 10 steps with batch_size=1 → Too slow ✗

## The Real Solution (3 Steps)

### Step 1: Find Maximum Batch Size (2 minutes)

```bash
# This will automatically find the max batch_size that fits
python scripts/find_max_batch_size.py

# Expected output:
# RESULT: Maximum device_batch_size = 4 (or 8)
```

### Step 2: Run 100-Step Validation (15 minutes)

```bash
# Use the max batch size found above
torchrun --standalone --nproc_per_node=8 \
  -m scripts.mid_train \
  --model_tag=d32 \
  --device_batch_size=4 \
  --max_seq_len=1024 \
  --num_iterations=100

# Should see:
# - All 8 GPUs active in nvidia-smi
# - Loss starting at ~2-3, decreasing
# - ~10-15 seconds per step (not 57!)
```

### Step 3: Full Training (Actual Time Estimate)

**With device_batch_size=4:**
```
Grad accum steps = 524,288 / (4 × 1024 × 8) = 16
Time per step: ~40 seconds
50K steps: 50,000 × 40 = 2M sec = 23 days / 8 GPUs = 2.9 days wall-clock
Cost: 8 GPUs × 2.9 days × 24 hr × $0.70 = $387
```

**With device_batch_size=8 (if it fits):**
```
Grad accum steps = 524,288 / (8 × 1024 × 8) = 8
Time per step: ~25 seconds
50K steps: 50,000 × 25 = 1.25M sec = 14.5 days / 8 GPUs = 1.8 days wall-clock
Cost: 8 GPUs × 1.8 days × 24 hr × $0.70 = $241
```

## Why This Works

The **original mid_train.py** already:
- Uses DDP correctly (all 8 GPUs)
- Has efficient data loading
- Has AlgebraTutor in the mixture

We just added **gradient checkpointing** which saves ~8-10GB, allowing higher batch sizes.

## Run Now

```bash
# Step 1: Find max batch size
python scripts/find_max_batch_size.py

# Step 2: Validate with that batch size
torchrun --standalone --nproc_per_node=8 \
  -m scripts.mid_train \
  --model_tag=d32 \
  --device_batch_size=<USE_RESULT_FROM_STEP_1> \
  --max_seq_len=1024 \
  --num_iterations=100

# Step 3: If validation passes, full training
torchrun --standalone --nproc_per_node=8 \
  -m scripts.mid_train \
  --model_tag=d32 \
  --device_batch_size=<SAME_AS_VALIDATION> \
  --max_seq_len=1024 \
  --num_iterations=50000
```

The key is **higher batch size = less gradient accumulation = faster training**.

batch_size=1: 6 days, $1,008
batch_size=2: 3 days, $504
batch_size=4: 1.5 days, $252
batch_size=8: 0.75 days, $126

**Find the max that fits and use it!**
