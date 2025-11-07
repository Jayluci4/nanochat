"""
Automatically find the maximum device_batch_size that fits in memory.
Uses binary search with checkpointing enabled.

Run as: python -m scripts.find_max_batch_size
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat.common import compute_init, compute_cleanup, autodetect_device_type, print0
from nanochat.checkpoint_manager import load_model
from torch.utils.checkpoint import checkpoint

print0("="*80)
print0("Finding Maximum Batch Size with Gradient Checkpointing")
print0("="*80)

# Setup
device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

if ddp_rank != 0:
    sys.exit(0)  # Only run on rank 0

# Load model
print0("Loading d32 model...")
model, tokenizer, meta = load_model("base", device, phase="train", model_tag="d32")

# Apply checkpointing
print0("Applying gradient checkpointing...")
for module in model.modules():
    if module.__class__.__name__ == 'CausalSelfAttention':
        original_fwd = module.forward
        module.forward = lambda *args, **kwargs: checkpoint(original_fwd, *args, **kwargs, use_reentrant=False)

# Create simple AdamW optimizer (avoid Muon's strict gradient requirements)
print0("Creating AdamW optimizer for testing...")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)

max_seq_len = 1024

def test_batch_size(batch_size):
    """Test if a batch size fits in memory."""
    print0(f"\nTesting batch_size={batch_size}...")

    try:
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Create dummy batch
        inputs = torch.randint(0, 65536, (batch_size, max_seq_len), device=device)
        targets = torch.randint(0, 65536, (batch_size, max_seq_len), device=device)

        # Forward
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        # Backward
        loss.backward()

        # Optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Check memory
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print0(f"  Peak memory: {peak_memory:.2f} GB")

        if peak_memory > 21.5:
            print0(f"  [WARNING] Too close to limit!")
            return False

        print0(f"  [OK] Fits comfortably")
        return True

    except RuntimeError as e:
        if 'out of memory' in str(e):
            print0(f"  [OOM] Doesn't fit")
            return False
        else:
            raise

# Binary search for max batch size
print0("\nStarting binary search...")
low, high = 1, 4  # L4 GPUs are smaller, start with realistic range

while low < high:
    mid = (low + high + 1) // 2

    if test_batch_size(mid):
        low = mid  # Can fit, try higher
    else:
        high = mid - 1  # OOM, try lower

print0("\n" + "="*80)
print0(f"RESULT: Maximum device_batch_size = {low}")
print0("="*80)
print0(f"\nUse this in your training:")
print0(f"  torchrun --standalone --nproc_per_node=8 -m scripts.mid_train \\")
print0(f"    --model_tag=d32 --device_batch_size={low} --max_seq_len=1024 --num_iterations=50000")

# Calculate training time and cost
world_tokens = low * 1024 * 8
grad_accum = 524288 // world_tokens
est_time_per_step = grad_accum * 2.5  # ~2.5 sec per micro-batch
total_time_sec = 50000 * est_time_per_step
total_time_hours = total_time_sec / 3600
total_time_days = total_time_hours / 24
cost = total_time_hours * 8 * 0.70

print0(f"\nTraining estimates for 50K steps:")
print0(f"  Grad accum steps: {grad_accum}")
print0(f"  Time per step: ~{est_time_per_step:.0f} sec")
print0(f"  Total time: {total_time_days:.1f} days ({total_time_hours:.0f} hours)")
print0(f"  Estimated cost: ${cost:.0f} (8 GPUs × {total_time_hours:.0f}h × $0.70/hr)")

if total_time_days > 4:
    print0(f"\n[WARNING] Training will take {total_time_days:.1f} days!")
    print0(f"[INFO] Consider reducing max_seq_len to 768 or 512 to increase batch_size")
elif total_time_days > 2:
    print0(f"\n[INFO] {total_time_days:.1f} days is acceptable for d32 scale")
else:
    print0(f"\n[OK] {total_time_days:.1f} days is good!")

print0("="*80)

compute_cleanup()
