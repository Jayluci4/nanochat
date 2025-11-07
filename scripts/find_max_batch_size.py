"""
Automatically find the maximum device_batch_size that fits in memory.
Uses binary search with checkpointing enabled.
"""

import torch
import sys
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

# Create optimizer
optimizers = model.setup_optimizers(unembedding_lr=0.2, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0)

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
        for opt in optimizers:
            opt.step()
            opt.zero_grad()

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
low, high = 1, 16

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
print0(f"  --device_batch_size={low}")
print0(f"\nGrad accumulation with 8 GPUs:")
print0(f"  World tokens/step: {low * 1024 * 8:,}")
print0(f"  Grad accum steps: {524288 // (low * 1024 * 8)}")
print0("="*80)

compute_cleanup()
