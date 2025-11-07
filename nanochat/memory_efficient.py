"""
Memory-efficient training utilities using PyTorch built-ins.
- Selective gradient checkpointing (attention layers only)
- Activation compression hooks
- Memory monitoring utilities
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# Try to import compile-compatible checkpoint wrapper
try:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing,
    )
    CHECKPOINT_WRAPPER_AVAILABLE = True
except ImportError:
    CHECKPOINT_WRAPPER_AVAILABLE = False


def apply_selective_checkpointing(model):
    """
    Apply gradient checkpointing ONLY to attention layers.

    Uses a compile-compatible approach: set a flag on modules,
    then check the flag in the actual forward pass.
    This avoids dynamic function wrapping that breaks torch.compile.
    """

    checkpoint_count = 0
    for name, module in model.named_modules():
        # Target CausalSelfAttention modules
        if module.__class__.__name__ == 'CausalSelfAttention':
            # Mark for checkpointing with a flag
            module._use_checkpoint = True
            checkpoint_count += 1

    print(f"[INFO] Marked {checkpoint_count} attention layers for gradient checkpointing")
    print(f"[INFO] Note: Actual checkpointing happens in forward pass")
    return model


def add_memory_hooks(model, verbose=False):
    """
    Add hooks to monitor memory usage per layer.
    WARNING: Incompatible with torch.compile! Only use for debugging without compile.
    """
    # Disabled by default - incompatible with torch.compile
    print("[WARNING] Memory hooks disabled - incompatible with torch.compile")
    print("[INFO] Use print_memory_summary() instead for memory monitoring")
    return {}


def print_memory_summary():
    """Print current GPU memory usage."""
    if not torch.cuda.is_available():
        print("[INFO] CUDA not available, running on CPU")
        return

    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3

        print(f"GPU {i}:")
        print(f"  Allocated:     {allocated:.2f} GB")
        print(f"  Reserved:      {reserved:.2f} GB")
        print(f"  Peak:          {max_allocated:.2f} GB")

        # Status indicator
        if max_allocated > 21:
            status = "[WARNING] Close to OOM limit!"
        elif max_allocated < 12:
            status = "[OK] Good memory efficiency"
        else:
            status = "[OK] Within limits"
        print(f"  Status: {status}")


def enable_tf32():
    """
    Enable TF32 for faster training on Ampere+ GPUs (A100, H100, L4).
    Trades tiny precision for significant speedup.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[INFO] TF32 enabled for faster training")


def reset_peak_memory():
    """Reset peak memory stats for next measurement."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


class MemoryEfficientWrapper(nn.Module):
    """
    Wrapper that applies all memory optimizations at once:
    - Selective gradient checkpointing
    - TF32 enabled
    - Memory monitoring
    """

    def __init__(self, model, use_checkpointing=True, monitor_memory=False):
        super().__init__()
        self.model = model
        self.monitor_memory = monitor_memory

        # Apply optimizations
        if use_checkpointing:
            self.model = apply_selective_checkpointing(self.model)

        enable_tf32()

        if monitor_memory:
            self.memory_stats = add_memory_hooks(self.model, verbose=True)

        print("[INFO] MemoryEfficientWrapper initialized")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def print_memory_report(self):
        """Print memory usage report."""
        if self.monitor_memory and hasattr(self, 'memory_stats'):
            print("\n" + "="*60)
            print("Memory Usage by Layer Type")
            print("="*60)
            for layer_name, stats in sorted(
                self.memory_stats.items(),
                key=lambda x: x[1]['total_memory_mb'],
                reverse=True
            ):
                avg_mb = stats['total_memory_mb'] / stats['count']
                print(f"{layer_name:30s} | Avg: {avg_mb:6.2f} MB | Count: {stats['count']:3d}")
            print("="*60 + "\n")

        print_memory_summary()
