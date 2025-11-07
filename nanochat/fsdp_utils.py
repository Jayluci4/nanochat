"""
FSDP (Fully Sharded Data Parallel) utilities.
PyTorch's native ZeRO-3 equivalent - shards model, optimizer, and gradients.

FSDP is better than DDP for large models:
- DDP: Each GPU stores full model + full optimizer (redundant!)
- FSDP: Shards everything across GPUs (efficient!)
"""

import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from functools import partial


def get_fsdp_config():
    """
    Get FSDP configuration optimized for L4 GPUs.

    Uses SHARD_GRAD_OP strategy (ZeRO-2 equivalent):
    - Shards optimizer states and gradients
    - Keeps model replicated (faster than full ZeRO-3)
    - Good balance of memory and speed
    """

    # Mixed precision config
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    return {
        'sharding_strategy': ShardingStrategy.SHARD_GRAD_OP,  # ZeRO-2 equivalent
        'mixed_precision': mixed_precision_policy,
        'use_orig_params': True,  # Better for optimizer compatibility
        'limit_all_gathers': True,  # Reduce communication overhead
        'sync_module_states': True,  # Ensure all GPUs start synchronized
    }


def wrap_model_fsdp(model, device_id=None):
    """
    Wrap model with FSDP for memory-efficient distributed training.

    Args:
        model: The model to wrap
        device_id: GPU device ID (for multi-GPU)

    Returns:
        FSDP-wrapped model
    """

    # Auto-wrap policy: wrap each transformer block
    # This gives good granularity for sharding
    auto_wrap_policy = ModuleWrapPolicy(
        {torch.nn.TransformerEncoderLayer, torch.nn.TransformerDecoderLayer}
    )

    fsdp_config = get_fsdp_config()

    wrapped_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=device_id,
        **fsdp_config
    )

    return wrapped_model


def is_fsdp_available():
    """Check if FSDP can be used (requires distributed setup)."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def print_fsdp_memory_summary(model):
    """Print FSDP memory statistics."""
    if isinstance(model, FSDP):
        # FSDP tracks sharded memory separately
        print("\n[FSDP Memory Summary]")
        print(f"  Sharded params: {model.num_params_to_shard:,}")
        print(f"  Full params:    {sum(p.numel() for p in model.parameters()):,}")
    else:
        print("[INFO] Model not wrapped with FSDP")
