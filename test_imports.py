"""
Test script to verify all new imports work correctly.
Run this before pushing to GitHub.
"""

print("Testing imports...")
print("="*60)

# Test 1: Low-rank optimizer
try:
    from nanochat.lowrank_optim import LowRankAdamW, create_lowrank_optimizer
    print("[OK] lowrank_optim module")
except Exception as e:
    print(f"[ERROR] lowrank_optim: {e}")

# Test 2: Memory efficient utilities
try:
    from nanochat.memory_efficient import (
        apply_selective_checkpointing,
        MemoryEfficientWrapper,
        print_memory_summary,
        enable_tf32,
    )
    print("[OK] memory_efficient module")
except Exception as e:
    print(f"[ERROR] memory_efficient: {e}")

# Test 3: FSDP utilities
try:
    from nanochat.fsdp_utils import get_fsdp_config, is_fsdp_available
    print("[OK] fsdp_utils module")
except Exception as e:
    print(f"[ERROR] fsdp_utils: {e}")

# Test 4: PyTorch features we rely on
try:
    import torch
    from torch.utils.checkpoint import checkpoint
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    print(f"[OK] PyTorch {torch.__version__}")
    print(f"[OK] FSDP available: {torch.distributed.is_available()}")
except Exception as e:
    print(f"[ERROR] PyTorch features: {e}")

# Test 5: Check if algebra_tutor exists
try:
    from tasks.algebra_tutor import AlgebraTutor
    task = AlgebraTutor(size=10, split="train")
    example = next(iter(task))
    assert 'messages' in example
    print("[OK] AlgebraTutor task")
except Exception as e:
    print(f"[ERROR] AlgebraTutor: {e}")

print("="*60)
print("Import tests complete!")
print("\nIf all tests passed, you're ready to:")
print("  1. Push to GitHub")
print("  2. Clone on cloud VM")
print("  3. Run training")
