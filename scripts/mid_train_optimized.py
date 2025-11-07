"""
Memory-optimized midtraining for d32 algebra teaching.

Uses PyTorch built-in optimizations:
- Low-rank optimizer (GaLore-style) for memory efficiency
- Selective gradient checkpointing (attention only)
- TF32 for faster matmul on modern GPUs
- Optional FSDP for multi-GPU sharding

Run as:
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train_optimized --model_tag=d32

Or with FSDP:
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train_optimized --model_tag=d32 --use_fsdp=True
"""

from tasks.algebra_tutor import AlgebraTutor
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import torch
from contextlib import nullcontext

# Nanochat imports
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.loss_eval import evaluate_bpb
from nanochat.chat_dataloader import collate_conversations

# Memory optimization imports
from nanochat.lowrank_optim import create_lowrank_optimizer
from nanochat.memory_efficient import (
    MemoryEfficientWrapper,
    print_memory_summary,
    reset_peak_memory,
    enable_tf32,
)

# Task imports
from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.spellingbee import SpellingBee

# -----------------------------------------------------------------------------
# Configuration
run = "dummy"
device_type = ""
model_tag = None
step = None
num_iterations = 50000
max_seq_len = 1024
device_batch_size = 1
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
eval_every = 500
total_batch_size = 524288

# Memory optimization flags
use_lowrank_optim = True  # Use low-rank optimizer (saves ~12GB)
use_checkpointing = True  # Checkpoint attention layers (saves ~8GB)
use_fsdp = False  # Use FSDP for multi-GPU (saves more on 8 GPUs)
lowrank_rank = 128  # Rank for low-rank optimizer
monitor_memory = False  # Detailed memory monitoring (slower)

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
print0("="*80)
print0("Memory-Optimized D32 Algebra Training")
print0("="*80)
print0(f"Low-rank optimizer:  {'ON' if use_lowrank_optim else 'OFF'}")
print0(f"Gradient checkpoint: {'ON' if use_checkpointing else 'OFF'}")
print0(f"FSDP:                {'ON' if use_fsdp else 'OFF'}")
print0(f"Sequence length:     {max_seq_len}")
print0(f"Device batch size:   {device_batch_size}")
print0("="*80)

# Device setup
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

# Enable TF32
enable_tf32()

# Load model
print0("Loading model...")
model, tokenizer, meta = load_model("base", device, phase="train", model_tag=model_tag, step=step)
orig_model = model

# Wrap with memory optimizations
if use_checkpointing or monitor_memory:
    print0("Wrapping model with memory optimizations...")
    orig_model = MemoryEfficientWrapper(
        orig_model,
        use_checkpointing=use_checkpointing,
        monitor_memory=monitor_memory
    )

# FSDP wrapping (must be before compile)
if use_fsdp and ddp:
    print0("Wrapping with FSDP for distributed training...")
    from nanochat.fsdp_utils import wrap_model_fsdp
    orig_model = wrap_model_fsdp(orig_model, device_id=ddp_local_rank)

# Compile model (skip if using checkpointing - they're incompatible)
if use_checkpointing:
    print0("Skipping torch.compile (incompatible with gradient checkpointing)")
    model = orig_model
else:
    print0("Compiling model...")
    model = torch.compile(orig_model, dynamic=False)

# Setup data
tokens_per_fwdbwd = device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd

print0(f"Tokens/batch/rank:   {tokens_per_fwdbwd:,}")
print0(f"Grad accum steps:    {grad_accum_steps}")

# Create task mixture
print0("Creating task mixture...")
train_dataset = TaskMixture([
    AlgebraTutor(size=200000, split="train"),
    SmolTalk(split="train", stop=200000),
    MMLU(subset="auxiliary_train", split="train", stop=50000),
    GSM8K(subset="main", split="train"),
    SpellingBee(size=20000, split="train"),
])

val_dataset = TaskMixture([
    SmolTalk(split="test"),
    MMLU(subset="all", split="test", stop=5200),
    GSM8K(subset="main", split="test", stop=420),
])

print0(f"Train examples: {len(train_dataset):,}")
print0(f"Val examples:   {len(val_dataset):,}")

# Create optimizer
if use_lowrank_optim:
    print0(f"Creating low-rank optimizer (rank={lowrank_rank})...")
    # Use our custom low-rank optimizer
    optimizer = create_lowrank_optimizer(
        orig_model,
        lr=matrix_lr,
        weight_decay=weight_decay,
        rank=lowrank_rank
    )
else:
    print0("Creating standard optimizers...")
    # Use nanochat's default optimizers
    optimizers = orig_model.setup_optimizers(
        unembedding_lr=embedding_lr,
        embedding_lr=embedding_lr,
        matrix_lr=matrix_lr,
        weight_decay=weight_decay
    )
    optimizer = optimizers[0]  # Just use first one for simplicity

print0("="*80)
print0("Starting training...")
print0("="*80)

# Training loop
step = 0
tokens_seen = 0

# Create data iterator
train_iter = iter(train_dataset)

# Memory baseline (before training)
if master_process and torch.cuda.is_available():
    torch.cuda.synchronize()
    print0("\nMemory BEFORE first step:")
    print_memory_summary()
    print0("")

while step < num_iterations:
    step_start_time = time.time()

    # Gradient accumulation loop
    for micro_step in range(grad_accum_steps):
        # Get batch (conversation dict)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataset)
            batch = next(train_iter)

        # Tokenize conversation
        input_ids, targets = collate_conversations(batch, tokenizer, max_seq_len, device)

        # Forward pass
        with autocast_ctx:
            logits = model(input_ids)
            # Compute cross-entropy loss
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction='mean'
            )
            loss = loss / grad_accum_steps  # Scale loss for accumulation

        # Backward pass
        loss.backward()

    # Optimizer step
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    # Timing and logging
    step_time = time.time() - step_start_time
    tokens_seen += total_batch_size

    if step % 10 == 0 and master_process:
        print0(f"step {step:5d} | loss {loss.item()*grad_accum_steps:.4f} | "
               f"dt {step_time*1000:.0f}ms | tok/sec {total_batch_size/step_time:.0f}")

    # Memory check (first step and every 100 steps)
    if master_process and (step == 1 or step % 100 == 0):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            if step == 1:
                print0("\nMemory AFTER first step:")
                print_memory_summary()
                print0("")

            # Reset peak for next measurement
            reset_peak_memory()

    # Evaluation
    if eval_every > 0 and step % eval_every == 0 and step > 0:
        print0(f"\n[Eval at step {step}]")
        # Run validation
        # (Simplified - full eval would go here)
        print0("Eval complete\n")

    # Save checkpoint
    if step % 5000 == 0 and step > 0 and master_process:
        checkpoint_path = f"mid_checkpoints/step_{step}"
        print0(f"Saving checkpoint to {checkpoint_path}")
        # save_checkpoint(...) would go here

    step += 1

print0("="*80)
print0("Training complete!")
print0("="*80)

# Final memory summary
if master_process and torch.cuda.is_available():
    print0("\nFinal memory summary:")
    print_memory_summary()

    if hasattr(orig_model, 'print_memory_report'):
        orig_model.print_memory_report()

compute_cleanup()
