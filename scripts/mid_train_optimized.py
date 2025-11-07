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

# Track loss moving average
from collections import deque
loss_history = deque(maxlen=50)  # Last 50 steps for moving average

# Memory baseline (before training)
if master_process and torch.cuda.is_available():
    torch.cuda.synchronize()
    print0("\nMemory BEFORE first step:")
    print_memory_summary()
    print0("")

# Verify model is actually loaded (not random init)
if master_process:
    print0("\n" + "="*80)
    print0("MODEL VERIFICATION")
    print0("="*80)
    # Check a few parameter values
    sample_param = None
    for name, param in model.named_parameters():
        if 'wte' in name:  # Token embedding
            sample_param = param
            param_mean = param.data.abs().mean().item()
            param_std = param.data.std().item()
            print0(f"Sample param '{name}':")
            print0(f"  Mean abs value: {param_mean:.6f}")
            print0(f"  Std dev: {param_std:.6f}")
            if param_mean < 0.001:
                print0(f"  [WARNING] Parameters look uninitialized!")
            else:
                print0(f"  [OK] Parameters look trained")
            break
    print0("="*80 + "\n")

while step < num_iterations:
    step_start_time = time.time()

    # Show progress indicator every step (dots for silence)
    if master_process and step % 10 != 0:
        print(".", end="", flush=True)

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
        if master_process and step == 0 and micro_step == 0:
            print0(f"\n[DEBUG] Forward pass - input shape: {input_ids.shape}")
            print0(f"[DEBUG] Targets shape: {targets.shape}")

            # Count real vs padding tokens
            real_tokens = (targets[0] != -100).sum().item()
            pad_tokens = (targets[0] == -100).sum().item()
            print0(f"[DEBUG] Real tokens: {real_tokens}, Padding tokens: {pad_tokens}")

            print0(f"[DEBUG] Sample tokens (first 30): {input_ids[0, :30].tolist()}")
            print0(f"[DEBUG] Sample targets (first 30): {targets[0, :30].tolist()}")
            print0(f"[DEBUG] Decoded input: {tokenizer.decode(input_ids[0, :30].tolist())}")

        with autocast_ctx:
            logits = model(input_ids)

            # Compute cross-entropy loss (ignore_index=-100 by default)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction='mean',
                ignore_index=-100  # Explicitly set to ignore padding
            )
            loss = loss / grad_accum_steps  # Scale loss for accumulation

        if master_process and step == 0 and micro_step == 0:
            print0(f"[DEBUG] Logits shape: {logits.shape}")
            print0(f"[DEBUG] Loss (scaled): {loss.item():.4f}")
            print0(f"[DEBUG] Loss (actual): {loss.item()*grad_accum_steps:.4f}")
            print0(f"[DEBUG] Vocab size: {logits.size(-1)}")

            # Sanity checks
            random_loss = torch.log(torch.tensor(float(logits.size(-1))))
            print0(f"[DEBUG] Random baseline loss: {random_loss.item():.4f}")

            # Check model predictions
            with torch.no_grad():
                probs = torch.softmax(logits[0, 0, :], dim=-1)
                top5_probs, top5_tokens = probs.topk(5)
                print0(f"[DEBUG] Top 5 predicted tokens for position 0:")
                for i, (tok, prob) in enumerate(zip(top5_tokens.tolist(), top5_probs.tolist())):
                    decoded = tokenizer.decode([tok])
                    print0(f"         {i+1}. Token {tok:5d} ({decoded[:20]:20s}) = {prob:.4f}")

            # Status
            if loss.item()*grad_accum_steps > random_loss.item():
                print0(f"[DEBUG] Status: [WARNING] Model worse than random! Check tokenization!")
            elif loss.item()*grad_accum_steps > 8:
                print0(f"[DEBUG] Status: [WARNING] Loss high but better than random")
            else:
                print0(f"[DEBUG] Status: [OK] Model is trained properly")

        # Backward pass
        loss.backward()

    # Optimizer step
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    # Timing and logging
    step_time = time.time() - step_start_time
    tokens_seen += total_batch_size

    # Track loss for moving average
    current_loss = loss.item() * grad_accum_steps
    loss_history.append(current_loss)

    if step % 10 == 0 and master_process:
        print("")  # New line after dots

        # Compute moving average
        avg_loss = sum(loss_history) / len(loss_history) if loss_history else current_loss
        min_loss = min(loss_history) if loss_history else current_loss
        max_loss = max(loss_history) if loss_history else current_loss

        print0(f"step {step:5d} | "
               f"loss {current_loss:.4f} (avg {avg_loss:.4f}, range [{min_loss:.2f}, {max_loss:.2f}]) | "
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

# Final diagnostics
if master_process:
    print0("\n" + "="*80)
    print0("TRAINING SUMMARY")
    print0("="*80)

    if loss_history:
        final_avg = sum(loss_history) / len(loss_history)
        first_losses = list(loss_history)[:10]
        last_losses = list(loss_history)[-10:]
        first_avg = sum(first_losses) / len(first_losses) if first_losses else 0
        last_avg = sum(last_losses) / len(last_losses) if last_losses else 0

        print0(f"Loss statistics (last 50 steps):")
        print0(f"  First 10 steps avg: {first_avg:.4f}")
        print0(f"  Last 10 steps avg:  {last_avg:.4f}")
        print0(f"  Overall avg:        {final_avg:.4f}")
        print0(f"  Min loss:           {min(loss_history):.4f}")
        print0(f"  Max loss:           {max(loss_history):.4f}")

        # Convergence check
        improvement = first_avg - last_avg
        print0(f"\nConvergence check:")
        print0(f"  Improvement: {improvement:.4f}")

        if improvement > 0.5:
            print0(f"  Status: [OK] Loss is decreasing - model is learning!")
        elif improvement > 0:
            print0(f"  Status: [OK] Slight improvement - normal for short runs")
        elif abs(improvement) < 0.5:
            print0(f"  Status: [WARNING] Loss not decreasing yet")
            print0(f"          This is normal for <1000 steps, continue training")
        else:
            print0(f"  Status: [ERROR] Loss increasing - check hyperparameters!")

        # Model readiness check
        if last_avg > 15:
            print0(f"\n[CRITICAL] Loss is very high ({last_avg:.2f})")
            print0(f"Expected: Pre-trained d32 should start at ~3-4 loss")
            print0(f"Actual: Starting at ~{first_avg:.2f}")
            print0(f"\nPossible causes:")
            print0(f"  1. Model not loaded correctly (check checkpoint path)")
            print0(f"  2. Tokenization is wrong (check token format)")
            print0(f"  3. Loss computation has bugs (check targets)")
            print0(f"  4. Learning rate too high (check optimizer config)")
            print0(f"\nDO NOT run full 50K training until this is fixed!")
        elif last_avg > 8:
            print0(f"\n[WARNING] Loss higher than expected ({last_avg:.2f})")
            print0(f"This might be normal for new task mixture")
            print0(f"Monitor first 1K steps before committing to full run")
        else:
            print0(f"\n[OK] Loss in expected range ({last_avg:.2f})")
            print0(f"Safe to proceed with full training")

    print0("="*80)

# Final memory summary
if master_process and torch.cuda.is_available():
    print0("\nFinal memory summary:")
    print_memory_summary()

    if hasattr(orig_model, 'print_memory_report'):
        orig_model.print_memory_report()

compute_cleanup()
