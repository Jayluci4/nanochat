"""
Simple memory-optimized midtraining - JUST gradient checkpointing, no fancy tricks.

The Karpathy way: Make it work first.

Run as:
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train_simple --model_tag=d32 --device_batch_size=2
"""

from tasks.algebra_tutor import AlgebraTutor
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import torch
from contextlib import nullcontext
from collections import deque

# Nanochat imports
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir, autodetect_device_type
from nanochat.checkpoint_manager import load_model, save_checkpoint

# Task imports
from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.spellingbee import SpellingBee

# Simple selective checkpointing
from torch.utils.checkpoint import checkpoint

# -----------------------------------------------------------------------------
# Configuration
run = "dummy"
device_type = ""
model_tag = None
num_iterations = 50000
max_seq_len = 1024
device_batch_size = 2  # Try 2, fallback to 1 if OOM
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
eval_every = 500
total_batch_size = 524288
use_checkpointing = True  # The ONE optimization we use

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())

# -----------------------------------------------------------------------------
print0("="*80)
print0("Simple Memory-Optimized Training")
print0("="*80)
print0(f"Strategy: Gradient checkpointing (attention layers)")
print0(f"Device batch size: {device_batch_size}")
print0(f"Sequence length: {max_seq_len}")
print0("="*80)

# Device setup
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

# Load model
print0("Loading model...")
model, tokenizer, meta = load_model("base", device, phase="train", model_tag=model_tag)

# Apply selective checkpointing (ONLY optimization)
if use_checkpointing:
    print0("Applying gradient checkpointing to attention layers...")
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'CausalSelfAttention':
            original_forward = module.forward

            def make_checkpointed(fwd):
                def checkpointed_fwd(*args, **kwargs):
                    return checkpoint(fwd, *args, **kwargs, use_reentrant=False)
                return checkpointed_fwd

            module.forward = make_checkpointed(original_forward)

print0(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

# Data setup
tokens_per_fwdbwd = device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd

print0(f"Tokens/batch/rank: {tokens_per_fwdbwd:,}")
print0(f"World tokens/step: {world_tokens_per_fwdbwd:,}")
print0(f"Grad accum steps: {grad_accum_steps}")

if grad_accum_steps > 32:
    print0(f"[WARNING] High gradient accumulation ({grad_accum_steps} micro-batches per step)")
    print0(f"[INFO] This is SLOW. Consider increasing device_batch_size if memory allows.")

# Task mixture
print0("Creating task mixture...")
train_dataset = TaskMixture([
    AlgebraTutor(size=200000, split="train"),
    SmolTalk(split="train", stop=200000),
    MMLU(subset="auxiliary_train", split="train", stop=50000),
    GSM8K(subset="main", split="train"),
    SpellingBee(size=20000, split="train"),
])

print0(f"Train examples: {len(train_dataset):,}")

# Optimizer
print0("Creating optimizers...")
optimizers = model.setup_optimizers(
    unembedding_lr=embedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay
)

# Dataloader helper
def get_batch(dataset, tokenizer):
    """Get and tokenize a batch."""
    batch = next(iter(dataset))
    ids, mask = tokenizer.render_conversation(batch, max_tokens=max_seq_len)

    # Convert to tensors and shift for autoregressive
    ids_tensor = torch.tensor(ids, dtype=torch.long, device=device)
    mask_tensor = torch.tensor(mask, dtype=torch.long, device=device)

    # Inputs and targets
    inputs = ids_tensor[:-1]
    targets = ids_tensor[1:]
    target_mask = mask_tensor[1:]
    targets[target_mask == 0] = -1  # Ignore non-assistant tokens

    return inputs.unsqueeze(0), targets.unsqueeze(0)

print0("="*80)
print0("Starting training...")
print0("="*80)

# Training loop
loss_history = deque(maxlen=50)

for step in range(num_iterations):
    step_start_time = time.time()

    # Accumulate gradients
    for micro_step in range(grad_accum_steps):
        inputs, targets = get_batch(train_dataset, tokenizer)

        with autocast_ctx:
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            ) / grad_accum_steps

        loss.backward()

    # Optimizer step
    for opt in optimizers:
        torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 1.0)
        opt.step()
        opt.zero_grad()

    # Logging
    step_time = time.time() - step_start_time
    current_loss = loss.item() * grad_accum_steps
    loss_history.append(current_loss)

    if step % 10 == 0 and master_process:
        avg_loss = sum(loss_history) / len(loss_history) if loss_history else current_loss
        print0(f"step {step:5d} | loss {current_loss:.4f} (avg {avg_loss:.4f}) | "
               f"dt {step_time:.1f}s | tok/sec {total_batch_size/step_time:.0f}")

        # NaN check
        if torch.isnan(loss):
            print0("[ERROR] Loss is NaN! Training diverged. Stopping.")
            break

print0("Training complete!")
compute_cleanup()
