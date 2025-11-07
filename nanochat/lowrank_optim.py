"""
Low-rank optimizer inspired by GaLore but using only PyTorch built-ins.
No external dependencies, simple and production-ready.

The idea: Project gradients to low-rank subspace, update there, project back.
Memory: Instead of storing momentum/variance for all 1.9B params,
        store them for rank*params (~128*1.9M = 0.24B params)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any


class LowRankAdamW(torch.optim.Optimizer):
    """
    AdamW with low-rank gradient projection to save memory.

    Simple implementation:
    1. Every N steps, compute SVD of gradient to find low-rank subspace
    2. Store momentum/variance in that subspace (small!)
    3. Update in low-rank space, project back to full space
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        rank=128,
        update_proj_gap=200,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rank=rank,
            update_proj_gap=update_proj_gap,
        )
        super().__init__(params, defaults)

        self.step_count = 0

    def _maybe_update_projector(self, p, state, rank):
        """Update projection matrix using SVD of recent gradients."""
        if 'projector' not in state or self.step_count % self.defaults['update_proj_gap'] == 0:
            # Compute low-rank approximation of gradient
            grad_2d = p.grad.reshape(-1, 1) if p.grad.dim() == 1 else p.grad.reshape(p.grad.shape[0], -1)

            # Ensure rank doesn't exceed matrix dimensions
            max_rank = min(rank, min(grad_2d.shape) - 1)
            if max_rank < 1:
                # Skip projection for tiny matrices
                state['projector'] = None
                return

            # For very wide matrices, use randomized SVD approximation
            if grad_2d.shape[1] > rank * 4:
                # Simple randomized approximation
                random_proj = torch.randn(grad_2d.shape[1], max_rank, device=grad_2d.device, dtype=grad_2d.dtype)
                projected = grad_2d @ random_proj
                # Ensure q doesn't exceed projected dimensions
                q = min(max_rank, min(projected.shape) - 1)
                if q < 1:
                    state['projector'] = None
                    return
                U, _, _ = torch.svd_lowrank(projected, q=q)
            else:
                # Use PyTorch's low-rank SVD
                U, _, _ = torch.svd_lowrank(grad_2d, q=max_rank)

            state['projector'] = U  # Shape: [param_size, rank]

            # Initialize momentum/variance in low-rank space if needed
            if 'exp_avg_lowrank' not in state:
                state['exp_avg_lowrank'] = torch.zeros(U.shape[1], device=p.device, dtype=p.dtype)
                state['exp_avg_sq_lowrank'] = torch.zeros(U.shape[1], device=p.device, dtype=p.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        """Single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_count += 1

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            rank = group['rank']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay (AdamW style - directly to params)
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                state = self.state[p]

                # For small parameters, use standard Adam (not worth the projection overhead)
                if p.numel() < 1000:
                    # Standard Adam
                    if len(state) == 0:
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    step_size = group['lr']

                    p.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    # Low-rank Adam for large parameters
                    self._maybe_update_projector(p, state, rank)

                    # Check if projection succeeded
                    if state.get('projector') is None:
                        # Fall back to standard Adam if projection failed
                        if 'exp_avg' not in state:
                            state['exp_avg'] = torch.zeros_like(p)
                            state['exp_avg_sq'] = torch.zeros_like(p)

                        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                        denom = exp_avg_sq.sqrt().add_(group['eps'])
                        step_size = group['lr']

                        p.addcdiv_(exp_avg, denom, value=-step_size)
                    else:
                        # Project gradient to low-rank space
                        grad_flat = grad.reshape(-1)
                        projector = state['projector']
                        grad_lowrank = projector.T @ grad_flat  # Shape: [rank]

                        # Update momentum and variance in low-rank space
                        exp_avg_lr = state['exp_avg_lowrank']
                        exp_avg_sq_lr = state['exp_avg_sq_lowrank']

                        exp_avg_lr.mul_(beta1).add_(grad_lowrank, alpha=1 - beta1)
                        exp_avg_sq_lr.mul_(beta2).addcmul_(grad_lowrank, grad_lowrank, value=1 - beta2)

                        # Compute update in low-rank space
                        denom_lr = exp_avg_sq_lr.sqrt().add_(group['eps'])
                        update_lowrank = exp_avg_lr / denom_lr

                        # Project back to full space
                        update_full = projector @ update_lowrank

                        # Apply update
                        p.add_(update_full.reshape(p.shape), alpha=-group['lr'])

        return loss


def create_lowrank_optimizer(model, lr=1e-3, weight_decay=0.0, rank=128):
    """
    Create a memory-efficient low-rank optimizer for the model.

    Separates embedding/output params (use standard Adam) from
    transformer params (use low-rank Adam).
    """
    # Separate small params (embeddings) from large params (transformers)
    embed_params = []
    lowrank_params = []

    for name, param in model.named_parameters():
        if 'wte' in name or 'wpe' in name or 'lm_head' in name:
            embed_params.append(param)
        else:
            lowrank_params.append(param)

    # Use standard Adam for embeddings, low-rank for rest
    optimizer = LowRankAdamW(
        [
            {'params': embed_params, 'lr': lr * 10, 'rank': 0},  # rank=0 means standard Adam
            {'params': lowrank_params, 'lr': lr, 'rank': rank},
        ],
        weight_decay=weight_decay,
    )

    return optimizer
