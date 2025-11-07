"""
GaLore optimizer - CORRECT implementation from the paper.
Key: Never materialize full P matrix, use on-the-fly projection.
"""

import torch


class GaLoreAdamW(torch.optim.Optimizer):
    """
    GaLore: Gradient Low-Rank Projection for memory-efficient training.

    Key insight from paper:
    - Don't store momentum/variance for full gradient (m×n)
    - Project gradient to low-rank: R = P^T @ G (r×n)
    - Store momentum/variance for R only (r×n instead of m×n)
    - Never materialize full P matrix!
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, rank=128, update_proj_gap=200):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       rank=rank, update_proj_gap=update_proj_gap)
        super().__init__(params, defaults)
        self.step_count = 0

    @torch.no_grad()
    def step(self):
        self.step_count += 1

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            rank = group['rank']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Weight decay (AdamW style)
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                # For 1D params or tiny params, use standard Adam
                if p.ndim == 1 or p.numel() < 10000:
                    if 'exp_avg' not in state:
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                    state['exp_avg'].mul_(beta1).add_(grad, alpha=1-beta1)
                    state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                    denom = state['exp_avg_sq'].sqrt().add_(group['eps'])
                    p.addcdiv_(state['exp_avg'], denom, value=-group['lr'])
                    continue

                # GaLore for 2D+ params
                # Reshape to 2D: (m, n) where m ≤ n
                original_shape = grad.shape
                if grad.ndim > 2:
                    grad_2d = grad.reshape(grad.shape[0], -1)
                else:
                    grad_2d = grad

                m, n = grad_2d.shape

                # Ensure m ≤ n (transpose if needed for efficiency)
                if m > n:
                    grad_2d = grad_2d.T
                    m, n = n, m
                    transposed = True
                else:
                    transposed = False

                # Update projection subspace every update_proj_gap steps
                if 'P' not in state or self.step_count % group['update_proj_gap'] == 0:
                    # Key trick: Use torch.svd_lowrank which DOESN'T materialize full SVD
                    # It uses randomized algorithms internally
                    actual_rank = min(rank, min(m, n) - 1)

                    if actual_rank < 1:
                        # Fallback to standard Adam
                        state['P'] = None
                        continue

                    # Compute left singular vectors ONLY (don't need V)
                    # This avoids full SVD
                    U, S, V = torch.svd_lowrank(grad_2d, q=actual_rank, niter=2)
                    state['P'] = U  # Store only P (m × r)
                    state['rank'] = actual_rank

                    # Initialize Adam stats in low-rank space
                    if 'exp_avg_lr' not in state:
                        state['exp_avg_lr'] = torch.zeros(actual_rank, n, device=p.device, dtype=p.dtype)
                        state['exp_avg_sq_lr'] = torch.zeros(actual_rank, n, device=p.device, dtype=p.dtype)

                # Project gradient to low-rank space
                P = state['P']
                if P is None:
                    continue

                R = P.T @ grad_2d  # (r × m) @ (m × n) = (r × n) - This is the key operation!

                # Update Adam statistics in low-rank space
                exp_avg_lr = state['exp_avg_lr']
                exp_avg_sq_lr = state['exp_avg_sq_lr']

                exp_avg_lr.mul_(beta1).add_(R, alpha=1-beta1)
                exp_avg_sq_lr.mul_(beta2).addcmul_(R, R, value=1-beta2)

                # Compute update in low-rank space
                denom = exp_avg_sq_lr.sqrt().add_(group['eps'])
                update_lr = exp_avg_lr / denom

                # Project back to full space
                update_2d = P @ update_lr  # (m × r) @ (r × n) = (m × n)

                # Un-transpose if needed
                if transposed:
                    update_2d = update_2d.T

                # Apply update
                p.add_(update_2d.reshape(original_shape), alpha=-group['lr'])

        return None
