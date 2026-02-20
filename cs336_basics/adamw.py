from collections.abc import Callable
from typing import Optional
import torch
import math


class adamw(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):
        if lr <= 0:
            raise ValueError(f"lr must be positive: {lr}")
        if not 0 <= betas[0] < 1:
            raise ValueError(f"beta_1 must be in [0,1): {betas[0]}")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"beta_2 must be in [0,1): {betas[1]}")
        if eps <= 0:
            raise ValueError(f"eps must be positive: {eps}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m = state["m"]
                v = state["v"]
                t = state["t"] + 1

                # Update biased first moment estimate
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second moment estimate
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t

                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                # Decoupled weight decay (AdamW)
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                # Parameter update
                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-step_size)

                state["t"] = t

        return loss