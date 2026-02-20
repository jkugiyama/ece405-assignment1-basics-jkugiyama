from collections.abc import Iterable
import math
import torch


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
):
    # Convert to list in case it's a generator
    parameters = list(parameters)

    # Compute total L2 norm of all gradients
    total_norm_sq = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        total_norm_sq += p.grad.data.pow(2).sum().item()

    total_norm = math.sqrt(total_norm_sq)

    # If norm exceeds max_l2_norm, scale gradients
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(scale)