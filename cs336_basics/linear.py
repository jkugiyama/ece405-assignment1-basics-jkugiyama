import math
import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()

        # Store W (not W^T): shape (out_features, in_features)
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        # Initialization (similar scale to common practice)
        std = 2.0 / (in_features + out_features)
        sigma = math.sqrt(std)
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., in_features)
        # W shape: (out_features, in_features)
        # output shape: (..., out_features)
        # Compute: x @ W.T  which gives (..., in_features) @ (in_features, out_features) = (..., out_features)
        return x @ self.W.T