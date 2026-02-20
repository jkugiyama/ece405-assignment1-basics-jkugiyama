import torch
from jaxtyping import Float
from torch import Tensor

from cs336_basics import linear


class Swiglu(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.w1 = linear.Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = linear.Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = linear.Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(
        self, x: Float[Tensor, " ... d_model"]
    ) -> Float[Tensor, " ... d_model"]:
        
        x1 = self.w1(x)
        x3 = self.w3(x)

        # SiLU activation
        silu = x1 * torch.sigmoid(x1)

        # GLU-style gating
        gated = silu * x3

        return self.w2(gated)