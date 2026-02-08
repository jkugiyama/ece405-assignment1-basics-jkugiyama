import torch
from torch import Tensor
from jaxtyping import Float, Int


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        assert d_k % 2 == 0, "d_k must be even for RoPE"

        positions = torch.arange(max_seq_len, device=device).float()  # (L,)
        dim_indices = torch.arange(0, d_k, 2, device=device).float()  # (d_k/2,)

        freqs = 1.0 / (theta ** (dim_indices / d_k))                  # (d_k/2,)
        angles = positions[:, None] * freqs[None, :]                 # (L, d_k/2)

        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"],
    ) -> Tensor:
        """
        Apply rotary positional embeddings to x.
        """

        seq_len = x.shape[-2]
        d_k = x.shape[-1]

        # Split into even/odd
        x_even = x[..., 0::2]  # (..., seq_len, d_k/2)
        x_odd = x[..., 1::2]   # (..., seq_len, d_k/2)

        # Gather cos/sin using token positions
        cos = self.cos_cached.index_select(0, token_positions.reshape(-1))
        sin = self.sin_cached.index_select(0, token_positions.reshape(-1))

        cos = cos.view(*token_positions.shape, -1)
        sin = sin.view(*token_positions.shape, -1)

        # Apply rotation
        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot = x_even * sin + x_odd * cos

        # Interleave even/odd back together
        out = torch.empty_like(x)
        out[..., 0::2] = x_even_rot
        out[..., 1::2] = x_odd_rot

        return out
