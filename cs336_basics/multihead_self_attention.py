import math
import torch
import torch.nn as nn
from typing import Optional
from cs336_basics.linear import Linear


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # dk = dv

        # QKV projection in one matrix using custom Linear (no bias)
        self.qkv_proj = Linear(
            d_model, 3 * d_model, device=device, dtype=dtype
        )

        # Output projection using custom Linear (no bias)
        self.out_proj = Linear(
            d_model, d_model, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        returns: (batch_size, seq_len, d_model)
        """
        B, T, D = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3 * D)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for heads
        # (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # attn_scores: (B, num_heads, T, T)

        # Causal mask
        causal_mask = torch.tril(
            torch.ones(T, T, device=x.device, dtype=torch.bool)
        )
        attn_scores = attn_scores.masked_fill(
            ~causal_mask, float("-inf")
        )

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ v
        # (B, num_heads, T, head_dim)

        # Merge heads
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(B, T, D)
        )

        return self.out_proj(attn_output)