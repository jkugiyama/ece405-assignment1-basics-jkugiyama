import torch
from torch import Tensor
from jaxtyping import Float, Int
from typing import Optional
from cs336_basics.linear import Linear
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention 
from cs336_basics.rope import RotaryPositionalEmbedding


class MultiheadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 0,
        theta: float = 0.0,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.d_k = self.d_v = d_model // n_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if theta > 0 and max_seq_len > 0:
            self.rope = RotaryPositionalEmbedding(
                theta, self.d_k, max_seq_len, device=device
            )
        else:
            if theta > 0 or max_seq_len > 0:
                raise Exception(
                    "If either theta or max_seq_len are gte zero, both must be."
                )
            self.rope = None

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_in"],
        token_positions: Optional[Int[Tensor, " ... sequence_length"]] = None,
    ) -> Float[Tensor, "... d_model"]:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # unflatten: split d_model across the heads
        # transpose: move the head dim into a batch position
        q_heads = q.unflatten(-1, (self.n_heads, self.d_k)).transpose(-3, -2)
        k_heads = k.unflatten(-1, (self.n_heads, self.d_k)).transpose(-3, -2)
        v_heads = v.unflatten(-1, (self.n_heads, self.d_v)).transpose(-3, -2)

        if token_positions is not None:
            if not self.rope:
                raise Exception("rope must have been initialized for token_positions")
            # Insert a singleton dimension where the head dim is so that RoPE ops are compatible.
            head_token_positions = token_positions.unsqueeze(-2)
            q_heads = self.rope(q_heads, head_token_positions)
            k_heads = self.rope(k_heads, head_token_positions)

        seq_len = x.shape[-2]
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )

        attention_heads = scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)

        attention = attention_heads.transpose(-2, -3).flatten(-2, -1)

        o = self.output_proj(attention)
        return o