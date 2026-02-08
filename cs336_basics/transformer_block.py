from typing import Optional
import torch

from cs336_basics import attention, linear, rmsnorm


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: Optional[str | torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        # Pre-norm layers
        self.ln1 = rmsnorm.RmsNorm(d_model, device=device, dtype=dtype)
        self.ln2 = rmsnorm.RmsNorm(d_model, device=device, dtype=dtype)

        # Attention
        self.attn = attention.CausalMultiheadSelfAttention(
            d_model, num_heads, max_seq_len, theta, device, dtype
        )

        # Standard FFN (no SwiGLU)
        self.ffn_in = linear.Linear(d_model, d_ff, device=device, dtype=dtype)
        self.ffn_out = linear.Linear(d_ff, d_model, device=device, dtype=dtype)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        seq_len = x.shape[-2]

        token_positions = (
            torch.arange(seq_len, device=x.device)
            .unsqueeze(0)
            .expand(x.shape[:-1])
        )

        # ---- Attention block (pre-norm + residual) ----
        attn_out = self.attn(self.ln1(x), token_positions)
        x = x + attn_out

        # ---- Feed-forward block (pre-norm + residual) ----
        ffn_out = self.ffn_out(self.activation(self.ffn_in(self.ln2(x))))
        x = x + ffn_out

        return x
