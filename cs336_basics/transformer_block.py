from typing import Optional

import torch

from cs336_basics.rmsnorm import RmsNorm
from cs336_basics.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.swiglu import Swiglu


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

        self.ln1 = RmsNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(
            d_model, num_heads, max_seq_len, theta, device, dtype
        )
        self.ln2 = RmsNorm(d_model, device=device, dtype=dtype)
        self.ffn = Swiglu(d_model, d_ff, device, dtype)

    def forward(self, x):
        seq_len = x.shape[-2]
        token_positions = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.shape[:-1])
        )

        out = self.attn(self.ln1(x), token_positions) + x
        out = self.ffn(self.ln2(out)) + out
        return out
