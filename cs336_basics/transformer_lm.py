from typing import Optional

import torch
from torch import Tensor
from jaxtyping import Int

from cs336_basics.embedding import Embedding 
from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RmsNorm
from cs336_basics.transformer_block import TransformerBlock

class TransformerLm(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        device: Optional[str | torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype
        )
        self.layers = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    max_seq_len=context_length,
                    theta=theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RmsNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: Int[Tensor, "batch context_length"]):
        out = self.token_embeddings(x)
        for layer in self.layers:
            out = layer(out)
        out = self.lm_head(self.ln_final(out))
        return out
    