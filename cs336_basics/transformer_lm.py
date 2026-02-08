from typing import Optional

import torch
from torch import Tensor
from jaxtyping import Int

from cs336_basics import embedding, linear, rmsnorm

from .transformer_block import TransformerBlock

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

        # Token embeddings
        self.token_embeddings = embedding.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        # Transformer blocks
        self.layers = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        # Final normalization
        self.ln_final = rmsnorm.RmsNorm(
            d_model,
            device=device,
            dtype=dtype,
        )

        # Language modeling head
        self.lm_head = linear.Linear(
            d_model,
            vocab_size,
            device=device,
            dtype=dtype,
        )

        # Optional but common: weight tying
        # self.lm_head.weight = self.token_embeddings.weight

    def forward(
        self, x: Int[Tensor, "batch context_length"]
    ) -> Tensor:
        """
        Returns:
            logits: (batch, context_length, vocab_size)
        """
        out = self.token_embeddings(x)

        for layer in self.layers:
            out = layer(out)

        out = self.ln_final(out)
        logits = self.lm_head(out)

        return logits
