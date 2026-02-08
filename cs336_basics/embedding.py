import torch
from torch import Tensor
from jaxtyping import Int, Float


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        num_embeddings: vocab size
        embedding_dim: d_model
        """
        super().__init__()

        self.weight = torch.nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype,
            )
        )

        # Initialize weights
        torch.nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(
        self, token_ids: Int[Tensor, "batch context_length"]
    ) -> Float[Tensor, "batch context_length d_model"]:
        """
        Lookup embeddings for token IDs.
        """

        # One-hot encode tokens → (batch, context_length, vocab)
        one_hot = torch.nn.functional.one_hot(
            token_ids, num_classes=self.weight.shape[0]
        ).to(self.weight.dtype)

        # Multiply by embedding matrix → (batch, context_length, d_model)
        embeddings = torch.einsum("bsv,vd->bsd", one_hot, self.weight)

        return embeddings
