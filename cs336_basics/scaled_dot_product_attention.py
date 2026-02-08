import math
import torch
from typing import Optional


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention.

    Args:
        query: Tensor of shape (..., seq_len, d_k)
        key:   Tensor of shape (..., seq_len, d_k)
        value: Tensor of shape (..., seq_len, d_v)
        mask:  Optional boolean tensor of shape (seq_len, seq_len)
               True = allowed, False = masked out

    Returns:
        Tensor of shape (..., seq_len, d_v)
    """
    d_k = query.size(-1)

    # (..., seq_len, seq_len)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # mask: (seq_len, seq_len) → broadcastable to scores
        scores = scores.masked_fill(~mask, float("-inf"))

    # (..., seq_len, seq_len)
    attn_probs = torch.softmax(scores, dim=-1)

    # (..., seq_len, d_v)
    output = torch.matmul(attn_probs, value)

    return output
