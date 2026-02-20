import torch
from torch import Tensor

from jaxtyping import Float, Int

def cross_entropy(
    inputs: Float[Tensor, "batch_size vocab_size"],
    targets: Int[Tensor, "batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.
    """

    # Numerical stability trick (subtract max per row)
    maxes, _ = inputs.max(dim=-1, keepdim=True)
    shifted = inputs - maxes

    # Log-softmax
    log_softmax = shifted - shifted.exp().sum(dim=-1, keepdim=True).log()

    # Select correct class log-probabilities
    selected_log_probs = log_softmax[torch.arange(inputs.size(0)), targets]

    # Return average negative log likelihood
    return -selected_log_probs.mean()