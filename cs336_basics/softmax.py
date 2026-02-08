import torch

def softmax(tensor: torch.Tensor, *, dim: int):
    # subtract max for numerical stability
    shifted = tensor - tensor.max(dim=dim, keepdim=True).values
    exp_tensor = torch.exp(shifted)
    return exp_tensor / exp_tensor.sum(dim=dim, keepdim=True)
