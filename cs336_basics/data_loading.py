import numpy as np
import numpy.typing as npt
import torch


def data_loading(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        inputs  - (batch_size, context_length)
        targets - (batch_size, context_length)
    """

    # Convert dataset to torch once
    data = torch.as_tensor(dataset, dtype=torch.long)

    # Sample random starting positions
    max_start = len(data) - context_length - 1
    starts = torch.randint(0, max_start + 1, (batch_size,))

    # Build input and target tensors
    inputs = torch.stack([
        data[start : start + context_length]
        for start in starts
    ])

    targets = torch.stack([
        data[start + 1 : start + 1 + context_length]
        for start in starts
    ])

    return inputs.to(device), targets.to(device)