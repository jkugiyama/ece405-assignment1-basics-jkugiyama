from typing import Optional
import torch


class RmsNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: Optional[str | torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original dtype
        in_dtype = x.dtype

        # Upcast to float32 for numerical stability
        x = x.to(torch.float32)

        # RMS computation over the last dimension (d_model)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        # Normalize and scale
        x = x / rms
        x = x * self.weight

        # Downcast back to original dtype
        return x.to(in_dtype)
