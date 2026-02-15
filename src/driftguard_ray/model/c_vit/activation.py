import torch
import torch.nn as nn

class DropPath(nn.Module):
    """Stochastic Depth / DropPath.

    During training, randomly zeroes whole samples (residual branches) with
    probability `drop_prob`. In eval mode it is a no-op.
    """
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # shape: (batch, 1, 1, ..., 1) to broadcast over remaining dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1) # create shape for broadcasting
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob) # generate binary mask
        return x.div(keep_prob) * random_tensor # scale output and apply mask