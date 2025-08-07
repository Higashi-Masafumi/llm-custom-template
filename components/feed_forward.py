import mlx.core as mx
import mlx.nn as nn

from model_config import ModelConfig


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        # nn.Linear(a, b) is a linear layer that maps a tensor of shape (a) to a tensor of shape (b)
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        # nn.silu is a silu activation function
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        # これは単なるReLUでは必要ないが、SiLUでは必要
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # nn.Linear(a, b) is a linear layer that maps a tensor of shape (a) to a tensor of shape (b)
        # nn.silu is a silu activation function
        # silu関数：https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html
        return self.w2(nn.silu(self.w1(x) * self.w3(x)))
