import mlx.core as mx
import mlx.nn as nn

from components.attention import Attention
from components.feed_forward import FeedForward
from model_config import ModelConfig


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads: int = config.n_heads
        self.dim: int = config.dim
        self.attention = Attention(config=config)
        self.feed_forward = FeedForward(config=config)
        # Layer Normalization: https://arxiv.org/pdf/1607.06450#page=10
        # transformer blockのattentionの前に行うLayer Normalization
        self.attention_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        # transformer blockのfeed forwardの前に行うLayer Normalization
        self.ffn_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        r, cache = self.attention(
            x=self.attention_norm(x),
            mask=mask,
            cache=cache,
        )
        # Feed Forwardに入力する前に入力とattentionの結果を足し合わせる
        # https://arxiv.org/pdf/1706.03762#page=3
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out, cache
