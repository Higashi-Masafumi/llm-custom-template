import mlx.core as mx
import mlx.nn as nn

from components.transformer_block import TransformerBlock
from model_config import ModelConfig


class BaseModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.vocab_size: int = config.vocab_size
        self.n_layers: int = config.n_layers
        assert self.vocab_size > 0, "vocab_size must be greater than 0"
        self.token_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = [TransformerBlock(config=config) for _ in range(config.n_layers)]
        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, self.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ) -> tuple[mx.array, list[mx.array]]:
        h = self.token_embeddings(inputs)
        # マスクの初期化
        mask = None
        # h.shape[1] is the sequence length
        if h.shape[1] > 1:
            # 自己回帰生成のためのマスク処理、i番目のトークンはi+1番目以降のトークンを見れない
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            # マスクのデータ型をhのデータ型に合わせる
            mask = mask.astype(h.dtype)

        # キャッシュの初期化
        if cache is None:
            cache = [None] * self.n_layers

        # 各層を順番に処理
        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask=mask, cache=cache[e])

        return self.output(self.norm(h)), cache
