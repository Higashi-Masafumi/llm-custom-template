import mlx.core as mx
import mlx.nn as nn

from components.transformer_block import MoeTransformerBlock
from model_config import ModelConfig
from models.model import BaseModel


class MoEModel(BaseModel):
    """
    MoEモデルの基底クラス
    モデルの構造を定義する
    Args:
        config: モデルの設定(ModelConfig)
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.layers = [
            MoeTransformerBlock(config=config) for _ in range(config.n_layers)
        ]

    def __call__(self, inputs: mx.array, cache=None) -> tuple[mx.array, list[mx.array]]:
        h = self.token_embeddings(inputs)

        mask = None
        # T: トークンの数
        T = h.shape[1]
        # マスクの作成
        if T > 1:
            # 未来のトークンをマスクする
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(x=h, mask=mask, cache=cache[e])

        return self.outpu(self.norm(h[:, T - 1 : T, :])), cache
