import mlx.core as mx
import mlx.nn as nn

from model_config import ModelConfig


# pytorchの対応するドキュメント：
# https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads: int = config.n_heads
        self.n_kv_heads: int = config.n_kv_heads

        self.repeats: int = config.n_heads // config.n_kv_heads
        self.scale: float = config.head_dim**-0.5

        # nn.Linear(a, b) is a linear layer that maps a tensor of shape (a) to a tensor of shape (b)
        self.wq = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)
        self.rope = nn.RoPE(config.head_dim, traditional=True, base=config.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        B, L, D = (
            x.shape
        )  # batch_size, sequence_length, dimension（ここでのbatch_sizeは同時に処理するサンプルの数）

        # クエリ、キー、バリューを計算
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        # クエリとキーとバリューをAttentionの計算に適した形に変換
        # reshape前：（batch_size, sequence_length, head_dim * n_heads）
        # reshape後：（batch_size, sequence_length, n_heads, head_dim）
        # transpose後：（batch_size, n_heads, sequence_length, head_dim）
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # キャッシュがあれば、キャッシュを更新
        if cache is not None:
            key_cache, value_cache = cache
            # cacheのsequence_length分はオフセットしてRoPEを適用
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            # キーとバリューを結合(axis=2はシーケンス方向)
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # スケールドドット積Attentionを計算
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        # transpose後：（batch_size, n_heads, sequence_length, head_dim）
        # reshape後：（batch_size, sequence_length, n_heads * head_dim）
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        # 重みを掛けて出力
        return self.wo(output), (keys, values)
