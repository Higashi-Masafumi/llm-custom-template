import mlx.core as mx
import mlx.nn as nn

from components.feed_forward import FeedForward
from model_config import ModelConfig


class MoeFeedForward(nn.Module):
    """Mixture of Experts Feed Forward

    Args:
        config: モデルの設定

    Note:
        - Mixture of Expertsの詳しい原理については次の論文を参照：https://www.alphaxiv.org/ja/overview/1701.06538v1
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert (
            config.moe_config is not None
        ), "Mixture of Experts configuration is required"
        self.num_experts = config.moe_config.num_experts
        self.num_experts_per_token = config.moe_config.num_experts_per_token
        self.experts = [FeedForward(config=config) for _ in range(self.num_experts)]
        # ゲートは、モデルの入力を専門家に分けるための線形層
        self.gate = nn.Linear(config.dim, self.num_experts, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        num_experts_per_token = self.num_experts_per_token
        B, L, D = x.shape
        x = x.reshape(-1, D)  # バッチサイズとシーケンス長を平坦化(B*L, D)
        gates = self.gate(x)

        # ゲートの値が大きい順に専門家を選択 トークンごとにnum_experts_per_token個の専門家を選択
        indices_of_experts = mx.argpartition(
            -gates, kth=num_experts_per_token - 1, axis=-1
        )[:, :num_experts_per_token]
        # 選択された専門家のゲートの値を正規化
        # 具体的には行ごとにsoftmaxを適用して、選択された専門家のゲートの値を正規化
        scores = mx.softmax(
            mx.take_along_axis(gates, indices_of_experts, axis=-1).astype(mx.float32),
            axis=-1,
        ).astype(gates.dtype)

        outputs = []
        for x_token, score_of_token, indices_of_experts_of_token in zip(
            x, scores, indices_of_experts.tolist()
        ):  # type: ignore
            # 選択された専門家の出力を結合（axis=-1で結合しているので、バッチ次元は保持される）
            yt = mx.concatenate(
                [
                    self.experts[e](x_token)[:, None]  # トークンが列ベクトルになるようにスライス
                    for e in indices_of_experts_of_token
                ],
                axis=-1,
            )
            # それそれのexpertの出力に対して、それぞれのexpertのgateスコアをかけて合計を計算する
            yt = (yt * score_of_token).sum(axis=-1)
            # あとで列方向で結合するために、行次元を追加
            # (B*L, D) -> (B*L, 1, D)
            outputs.append(yt[None, :])
        # 行方向で結合する
        y = mx.concatenate(outputs, axis=0)
        # 元のバッチサイズとシーケンス長に戻す
        # (B*L, 1, D) -> (B, L, D)
        return y.reshape(B, L, D)
