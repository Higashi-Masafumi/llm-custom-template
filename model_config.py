from pydantic import BaseModel, Field, StrictFloat, StrictInt


class ModelConfig(BaseModel):
    """Model configuration

    Args:
        dim: モデルのembeddingの埋め込み次元（モデルの総次元）
        n_layers: モデルのtransformer blockの総数
        head_dim: モデルのAttention Headの次元（一般的にはdim/n_heads）
        hidden_dim: モデルのFeed Forward Networkの次元（一般的には4*dim）
        n_heads: モデルのAttention Headの数
        n_kv_heads: モデルのKV Headの数（Grouped Query AttentionやMulti-Query Attentionを使用する場合に使用: https://arxiv.org/pdf/2406.09297v2#page=3）
        norm_eps: モデルの正規化層のepsilon値(Layer Normalizationのepsilon: https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
        vocab_size: モデルの語彙のサイズ
        rope_theta: モデルのRopeスケーリング係数のtheta値（Ropeのスケーリング係数: https://arxiv.org/pdf/1706.03762#page=6）
    """

    dim: StrictInt = Field(
        ...,
        title="Model dimension",
        description="The dimension of the model. This is the embedding dimension of the model.",
    )
    n_layers: StrictInt = Field(
        ...,
        title="Number of layers",
        description="The number of layers in the model.",
    )
    head_dim: StrictInt = Field(
        ...,
        title="Head dimension",
        description="The dimension of the head. This is the dimension of each head of the model.",
    )
    hidden_dim: StrictInt = Field(
        ...,
        title="Hidden dimension",
        description="The dimension of the hidden layer. This is the dimension of the hidden layer of the model.",
    )
    n_heads: StrictInt = Field(
        ...,
        title="Number of Attention heads",
        description="The number of attention heads in the model.",
    )
    n_kv_heads: StrictInt = Field(
        ...,
        title="Number of KV heads",
        description="The number of KV heads in the model.",
    )
    norm_eps: StrictFloat = Field(
        ...,
        title="Normalization epsilon",
        description="The epsilon value for the normalization layer.",
    )
    vocab_size: StrictInt = Field(
        ...,
        title="Vocabulary size",
        description="The size of the vocabulary of the model.",
    )
    rope_theta: StrictFloat = Field(
        10000,
        title="Rope theta",
        description="The theta value for the rope scaling factor.",
    )
    moe_config: "MixtureOfExpertsConfig" | None = Field(
        None,
        title="Mixture of Experts configuration",
        description="The configuration for the Mixture of Experts.",
    )


class MixtureOfExpertsConfig(BaseModel):
    """Mixture of Experts configuration

    Args:
        num_experts: モデルの専門家の数
        num_experts_per_token: モデルのトークンごとの専門家の数
    """

    num_experts: StrictInt = Field(
        ...,
        title="Number of experts",
        description="The number of experts in the model.",
    )
    num_experts_per_token: StrictInt = Field(
        ...,
        title="Number of experts per token",
        description="The number of experts per token in the model.",
    )
