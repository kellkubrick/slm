import torch
from torch import nn
from sublayers import ScaledDotProductAttention


class MaskMultiHeadAttention(nn.Module):
    """A class for implementing Multi-Head Attention block.

    For each sample this block makes projections for queries, keys and values and a final projection
        of concatenated SDPA heads as follows:
            Q_projection = Q * W_Q,
            K_projection = K * W_K,
            V_projection = V * W_V,

            MHA(Q, K, V) = SDPA(Q_projection, K_projection, V_projection) * W_O,

            where:
                - Q is of shape (M, d_model), K and V are of shape (N, d_model)
                - W_Q, W_K, W_V and W_O are trainable parameters of shape (d_model, d_model)

    Note that W_Q, W_K and W_V are stacked parameters for all heads (each of shape (d_model, d_k))
            assuming d_k = d_model // heads_num
    """

    def __init__(self, config):
        """Layers initialization."""
        super(MaskMultiHeadAttention, self).__init__()
        self.config = config

        d_k = config.d_model // config.heads_num
        self.scaled_dot_product_attention = ScaledDotProductAttention(d_k, config.heads_num)

        self.weights_q = nn.Linear(config.d_model, config.d_model, bias=config.attention_bias)
        self.weights_k = nn.Linear(config.d_model, config.d_model, bias=config.attention_bias)
        self.weights_v = nn.Linear(config.d_model, config.d_model, bias=config.attention_bias)
        self.weights_o = nn.Linear(config.d_model, config.d_model, bias=config.attention_bias)

        self._init_weights()

    def _init_weights(self):
        """Weights initialization."""
        torch.nn.init.xavier_uniform_(self.weights_q.weight)
        torch.nn.init.xavier_uniform_(self.weights_k.weight)
        torch.nn.init.xavier_uniform_(self.weights_v.weight)
        torch.nn.init.xavier_uniform_(self.weights_o.weight)
        if self.weights_q.bias is not None:
            nn.init.normal_(self.weights_q.bias, std=1e-6)
            nn.init.normal_(self.weights_k.bias, std=1e-6)
            nn.init.normal_(self.weights_v.bias, std=1e-6)
            nn.init.normal_(self.weights_o.bias, std=1e-6)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for the Multi-Head Attention block.

        Args:
            queries: Query tensor of shape (batch size, M, d_model).
            keys: Key tensor of shape (batch size, N, d_model).
            values: Value tensor of shape (batch size, N, d_model).
            mask: sequence mask with ones at positions that should be masked out

        Returns:
            Tensor of shape (batch size, M, d_model)
        """
        queries = self.weights_q(queries)
        keys = self.weights_k(keys)
        values = self.weights_v(values)
        bs = queries.size(0)
        mha = self.scaled_dot_product_attention(queries, keys, values).transpose(1, 2).contiguous().view(bs, -1,
                                                                                                         self.config.d_model)
        return self.weights_o(mha)


class LayerNorm(nn.Module):
    """A class for implementing Layer normalization."""

    def __init__(self, config):
        """Parameters initialization."""
        super(LayerNorm, self).__init__()

        self.gamma = nn.Parameter(torch.ones(config.d_model))
        self.beta = nn.Parameter(torch.zeros(config.d_model))
        self.eps = config.eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Layer normalization block.

        Args:
            inputs: tensor of shape (batch_size, sequence length, d_model)

        Returns:
            Tensor of shape (batch size, sequence length, d_model)
        """
        mean, variance = inputs.mean(-1, keepdim=True), inputs.var(-1, keepdim=True, unbiased=False)
        normalized_inputs = (inputs - mean) / torch.sqrt(variance + self.eps)
        output = self.gamma * normalized_inputs + self.beta
        return output


class FeedForward(nn.Module):
    """A class for implementing Feed Forward layer.

    For each sample this block makes two linear transformations defined as follows:
            FF(x) = max(0, x * W_1 + b_1) * W_2 + b_2,

            where:
                - x is an input tensor of shape (sequence length, d_model)
                - W_1 and b_1 are trainable parameters of first Linear layer
                        with output features num = d_ff and input features num = d_model
                - W_2 and b_2 are trainable parameters of first Linear layer
                        with output features num = d_model and input features num = d_ff
    """

    def __init__(self, config):
        """Layers initialization."""
        super(FeedForward, self).__init__()
        self.weights_1 = nn.Linear(config.d_model, config.d_ff)
        self.weights_2 = nn.Linear(config.d_ff, config.d_model)
        self.relu = getattr(nn, config.activation)()

        self._init_weights()

    def _init_weights(self):
        """Weights initialization."""
        nn.init.xavier_uniform_(self.weights_1.weight)
        nn.init.zeros_(self.weights_1.bias)
        nn.init.xavier_uniform_(self.weights_2.weight)
        nn.init.zeros_(self.weights_2.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Feed-Forward layer.

        Args:
            inputs: tensor of shape (batch size, sequence length, d_model).

        Returns:
            Tensor of shape (batch size, sequence length, d_model)
        """
        return self.weights_2(self.relu(self.weights_1(inputs)))
