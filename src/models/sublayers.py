import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    """A class for implementing Scaled Dot-Product Attention block for all heads at once.

    SDPA performs the following steps for each sample and each head:
        1. Takes projected Q (queries) of shape (M, d_model), K (keys) and V (values) each of shape (N, d_model) as inputs
        2. Calculates scaled attention scores as matrix multiplication of queries and transposed keys:
                attention_scores = Q * K^T / sqrt(d_k),
        3. Applies Softmax to the scaled attention scores row-wise (i.e. by the last dimension):
                weights = Softmax(attention_scores),

                where:
                    - attention_scores is a matrix of shape (M, N)
        4. Gets the whole block output by applying computed weights to the values projection as follows:
                SDPA(Q, K, V) = weights * V

    This block performs SDPA for all heads at once resulting in a tensor of shape (heads num, M, d_k)
            wrt the one sample.
    """

    def __init__(self, d_k: int = 64, heads_num: int = 8):
        """Layers initialization."""
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.heads_num = heads_num
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for the Scaled Dot-Product Attention block.

        Performs self-attention mechanism on queries, keys and values tensors for all heads at once.

        Args:
            queries: Query tensor of shape (batch size, M, d_model).
            keys: Key tensor of shape (batch size, N, d_model).
            values: Value tensor of shape (batch size, N, d_model).
            mask: sequence mask with ones at positions that should be masked out

        Returns:
            Tensor of shape (batch size, heads num, M, d_k) representing the values weighted with attention.
        """

        bs = queries.size(dim=0)

        nqueries = queries.view(bs, -1, self.heads_num, self.d_k).transpose(1, 2)
        nkeys = keys.view(bs, -1, self.heads_num, self.d_k).transpose(1, 2)
        nvalues = values.view(bs, -1, self.heads_num, self.d_k).transpose(1, 2)
        attention_scores = nqueries @ torch.transpose(nkeys, 2, 3) / ((self.d_k) ** 0.5)
        if mask is not None:
            attention_scores = torch.mul(attention_scores, mask)
        weights = self.softmax(attention_scores)
        sdpa = weights @ nvalues
        return sdpa
