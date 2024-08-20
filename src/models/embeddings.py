import math
import torch
from torch import nn


class Embedding(nn.Module):
    """A class for the embedding layer.

    These embeddings are often used to represent textual data inputs.
    """

    def __init__(self, vocabulary_size: int, d_model: int):
        """Embedding layer initialization.

        Args:
            vocabulary_size: data vocabulary size (i.e. the number of embeddings to store)
            d_model: embedding dimension
        """
        super().__init__()
        self.d_model = d_model
        self.embeddings = nn.Embedding(vocabulary_size, d_model, padding_idx=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Embedding layer.

        Args:
            inputs: tensor of shape (batch size, sequence length) representing raw inputs data

        Returns:
            Tensor of shape (batch_size, sequence length, d_model) representing the inputs embeddings
        """
        embeddings = self.embeddings(inputs)
        return embeddings


class PositionalEncoding(nn.Module):
    """A class for implementing Positional Encoding block.

    For each sample this block will add positional encoding (PE) matrix to the inputs. PE is defined as follows:
            PE(pos, 2 * i) = sin(pos / (10000 ^ (2 * i / d_model)))
            PE(pos, 2 * i + 1) = cos(pos / (10000 ^ (2 * i / d_model))),

            where:
                - pos is input sequence position number (from 0 to max sequence length - 1)
                - i is a counter which is used to represent even and odd embedding positions for each sequence element
                        (from 0 to d_model - 1 // 2)

    This block adds PE to each sample and applies Dropout to the resulting sum.
    """

    def __init__(self, max_sequence_length: int, d_model: int, dropout_rate: float):
        """Positional Encoding initialization.

        Args:
            max_sequence_length: maximum sequence length to expect
            d_model: embeddings dimension
            dropout_rate: Dropout probability
        """
        super(PositionalEncoding, self).__init__()
        positional_encoding = self.get_positional_encoding(max_sequence_length, d_model)
        self.register_buffer('positional_encoding', positional_encoding)
        self.dropout = nn.Dropout(dropout_rate)

    @staticmethod
    def get_positional_encoding(max_sequence_length: int, d_model: int) -> torch.Tensor:
        """Constructs PE matrix."""
        positions = torch.arange(max_sequence_length, dtype=torch.float32).unsqueeze(1)
        scaling_factor = torch.pow(torch.tensor(10000), torch.arange(0, d_model, 2) / d_model)

        positional_encoding = torch.zeros(max_sequence_length, d_model)
        positional_encoding[:, 0::2] = torch.sin(positions / scaling_factor)
        positional_encoding[:, 1::2] = torch.cos(positions / scaling_factor)
        return positional_encoding.unsqueeze(0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Positional Encoding block.

        Args:
            inputs: tensor of shape (batch size, sequence length, d_model)

        Returns:
            Tensor of shape (batch size, sequence length, d_model) representing "positional encoded" inputs
        """
        print(self.positional_encoding[:, :inputs.size(1)].size())
        return self.dropout(inputs + self.positional_encoding[:, :inputs.size(1)])

if __name__ == "__main__":
    embedder = Embedding(100, 512)(torch.tensor([[1, 2, 54, 24, 67]]))
    pe = PositionalEncoding(1024, 512, 0.1)(embedder)
    print(pe)