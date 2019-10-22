"""Character embedding module."""
import typing

import torch
import torch.nn as nn


class CharacterEmbedding(nn.Module):
    """
    Character embedding module.

    :param char_embedding_input_dim: The input dimension of character embedding layer.
    :param char_embedding_output_dim: The output dimension of character embedding layer.
    :param char_conv_filters: The filter size of character convolution layer.
    :param char_conv_kernel_size: The kernel size of character convolution layer.

    Examples:
        >>> import torch
        >>> character_embedding = CharacterEmbedding()
        >>> x = torch.ones(10, 32, 16, dtype=torch.long)
        >>> x.shape
        torch.Size([10, 32, 16])
        >>> character_embedding(x).shape
        torch.Size([10, 32, 100])

    """

    def __init__(
        self,
        char_embedding_input_dim: int = 100,
        char_embedding_output_dim: int = 8,
        char_conv_filters: int = 100,
        char_conv_kernel_size: int = 5
    ):
        """Init."""
        super().__init__()
        self.char_embedding = nn.Embedding(
            num_embeddings=char_embedding_input_dim,
            embedding_dim=char_embedding_output_dim
        )
        self.conv = nn.Conv1d(
            in_channels=char_embedding_output_dim,
            out_channels=char_conv_filters,
            kernel_size=char_conv_kernel_size
        )

    def forward(self, x):
        """Forward."""
        embed_x = self.char_embedding(x)

        batch_size, seq_len, word_len, embed_dim = embed_x.shape

        embed_x = embed_x.contiguous().view(-1, word_len, embed_dim)

        embed_x = self.conv(embed_x.transpose(1, 2))
        embed_x = torch.max(embed_x, dim=-1)[0]

        embed_x = embed_x.view(batch_size, seq_len, -1)
        return embed_x
