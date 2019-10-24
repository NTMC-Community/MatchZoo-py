"""An implementation of aNMM Model."""
import typing

import torch
import torch.nn as nn

from matchzoo.dataloader import callbacks
from matchzoo.engine import hyper_spaces
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.param import Param
from matchzoo.engine.param_table import ParamTable
from matchzoo.modules import Attention, Matching
from matchzoo.utils import parse_activation


class aNMM(BaseModel):
    """
    aNMM: Ranking Short Answer Texts with Attention-Based Neural Matching Model.

    Examples:
        >>> model = aNMM()
        >>> model.params['embedding_output_dim'] = 300
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_embedding=True)
        params.add(Param(name='mask_value', value=0,
                         desc="The value to be masked from inputs."))
        params.add(Param(name='num_bins', value=200,
                         desc="Integer, number of bins."))
        params.add(Param(name='hidden_sizes', value=[100],
                         desc="Number of hidden size for each hidden layer"))
        params.add(Param(name='activation', value='relu',
                         desc="The activation function."))

        params.add(Param(
            'dropout_rate', 0.0,
            hyper_space=hyper_spaces.quniform(
                low=0.0, high=0.8, q=0.01),
            desc="The dropout rate."
        ))
        return params

    def build(self):
        """
        Build model structure.

        aNMM: Ranking Short Answer Texts with Attention-Based Neural Matching Model.
        """
        self.embedding = self._make_default_embedding_layer()

        # QA Matching
        self.matching = Matching(matching_type='dot', normalize=True)

        # Value-shared Weighting
        activation = parse_activation(self._params['activation'])
        in_hidden_size = [
            self._params['num_bins'],
            *self._params['hidden_sizes']
        ]
        out_hidden_size = [
            *self._params['hidden_sizes'],
            1
        ]

        hidden_layers = [
            nn.Sequential(
                nn.Linear(in_size, out_size),
                activation
            )
            for in_size, out_size, in zip(
                in_hidden_size,
                out_hidden_size
            )
        ]
        self.hidden_layers = nn.Sequential(*hidden_layers)

        # Query Attention
        self.q_attention = Attention(self._params['embedding_output_dim'])

        self.dropout = nn.Dropout(p=self._params['dropout_rate'])

        # Build output
        self.out = self._make_output_layer(1)

    def forward(self, inputs):
        """Forward."""
        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   D = embedding size
        #   L = `input_left` sequence length
        #   R = `input_right` sequence length
        #   BI = number of bins

        # Left input and right input
        # shape = [B, L]
        # shape = [B, R]
        input_left, input_right = inputs['text_left'], inputs['text_right']

        # Process left and right input
        # shape = [B, L, D]
        # shape = [B, R, D]
        embed_left = self.embedding(input_left.long())
        embed_right = self.embedding(input_right.long())

        # Left and right input mask matrix
        # shape = [B, L]
        # shape = [B, R]
        left_mask = (input_left == self._params['mask_value'])
        right_mask = (input_right == self._params['mask_value'])

        # Compute QA Matching matrix
        # shape = [B, L, R]
        qa_matching_matrix = self.matching(embed_left, embed_right)
        qa_matching_matrix.masked_fill_(right_mask.unsqueeze(1), float(0))

        # Bin QA Matching Matrix
        B, L = qa_matching_matrix.shape[0], qa_matching_matrix.shape[1]
        BI = self._params['num_bins']
        device = qa_matching_matrix.device
        qa_matching_matrix = qa_matching_matrix.view(-1)
        qa_matching_detach = qa_matching_matrix.detach()

        bin_indexes = torch.floor((qa_matching_detach + 1.) / 2 * (BI - 1.)).long()
        bin_indexes = bin_indexes.view(B * L, -1)

        index_offset = torch.arange(start=0, end=(B * L * BI), step=BI,
                                    device=device).long().unsqueeze(-1)
        bin_indexes += index_offset
        bin_indexes = bin_indexes.view(-1)

        # shape = [B, L, BI]
        bin_qa_matching = torch.zeros(B * L * BI, device=device)
        bin_qa_matching.index_add_(0, bin_indexes, qa_matching_matrix)
        bin_qa_matching = bin_qa_matching.view(B, L, -1)

        # Apply dropout
        bin_qa_matching = self.dropout(bin_qa_matching)

        # MLP hidden layers
        # shape = [B, L, 1]
        hiddens = self.hidden_layers(bin_qa_matching)

        # Query attention
        # shape = [B, L, 1]
        q_attention = self.q_attention(embed_left, left_mask).unsqueeze(-1)

        # shape = [B, 1]
        score = torch.sum(hiddens * q_attention, dim=1)
        # shape = [B, *]
        out = self.out(score)
        return out
