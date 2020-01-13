"""An implementation of HCRN Model."""
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.param import Param
from matchzoo.engine.base_callback import BaseCallback
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine import hyper_spaces
from matchzoo.modules import Matching
from matchzoo.dataloader import callbacks
from matchzoo.utils import parse_activation


class HCRN(BaseModel):
    """
    Hermitian Co-Attention Networks for Text Matching in Asymmetrical Domains.

    Examples:
        >>> model = HCRN()
        >>> model.params['embedding_output_dim'] = 300
        >>> model.params['hidden_size'] = 300
        >>> model.params['pooling_type'] = 'alignment'
        >>> model.params['intra_attention'] = True
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_padding_callback(
        cls,
        fixed_length_left: int = None,
        fixed_length_right: int = None,
        pad_word_value: typing.Union[int, str] = 0,
        pad_word_mode: str = 'pre',
        with_ngram: bool = False,
        fixed_ngram_length: int = None,
        pad_ngram_value: typing.Union[int, str] = 0,
        pad_ngram_mode: str = 'pre'
    ) -> BaseCallback:
        """
        Model default padding callback.

        The pad_word_mode of this model is fixed to 'post'

        :return: Default padding callback.
        """
        return super().get_default_padding_callback(
            fixed_length_left=fixed_length_left,
            fixed_length_right=fixed_length_right,
            pad_word_value=pad_word_value,
            pad_word_mode='post',
            with_ngram=with_ngram,
            fixed_ngram_length=fixed_ngram_length,
            pad_ngram_value=pad_ngram_value,
            pad_ngram_mode=pad_ngram_mode)

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_embedding=True)
        params.add(Param(name='mask_value', value=0,
                         desc="The value to be masked from inputs."))
        params.add(Param(name='hidden_size', value=200,
                         desc="LSTM hidden size."))
        params.add(Param('pooling_type', value='extractive',
                         desc='Pooling type used to pool co attention. \
                               (Choices: alignment/extractive)',
                         validator=lambda x: x in ('alignment', 'extractive')))
        params.add(Param(name='intra_attention', value=True,
                         desc='whether use intra attention'))
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

        Hermitian Co-Attention Networks for Text Matching in Asymmetrical Domains.
        """
        self.embedding = self._make_default_embedding_layer()

        #  Word projection layer
        self.word_proj = nn.Sequential(
            nn.Linear(self.embedding.embedding_dim, self._params['hidden_size']),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            self._params['hidden_size'],
            self._params['hidden_size'],
            batch_first=True,
            bidirectional=True
        )

        self.hidden_size = self._params['hidden_size'] * 2
        if self._params['intra_attention']:
            self.intra_complex_proj = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU()
            )
            self.hidden_size *= 2

        # Complex part projection
        self.complex_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU()
        )

        # Interaction
        self.matching = Matching(matching_type='dot', normalize=False)

        self.out = self._make_output_layer(self.hidden_size * 2)

        self.dropout = nn.Dropout(p=self._params['dropout_rate'])

    def forward(self, inputs):
        """Forward."""

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   D = embedding size
        #   L = `input_left` sequence length
        #   R = `input_right` sequence length
        #   H = hidden size

        # Left input and right input.
        # shape = [B, L]
        # shape = [B, R]
        input_left, input_right = inputs['text_left'], inputs['text_right']

        # Left and right input mask matrix
        # shape = [B, L]
        # shape = [B, R]
        left_mask = (input_left == self._params['mask_value'])
        right_mask = (input_right == self._params['mask_value'])

        # Process left and right input.
        # shape = [B, L, D]
        # shape = [B, R, D]
        embed_left = self.embedding(input_left.long())
        embed_right = self.embedding(input_right.long())

        # Project left and right embedding into LSTM input
        # shape = [B, L, H]
        # shape = [B, R, H]
        proj_left = self.dropout(self.word_proj(embed_left))
        proj_right = self.dropout(self.word_proj(embed_right))

        # Pad sequence and send it into lstm, then unpack and restore the index
        # shape = [B, L, 2 * H]
        # shape = [B, R, 2 * H]
        length_left, length_right = inputs['length_left'], inputs['length_right']
        encode_left = self.dropout(
            self._forward_lstm_with_padded_sequence(proj_left, length_left))
        encode_right = self.dropout(
            self._forward_lstm_with_padded_sequence(proj_right, length_right))

        if self._params['intra_attention']:
            encode_left = self._forward_intra_attention(
                encode_left, left_mask)
            encode_right = self._forward_intra_attention(
                encode_right, right_mask)

        # Compute Hermitian Co Attention
        # shape = [B, L, R]
        real_matching = self.matching(encode_left, encode_right)
        complex_matching = self.matching(
            self.complex_proj(encode_left), self.complex_proj(encode_right))
        hermitian_co_attention = real_matching + complex_matching
        # use -inf may casue nan in sofmax calculation
        hermitian_co_attention.masked_fill_(left_mask.unsqueeze(-1), -1e16)
        hermitian_co_attention.masked_fill_(right_mask.unsqueeze(1), -1e16)

        # Pooling
        # shape = [B, 2 * H] or [B, 4 * H]
        # shape = [B, 2 * H] or [B, 4 * H]
        if self._params['pooling_type'] == 'alignment':
            # shape = [B, L, 2 * H] or [B, L, 4 * H]
            # shape = [B, R, 2 * H] or [B, L, 4 * H]
            pool_left = torch.bmm(F.softmax(hermitian_co_attention, dim=2), encode_right)
            pool_right = torch.bmm(
                F.softmax(hermitian_co_attention, dim=1).transpose(1, 2), encode_left)
            pool_left.masked_fill_(left_mask.unsqueeze(-1), 0)
            pool_right.masked_fill_(right_mask.unsqueeze(-1), 0)

            pool_left = torch.sum(pool_left, dim=1)
            pool_right = torch.sum(pool_right, dim=1)
        elif self._params['pooling_type'] == 'extractive':
            pool_left = torch.matmul(
                F.softmax(hermitian_co_attention.max(dim=2)[0], dim=-1)
                .unsqueeze(1), encode_left).squeeze(1)
            pool_right = torch.matmul(
                F.softmax(hermitian_co_attention.max(dim=1)[0], dim=-1)
                .unsqueeze(1), encode_right).squeeze(1)

        # Aggregataion
        # shape = [B, 4 * H] or [B, 8 * H]
        aggregate = torch.cat([pool_left, pool_right], dim=1)

        # shape = [B, *]
        out = self.out(self.dropout(aggregate))
        return out

    def _forward_intra_attention(self, input, mask):
        real_matching = self.matching(input, input)
        complex_input = self.intra_complex_proj(input)
        complex_matching = self.matching(complex_input, complex_input)
        hermitian_co_attention = real_matching + complex_matching
        hermitian_co_attention.masked_fill_(mask.unsqueeze(-1), -1e16)

        intra_attention = torch.bmm(F.softmax(hermitian_co_attention, dim=2), input)
        intra_attention.masked_fill_(mask.unsqueeze(-1), 0)
        return torch.cat([intra_attention, input], dim=-1)

    def _forward_lstm_with_padded_sequence(self, input, length):
        # pack the sequence in order to feed into lstm
        sorted_length, permutation_index = length.sort(0, descending=True)
        sorted_input = input.index_select(0, permutation_index)
        packed_input = pack_padded_sequence(
            sorted_input, sorted_length, batch_first=True)

        packed_output, _ = self.lstm(packed_input)
        unpacked_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # restore the original index
        index_range = torch.arange(0, len(length), device=length.device)
        _, reverse_mapping = permutation_index.sort(0, descending=False)
        restoration_indices = index_range.index_select(0, reverse_mapping)
        return unpacked_output.index_select(0, restoration_indices)
