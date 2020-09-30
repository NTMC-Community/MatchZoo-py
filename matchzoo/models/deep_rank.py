"""An implementation of DSSM, Deep Structured Semantic Model."""
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from matchzoo import preprocessors
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.base_preprocessor import BasePreprocessor
from matchzoo.dataloader import callbacks
from matchzoo.engine.base_callback import BaseCallback


class DeepRank(BaseModel):
    """
    Deep structured semantic model.

    Examples:
        >>> model = DeepRank()
        >>> embedding_matrix = np.ones((3000, 50), dtype=float)
        >>> term_weight_embedding_matrix = np.ones((3000, 1), dtype=float)
        >>> model.params['embedding'] = embedding_matrix
        >>> model.params['embedding_input_dim'] = embedding_matrix.shape[0]
        >>> model.params['embedding_output_dim'] = embedding_matrix.shape[1]
        >>> model.params['term_weight_embedding'] = term_weight_embedding_matrix
        >>> model.params['embedding_freeze'] = False
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(
            with_multi_layer_perceptron=False, with_embedding=True)
        params.add(Param(name='term_weight_embedding',
                         desc="Query term weight embedding matrix"))
        params.add(Param(name='reduce_out_dim', value=1,
                         desc="Output dimension of word embedding reduction"))
        params.add(Param(name='reduce_conv_kernel_size', value=1,
                         desc="Kernel size of convolution word embedding reduction"))
        params.add(Param(name='reduce_conv_stride', value=1,
                         desc="Stride of convolution word embedding reduction"))
        params.add(Param(name='reduce_conv_padding', value=0,
                         desc="Zero-padding added to both side of convolution \
                         word embedding reduction"))
        params.add(Param(name='half_window_size', value=5,
                         desc="Half of matching-window size, not including center term"))
        params.add(Param(name='encode_out_dim', value=4,
                         desc="Output dimension of encode"))
        params.add(Param(name='encode_conv_kernel_size', value=3,
                         desc="Kernel size of convolution encode"))
        params.add(Param(name='encode_conv_stride', value=1,
                         desc="Stride of convolution encode"))
        params.add(Param(name='encode_conv_padding', value=1,
                         desc="Zero-padding added to both side of convolution encode"))
        params.add(Param(name='encode_pool_out', value=1,
                         desc="Pooling size of global max-pooling for encoding matrix"))
        params.add(Param(name='encode_leaky', value=0.2,
                         desc="Relu leaky of encoder"))
        params.add(Param(name='gru_hidden_dim', value=3,
                         desc="Aggregation Network gru hidden dimension"))

        return params

    @classmethod
    def get_default_preprocessor(
        cls,
        truncated_mode: str = 'pre',
        truncated_length_left: int = None,
        truncated_length_right: int = None,
        filter_low_freq: float = 5,
        half_window_size: int = 5,
        padding_token_index: int = 0
    ) -> BasePreprocessor:
        """
        Model default preprocessor.

        The preprocessor's transform should produce a correctly shaped data
        pack that can be used for training.

        :return: Default preprocessor.
        """
        return preprocessors.DeepRankPreprocessor(
            truncated_mode=truncated_mode,
            truncated_length_left=truncated_length_left,
            truncated_length_right=truncated_length_right,
            filter_low_freq=filter_low_freq,
            half_window_size=half_window_size,
            padding_token_index=padding_token_index,
        )

    @classmethod
    def get_default_padding_callback(
            cls,
            pad_word_value: typing.Union[int, str] = 0,
            pad_word_mode: str = 'post',
    ) -> BaseCallback:
        """
        Only padding query.

        :return: Default padding callback.
        """
        return callbacks.BasicPadding(
            pad_word_value=pad_word_value,
            pad_word_mode=pad_word_mode
        )

    def build(self):
        """Build model structure."""
        # self.embedding = self._make_embedding_layer(
        #     freeze=self._params["embedding_freeze"],
        #     embedding=self._params["embedding"]
        # )
        self.embedding = self._make_default_embedding_layer()


        if self._params["term_weight_embedding"] is not None:
            term_weight_embedding = self._params["term_weight_embedding"]
        else:
            # self._params['embedding_input_dim'] = (
            #     self._params['embedding'].shape[0]
            # )
            # self._params['embedding_output_dim'] = 1
            term_weight_embedding = np.ones(
                (self._params["embedding_input_dim"], 1), dtype=float)
        self.term_weight_embedding = self._make_embedding_layer(
            freeze=self._params["embedding_freeze"],
            embedding=term_weight_embedding
        )

        self.query_reduce = nn.Conv1d(
            in_channels=self._params["embedding_output_dim"],
            out_channels=self._params["reduce_out_dim"],
            kernel_size=self._params["reduce_conv_kernel_size"],
            stride=self._params["reduce_conv_stride"],
            padding=self._params["reduce_conv_padding"]
        )

        self.doc_reduce = nn.Conv1d(
            in_channels=self._params["embedding_output_dim"],
            out_channels=self._params["reduce_out_dim"],
            kernel_size=self._params["reduce_conv_kernel_size"],
            stride=self._params["reduce_conv_stride"],
            padding=self._params["reduce_conv_padding"]
        )

        interact_tensor_channel = 2 * self._params["reduce_out_dim"] + 1
        self.encoding = nn.Sequential(
            nn.Conv2d(
                in_channels=interact_tensor_channel,
                out_channels=self._params["encode_out_dim"],
                kernel_size=self._params["encode_conv_kernel_size"],
                stride=self._params["encode_conv_stride"],
                padding=self._params["encode_conv_padding"],
            ),
            nn.AdaptiveAvgPool2d(output_size=self._params["encode_pool_out"]),
            nn.LeakyReLU(self._params["encode_leaky"])
        )

        gru_in_dim = \
            self._params["encode_out_dim"] * self._params["encode_pool_out"]**2 + 1
        self.gru = nn.GRU(
            input_size=gru_in_dim,
            hidden_size=self._params["gru_hidden_dim"],
            bidirectional=True
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.out = self._make_output_layer(
            in_features=2 * self._params["gru_hidden_dim"]
        )

    def forward(self, inputs):
        """Forward."""
        # Process left & right input.
        all_query: torch.LongTensor = inputs["text_left"]
        all_query_term_window_num: torch.LongTensor = inputs['term_window_num']
        all_query_len: torch.LongTensor = inputs["length_left"]

        all_window: torch.LongTensor = inputs['window_right']
        all_window_position: torch.LongTensor = inputs['window_position_right']
        all_query_window_num: torch.LongTensor = inputs['query_window_num']

        # all_query: [batch_size, max_q_seq_len]
        # all_query_term_window_num: [batch_size, max_q_seq_len]
        # all_query_len: [batch_size]

        # all_window: [batch_size, max_window_num, full_window_size]
        # all_window_position: [batch_size, max_window_num]
        # all_query_window_num: [batch_size]

        batch_size: int = all_query.shape[0]
        full_window_size: int = all_window.shape[2]
        device: torch.device = all_query.device

        ##############################
        # query embedding and reduce
        ##############################

        # all_query embedding
        # all_query: [batch_size, max_q_seq_len]
        #   -embedding-> [batch_size, max_q_seq_len, embed_dim]
        #   -permute(0, 2, 1)-> [batch_size, embed_dim, max_q_seq_len]
        all_query_embed = self.embedding(all_query).permute(0, 2, 1)

        # all_query reduce
        # all_query_embed:  [batch_size, embed_dim, max_q_seq_len]
        #   -all_query_reduce-> [batch_size, reduce_out_dim, max_q_seq_len]
        all_query_reduce = self.query_reduce(all_query_embed)

        ##############################
        # query term weight
        ##############################

        # all_query: [batch_size, max_q_seq_len]
        #   -term_weight_embedding-> [batch_size, max_q_seq_len, 1]
        #   -squeeze-> [batch_size, max_q_seq_len]
        all_query_term_weight = self.term_weight_embedding(all_query).squeeze(2)

        out = []
        for i in range(batch_size):
            one_query_seq_len = all_query_len[i].item()
            one_query_window_num = all_query_window_num[i].item()
            if one_query_window_num == 0:
                out.append(torch.zeros(self._params["gru_hidden_dim"] * 2, device=device))
                continue

            ##############################
            # window embedding and reduce
            ##############################

            # all_window: [batch_size, max_window_num, full_window_size]
            #   -[i]-> [one_query_window_num, full_window_size]
            one_query_windows = all_window[i][:one_query_window_num]

            # one_query_windows: [one_query_window_num, full_window_size]
            #   -embedding-> [one_query_window_num, full_window_size, embed_dim]
            #   -permute(0, 2, 1)-> [one_query_window_num, embed_dim, full_window_size]
            one_query_windows_embed = self.embedding(one_query_windows).permute(0, 2, 1)

            # one_query_windows_embed: [one_query_window_num, embed_dim, full_window_size]
            #   -doc_reduce-> [one_query_window_num, reduce_out_dim, full_window_size]
            one_query_windows_reduce = self.doc_reduce(one_query_windows_embed)

            ##############################
            # interaction signal
            ##############################

            # all_query_embed: [batch_size, embed_dim, max_q_seq_len]
            #   -[]-> [embed_dim, one_query_seq_len]
            one_query_embed = all_query_embed[i, :, :one_query_seq_len]

            # one_query_embed: [embed_dim, one_query_seq_len]
            # one_query_windows_embed: [one_query_window_num, embed_dim, full_window_size]
            #   -einsum("el,new->nlw")->
            #       [one_query_window_num, one_query_seq_len, full_window_size]
            #   -[]-> [one_query_window_num, 1, one_query_seq_len, full_window_size]
            interacition_signal = torch.einsum(
                "el,new->nlw", one_query_embed, one_query_windows_embed)[:, None, :, :]

            ##############################
            # encoding
            ##############################

            # all_query_reduce: [batch_size, reduce_out_dim, max_q_seq_len]
            #   -[]-> [reduce_out_dim, one_query_seq_len]
            one_query_reduce = all_query_reduce[i, :, :one_query_seq_len]

            # one_query_reduce: [reduce_dim, one_query_seq_len]
            #   -[]-> [1, reduce_dim, one_query_seq_len, 1]
            #   -expand->
            #     [one_query_window_num, reduce_dim, one_query_seq_len, full_window_size]
            one_query_reduce_expand = one_query_reduce[None, :, :, None] \
                .expand(one_query_window_num, -1, -1, full_window_size)

            # one_query_windows_reduce:
            #   [one_query_window_num, reduce_dim, full_window_size]
            #   -[]-> [one_query_window_num, reduce_dim, 1, full_window_size]
            #   -expand->
            #     [one_query_window_num, reduce_dim, one_query_seq_len, full_window_size]
            one_query_windows_reduce_expand = one_query_windows_reduce[:, :, None, :] \
                .expand(-1, -1, one_query_seq_len, -1)

            # one_query_reduce_expand:
            #   [one_query_window_num, reduce_dim, one_query_seq_len, full_window_size]
            # one_query_windows_reduce_expand:
            #   [one_query_window_num, reduce_dim, one_query_seq_len, full_window_size]
            # interacition_signal:
            #   [one_query_window_num, 1, one_query_seq_len, full_window_size]
            #   -stack->
            #     [one_query_window_num, 2*reduce_dim + 1,
            #       one_query_seq_len, full_window_size]
            encoder_input_tensor = torch.cat(
                [one_query_reduce_expand,
                 one_query_windows_reduce_expand,
                 interacition_signal],
                dim=1)

            # encoder_input_tensor:
            #       [one_query_window_num, 2*reduce_dim + 1,
            #        one_query_seq_len, full_window_size]
            #   -encoding->
            #     -Conv2d->
            #       [one_query_window_num, encode_out_dim,
            #        one_query_seq_len, full_window_size]
            #     -AdaptiveAvgPool2d+ReLU->
            #       [one_query_window_num, encode_out_dim,
            #        encode_pool_out, encode_pool_out]
            #   -flatten->
            #       [one_query_window_num, encode_out_dim * encode_pool_out^2]
            encoder_output_tensor = self.encoding(encoder_input_tensor).flatten(1)

            ##############################
            # position encoding and gru
            ##############################

            # all_window_position: [batch_size, max_window_num]
            #   -[]-> [one_query_window_num, 1]
            one_query_window_position = \
                all_window_position[i, :one_query_window_num, None]

            # one_query_window_position: [one_query_window_num, 1]
            #   --> [one_query_window_num, 1]
            window_position_encoding = (1. / (one_query_window_position + 1.)).float()

            # encoder_output_tensor:
            #   [one_query_window_num, encode_out_dim * encode_pool_out^2]
            # window_position_encoding:
            #   [one_query_window_num, 1]  gru_in_dim
            #   -cat->
            #       [one_query_window_num, gru_in_dim],
            #        gru_in_dim = encode_out_dim * encode_pool_out^2 + 1
            enc_and_pos = \
                torch.cat([encoder_output_tensor, window_position_encoding], dim=1)

            # all_query_term_window_num: [batch_size, max_q_seq_len]
            #   -[]-> [one_query_seq_len]
            #   -tolist-> list, len=one_query_seq_len
            one_query_term_window_num = \
                all_query_term_window_num[i, :one_query_seq_len].tolist()

            # enc_and_pos: [one_query_window_num, gru_in_dim]
            #   -split->
            #       tuple of tensor, len=one_query_seq_len,
            #           element i: tensor, [one_query_term_window_num[i], gru_in_dim]
            #   -pad_sequence->
            #       [max(one_query_term_window_num[i]), one_query_seq_len, gru_in_dim]
            one_query_pad_split_windows = \
                pad_sequence(enc_and_pos.split(one_query_term_window_num, dim=0))

            # one_query_pad_split_windows:
            #   [max(one_query_term_window_num[i]), one_query_seq_len, gru_in_dim]
            #   -gru->
            #     gru_out: [max(one_query_term_window_num[i]),
            #               one_query_seq_len, gru_hidden_dim * 2]
            #     gru_hidden: [num_layers, one_query_seq_len, gru_hidden_dim * 2]
            gru_out, gru_hidden = self.gru(one_query_pad_split_windows)
            # type: torch.Tensor, torch.Tensor

            ##############################
            # aggregate
            ##############################

            # gru_out:
            #       [max(one_query_term_window_num[i]),
            #        one_query_seq_len, gru_hidden_dim * 2]
            #   -permute(1,2,0)->
            #       [one_query_seq_len, gru_hidden_dim * 2,
            #        max(one_query_term_window_num[i])]
            gru_out = gru_out.permute(1, 2, 0)

            # gru_out:
            #       [one_query_seq_len, gru_hidden_dim * 2,
            #        max(one_query_term_window_num[i])]
            #   -pool-> [one_query_seq_len, gru_hidden_dim * 2, 1]
            #   -squeeze-> [one_query_seq_len, gru_hidden_dim * 2]
            pool_gru_out = self.pool(gru_out).squeeze(2)

            # all_query_term_weight: [batch_size, max_q_seq_len]
            #   -[]-> [one_query_seq_len]
            one_query_term_weight = all_query_term_weight[i, :one_query_seq_len]

            # pool_gru_out: [one_query_seq_len, gru_hidden_dim * 2]
            # one_query_term_weight: [one_query_seq_len]
            #   -einsum-> [gru_hidden_dim * 2]
            final_embed = torch.einsum("lh,l->h", pool_gru_out, one_query_term_weight)

            out.append(final_embed)
            pass

        ##############################
        # output score
        ##############################

        # out: list, len=batch_size, element i: tensor, [gru_hidden_dim * 2]
        #   -stack-> [batch_size, gru_hidden_dim * 2]
        out = torch.stack(out, dim=0)

        # out: [batch_size, gru_hidden_dim * 2]
        #   -out-> [batch_size, out_dim]
        out = self.out(out)
        return out
