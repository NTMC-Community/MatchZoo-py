"""An implementation of BiMPM Model."""
import typing

import torch
import torch.nn as nn
from torch.nn import functional as F

from matchzoo.engine import hyper_spaces
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel


class BiMPM(BaseModel):
    """
    BiMPM Model.

    Reference:
    - https://github.com/galsang/BIMPM-pytorch/blob/master/model/BIMPM.py

    Examples:
        >>> model = BiMPM()
        >>> model.params['num_perspective'] = 4
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(
            with_embedding=True,
            with_multi_layer_perceptron=False
        )
        params.add(Param(name='mask_value', value=0,
                         desc="The value to be masked from inputs."))
        params.add(Param(name='dropout', value=0.2,
                         desc="Dropout rate."))
        params.add(Param(name='hidden_size', value=100,
                         hyper_space=hyper_spaces.quniform(
                             low=100, high=300, q=100),
                         desc="Hidden size."))

        # BiMPM parameters
        params.add(Param(name='num_perspective', value=20,
                         hyper_space=hyper_spaces.quniform(
                             low=20, high=100, q=20),
                         desc='num_perspective'))

        return params

    def build(self):
        """Make function layers."""

        self.embedding = self._make_default_embedding_layer()

        # Context Representation Layer
        self.context_LSTM = nn.LSTM(
            input_size=self._params['embedding_output_dim'],
            hidden_size=self._params['hidden_size'],
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # Matching Layer
        for i in range(1, 9):
            setattr(self, f'mp_w{i}',
                    nn.Parameter(torch.rand(self._params['num_perspective'],
                                            self._params['hidden_size'])))

        # Aggregation Layer
        self.aggregation_LSTM = nn.LSTM(
            input_size=self._params['num_perspective'] * 8,
            hidden_size=self._params['hidden_size'],
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # Prediction Layer
        self.pred_fc1 = nn.Linear(
            self._params['hidden_size'] * 4,
            self._params['hidden_size'] * 2)
        self.pred_fc2 = self._make_output_layer(
            self._params['hidden_size'] * 2)

        # parameters
        self.reset_parameters()

    def forward(self, inputs):
        """Forward."""
        # Word Representation Layer
        # (batch, seq_len) -> (batch, seq_len, word_dim)

        # [B, L], [B, R]
        p, h = inputs['text_left'].long(), inputs['text_right'].long()

        # [B, L, D]
        # [B, R, D]
        p = self.embedding(p)
        h = self.embedding(h)

        p = self.dropout(p)
        h = self.dropout(h)

        # Context Representation Layer
        # (batch, seq_len, hidden_size * 2)
        con_p, _ = self.context_LSTM(p)
        con_h, _ = self.context_LSTM(h)

        con_p = self.dropout(con_p)
        con_h = self.dropout(con_h)

        # (batch, seq_len, hidden_size)
        con_p_fw, con_p_bw = torch.split(con_p,
                                         self._params['hidden_size'],
                                         dim=-1)
        con_h_fw, con_h_bw = torch.split(con_h,
                                         self._params['hidden_size'],
                                         dim=-1)

        # 1. Full-Matching

        # (batch, seq_len, hidden_size), (batch, hidden_size)
        #   -> (batch, seq_len, l)
        mv_p_full_fw = mp_matching_func(
            con_p_fw, con_h_fw[:, -1, :], self.mp_w1)
        mv_p_full_bw = mp_matching_func(
            con_p_bw, con_h_bw[:, 0, :], self.mp_w2)
        mv_h_full_fw = mp_matching_func(
            con_h_fw, con_p_fw[:, -1, :], self.mp_w1)
        mv_h_full_bw = mp_matching_func(
            con_h_bw, con_p_bw[:, 0, :], self.mp_w2)

        # 2. Maxpooling-Matching

        # (batch, seq_len1, seq_len2, l)
        mv_max_fw = mp_matching_func_pairwise(con_p_fw, con_h_fw, self.mp_w3)
        mv_max_bw = mp_matching_func_pairwise(con_p_bw, con_h_bw, self.mp_w4)

        # (batch, seq_len, l)
        mv_p_max_fw, _ = mv_max_fw.max(dim=2)
        mv_p_max_bw, _ = mv_max_bw.max(dim=2)
        mv_h_max_fw, _ = mv_max_fw.max(dim=1)
        mv_h_max_bw, _ = mv_max_bw.max(dim=1)

        # 3. Attentive-Matching

        # (batch, seq_len1, seq_len2)
        att_fw = attention(con_p_fw, con_h_fw)
        att_bw = attention(con_p_bw, con_h_bw)

        # (batch, seq_len2, hidden_size) -> (batch, 1, seq_len2, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # output:  -> (batch, seq_len1, seq_len2, hidden_size)
        att_h_fw = con_h_fw.unsqueeze(1) * att_fw.unsqueeze(3)
        att_h_bw = con_h_bw.unsqueeze(1) * att_bw.unsqueeze(3)
        # (batch, seq_len1, hidden_size) -> (batch, seq_len1, 1, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # output:  -> (batch, seq_len1, seq_len2, hidden_size)
        att_p_fw = con_p_fw.unsqueeze(2) * att_fw.unsqueeze(3)
        att_p_bw = con_p_bw.unsqueeze(2) * att_bw.unsqueeze(3)

        # (batch, seq_len1, hidden_size) / (batch, seq_len1, 1)
        # output:  -> (batch, seq_len1, hidden_size)
        att_mean_h_fw = div_with_small_value(
            att_h_fw.sum(dim=2),
            att_fw.sum(dim=2, keepdim=True))
        att_mean_h_bw = div_with_small_value(
            att_h_bw.sum(dim=2),
            att_bw.sum(dim=2, keepdim=True))
        # (batch, seq_len2, hidden_size) / (batch, seq_len2, 1)
        # output:  -> (batch, seq_len2, hidden_size)
        att_mean_p_fw = div_with_small_value(
            att_p_fw.sum(dim=1),
            att_fw.sum(dim=1, keepdim=True).permute(0, 2, 1))
        att_mean_p_bw = div_with_small_value(
            att_p_bw.sum(dim=1),
            att_bw.sum(dim=1, keepdim=True).permute(0, 2, 1))

        # (batch, seq_len, l)
        mv_p_att_mean_fw = mp_matching_func(
            con_p_fw, att_mean_h_fw, self.mp_w5)
        mv_p_att_mean_bw = mp_matching_func(
            con_p_bw, att_mean_h_bw, self.mp_w6)
        mv_h_att_mean_fw = mp_matching_func(
            con_h_fw, att_mean_p_fw, self.mp_w5)
        mv_h_att_mean_bw = mp_matching_func(
            con_h_bw, att_mean_p_bw, self.mp_w6)

        # 4. Max-Attentive-Matching

        # (batch, seq_len1, hidden_size)
        att_max_h_fw, _ = att_h_fw.max(dim=2)
        att_max_h_bw, _ = att_h_bw.max(dim=2)
        # (batch, seq_len2, hidden_size)
        att_max_p_fw, _ = att_p_fw.max(dim=1)
        att_max_p_bw, _ = att_p_bw.max(dim=1)

        # (batch, seq_len, l)
        mv_p_att_max_fw = mp_matching_func(con_p_fw, att_max_h_fw, self.mp_w7)
        mv_p_att_max_bw = mp_matching_func(con_p_bw, att_max_h_bw, self.mp_w8)
        mv_h_att_max_fw = mp_matching_func(con_h_fw, att_max_p_fw, self.mp_w7)
        mv_h_att_max_bw = mp_matching_func(con_h_bw, att_max_p_bw, self.mp_w8)

        # (batch, seq_len, l * 8)
        mv_p = torch.cat(
            [mv_p_full_fw, mv_p_max_fw, mv_p_att_mean_fw, mv_p_att_max_fw,
             mv_p_full_bw, mv_p_max_bw, mv_p_att_mean_bw, mv_p_att_max_bw],
            dim=2)
        mv_h = torch.cat(
            [mv_h_full_fw, mv_h_max_fw, mv_h_att_mean_fw, mv_h_att_max_fw,
             mv_h_full_bw, mv_h_max_bw, mv_h_att_mean_bw, mv_h_att_max_bw],
            dim=2)

        mv_p = self.dropout(mv_p)
        mv_h = self.dropout(mv_h)

        # Aggregation Layer
        # (batch, seq_len, l * 8) -> (2, batch, hidden_size)
        _, (agg_p_last, _) = self.aggregation_LSTM(mv_p)
        _, (agg_h_last, _) = self.aggregation_LSTM(mv_h)

        # 2 * (2, batch, hidden_size) -> 2 * (batch, hidden_size * 2)
        #   -> (batch, hidden_size * 4)
        x = torch.cat(
            [agg_p_last.permute(1, 0, 2).contiguous().view(
                -1, self._params['hidden_size'] * 2),
                agg_h_last.permute(1, 0, 2).contiguous().view(
                    -1, self._params['hidden_size'] * 2)],
            dim=1)
        x = self.dropout(x)

        # Prediction Layer
        x = torch.tanh(self.pred_fc1(x))
        x = self.dropout(x)
        x = self.pred_fc2(x)

        return x

    def reset_parameters(self):
        """Init Parameters."""

        # Word Representation Layer

        # <unk> vectors is randomly initialized
        nn.init.uniform_(self.embedding.weight.data[0], -0.1, 0.1)

        # Context Representation Layer
        nn.init.kaiming_normal_(self.context_LSTM.weight_ih_l0)
        nn.init.constant_(self.context_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.context_LSTM.weight_hh_l0)
        nn.init.constant_(self.context_LSTM.bias_hh_l0, val=0)

        nn.init.kaiming_normal_(self.context_LSTM.weight_ih_l0_reverse)
        nn.init.constant_(self.context_LSTM.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal_(self.context_LSTM.weight_hh_l0_reverse)
        nn.init.constant_(self.context_LSTM.bias_hh_l0_reverse, val=0)

        # Matching Layer
        for i in range(1, 9):
            w = getattr(self, f'mp_w{i}')
            nn.init.kaiming_normal_(w)

        # Aggregation Layer
        nn.init.kaiming_normal_(self.aggregation_LSTM.weight_ih_l0)
        nn.init.constant_(self.aggregation_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.aggregation_LSTM.weight_hh_l0)
        nn.init.constant_(self.aggregation_LSTM.bias_hh_l0, val=0)

        nn.init.kaiming_normal_(self.aggregation_LSTM.weight_ih_l0_reverse)
        nn.init.constant_(self.aggregation_LSTM.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal_(self.aggregation_LSTM.weight_hh_l0_reverse)
        nn.init.constant_(self.aggregation_LSTM.bias_hh_l0_reverse, val=0)

        # Prediction Layer ----
        nn.init.uniform_(self.pred_fc1.weight, -0.005, 0.005)
        nn.init.constant_(self.pred_fc1.bias, val=0)

        # nn.init.uniform(self.pred_fc2.weight, -0.005, 0.005)
        # nn.init.constant(self.pred_fc2.bias, val=0)

    def dropout(self, v):
        """Dropout Layer."""
        return F.dropout(v, p=self._params['dropout'], training=self.training)


def mp_matching_func(v1, v2, w):
    """
    Basic mp_matching_func.

    :param v1: (batch, seq_len, hidden_size)
    :param v2: (batch, seq_len, hidden_size) or (batch, hidden_size)
    :param w: (num_psp, hidden_size)
    :return: (batch, num_psp)
    """

    seq_len = v1.size(1)
    num_psp = w.size(0)

    # (1, 1, hidden_size, l)
    w = w.transpose(1, 0).unsqueeze(0).unsqueeze(0)
    # (batch, seq_len, hidden_size, l)
    v1 = w * torch.stack([v1] * num_psp, dim=3)
    if len(v2.size()) == 3:
        v2 = w * torch.stack([v2] * num_psp, dim=3)
    else:
        v2 = w * torch.stack(
            [torch.stack([v2] * seq_len, dim=1)] * num_psp, dim=3)

    m = F.cosine_similarity(v1, v2, dim=2)

    return m


def mp_matching_func_pairwise(v1, v2, w):
    """
    Basic mp_matching_func_pairwise.

    :param v1: (batch, seq_len1, hidden_size)
    :param v2: (batch, seq_len2, hidden_size)
    :param w: (num_psp, hidden_size)
    :param num_psp
    :return: (batch, num_psp, seq_len1, seq_len2)
    """

    num_psp = w.size(0)

    # (1, l, 1, hidden_size)
    w = w.unsqueeze(0).unsqueeze(2)
    # (batch, l, seq_len, hidden_size)
    v1, v2 = (w * torch.stack([v1] * num_psp, dim=1),
              w * torch.stack([v2] * num_psp, dim=1))
    # (batch, l, seq_len, hidden_size->1)
    v1_norm = v1.norm(p=2, dim=3, keepdim=True)
    v2_norm = v2.norm(p=2, dim=3, keepdim=True)

    # (batch, l, seq_len1, seq_len2)
    n = torch.matmul(v1, v2.transpose(2, 3))
    d = v1_norm * v2_norm.transpose(2, 3)

    # (batch, seq_len1, seq_len2, l)
    m = div_with_small_value(n, d).permute(0, 2, 3, 1)

    return m


def attention(v1, v2):
    """
    Attention.

    :param v1: (batch, seq_len1, hidden_size)
    :param v2: (batch, seq_len2, hidden_size)
    :return: (batch, seq_len1, seq_len2)
    """

    # (batch, seq_len1, 1)
    v1_norm = v1.norm(p=2, dim=2, keepdim=True)
    # (batch, 1, seq_len2)
    v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)

    # (batch, seq_len1, seq_len2)
    a = torch.bmm(v1, v2.permute(0, 2, 1))
    d = v1_norm * v2_norm

    return div_with_small_value(a, d)


def div_with_small_value(n, d, eps=1e-8):
    """
    Small values are replaced by 1e-8 to prevent it from exploding.

    :param n: tensor
    :param d: tensor
    :return: n/d: tensor
    """
    d = d * (d > eps).float() + eps * (d <= eps).float()
    return n / d
