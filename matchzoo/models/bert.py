"""An implementation of Bert Model."""
import typing

import torch
import torch.nn as nn
from pytorch_transformers import BertModel

from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine import hyper_spaces
from matchzoo.modules import BertModule


class Bert(BaseModel):
    """Bert Model."""

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params()
        params.add(Param(name='mode', value='bert-base-uncased',
                         desc="Pretrained Bert model."))
        params.add(Param(
            'dropout_rate', 0.0,
            hyper_space=hyper_spaces.quniform(
                low=0.0, high=0.8, q=0.01),
            desc="The dropout rate."
        ))
        return params

    def build(self):
        """Build model structure."""
        self.bert = BertModule(mode=self._params['mode'])
        self.dropout = nn.Dropout(p=self._params['dropout_rate'])
        if 'base' in self._params['mode']:
            dim = 768
        elif 'large' in self._params['mode']:
            dim = 1024
        self.out = self._make_output_layer(dim)

    def forward(self, inputs):
        """Forward."""

        input_left, input_right = inputs['text_left'], inputs['text_right']

        bert_output = self.bert(input_left, input_right)[1]

        out = self.out(self.dropout(bert_output))

        return out
