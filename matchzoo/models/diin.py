"""An implementation of DIIN Model."""
import typing

import torch
import torch.nn as nn

from matchzoo import preprocessors
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.base_callback import BaseCallback
from matchzoo.engine.base_preprocessor import BasePreprocessor
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine import hyper_spaces
from matchzoo.dataloader import callbacks
from matchzoo.modules import CharacterEmbedding, SemanticComposite, Matching, DenseNet


class DIIN(BaseModel):
    """
    DIIN model.

    Examples:
        >>> model = DIIN()
        >>> model.params['embedding_input_dim'] = 10000
        >>> model.params['embedding_output_dim'] = 300
        >>> model.params['mask_value'] = 0
        >>> model.params['char_embedding_input_dim'] = 100
        >>> model.params['char_embedding_output_dim'] = 8
        >>> model.params['char_conv_filters'] = 100
        >>> model.params['char_conv_kernel_size'] = 5
        >>> model.params['first_scale_down_ratio'] = 0.3
        >>> model.params['nb_dense_blocks'] = 3
        >>> model.params['layers_per_dense_block'] = 8
        >>> model.params['growth_rate'] = 20
        >>> model.params['transition_scale_down_ratio'] = 0.5
        >>> model.params['conv_kernel_size'] = (3, 3)
        >>> model.params['pool_kernel_size'] = (2, 2)
        >>> model.params['dropout_rate'] = 0.2
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(
            with_embedding=True
        )
        params.add(Param(name='mask_value', value=0,
                         desc="The value to be masked from inputs."))
        params.add(Param(name='char_embedding_input_dim', value=100,
                         desc="The input dimension of character embedding layer."))
        params.add(Param(name='char_embedding_output_dim', value=8,
                         desc="The output dimension of character embedding layer."))
        params.add(Param(name='char_conv_filters', value=100,
                         desc="The filter size of character convolution layer."))
        params.add(Param(name='char_conv_kernel_size', value=5,
                         desc="The kernel size of character convolution layer."))
        params.add(Param(name='first_scale_down_ratio', value=0.3,
                         desc="The channel scale down ratio of the convolution layer "
                              "before densenet."))
        params.add(Param(name='nb_dense_blocks', value=3,
                         desc="The number of blocks in densenet."))
        params.add(Param(name='layers_per_dense_block', value=8,
                         desc="The number of convolution layers in dense block."))
        params.add(Param(name='growth_rate', value=20,
                         desc="The filter size of each convolution layer in dense "
                              "block."))
        params.add(Param(name='transition_scale_down_ratio', value=0.5,
                         desc="The channel scale down ratio of the convolution layer "
                              "in transition block."))
        params.add(Param(name='conv_kernel_size', value=(3, 3),
                         desc="The kernel size of convolution layer in dense block."))
        params.add(Param(name='pool_kernel_size', value=(2, 2),
                         desc="The kernel size of pooling layer in transition block."))
        params.add(Param(
            'dropout_rate', 0.0,
            hyper_space=hyper_spaces.quniform(
                low=0.0, high=0.8, q=0.01),
            desc="The dropout rate."
        ))
        return params

    @classmethod
    def get_default_preprocessor(
        cls,
        truncated_mode: str = 'pre',
        truncated_length_left: typing.Optional[int] = None,
        truncated_length_right: typing.Optional[int] = None,
        filter_mode: str = 'df',
        filter_low_freq: float = 1,
        filter_high_freq: float = float('inf'),
        remove_stop_words: bool = False,
        ngram_size: typing.Optional[int] = 1,
    ) -> BasePreprocessor:
        """
        Model default preprocessor.

        The preprocessor's transform should produce a correctly shaped data
        pack that can be used for training.

        :return: Default preprocessor.
        """
        return preprocessors.BasicPreprocessor(
            truncated_mode=truncated_mode,
            truncated_length_left=truncated_length_left,
            truncated_length_right=truncated_length_right,
            filter_mode=filter_mode,
            filter_low_freq=filter_low_freq,
            filter_high_freq=filter_high_freq,
            remove_stop_words=remove_stop_words,
            ngram_size=ngram_size
        )

    @classmethod
    def get_default_padding_callback(
        cls,
        fixed_length_left: int = 10,
        fixed_length_right: int = 30,
        pad_word_value: typing.Union[int, str] = 0,
        pad_word_mode: str = 'pre',
        with_ngram: bool = True,
        fixed_ngram_length: int = None,
        pad_ngram_value: typing.Union[int, str] = 0,
        pad_ngram_mode: str = 'pre'
    ) -> BaseCallback:
        """
        Model default padding callback.

        The padding callback's on_batch_unpacked would pad a batch of data to
        a fixed length.

        :return: Default padding callback.
        """
        return callbacks.BasicPadding(
            fixed_length_left=fixed_length_left,
            fixed_length_right=fixed_length_right,
            pad_word_value=pad_word_value,
            pad_word_mode=pad_word_mode,
            with_ngram=with_ngram,
            fixed_ngram_length=fixed_ngram_length,
            pad_ngram_value=pad_ngram_value,
            pad_ngram_mode=pad_ngram_mode
        )

    def build(self):
        """Build model structure."""
        # Embedding
        self.embedding = self._make_default_embedding_layer()
        self.char_embedding = CharacterEmbedding(
            char_embedding_input_dim=self._params['char_embedding_input_dim'],
            char_embedding_output_dim=self._params['char_embedding_output_dim'],
            char_conv_filters=self._params['char_conv_filters'],
            char_conv_kernel_size=self._params['char_conv_kernel_size']
        )
        self.exact_maching = Matching(matching_type='exact')
        all_embed_dim = self._params['embedding_output_dim'] \
            + self._params['char_conv_filters'] + 1

        # Encoding
        self.left_encoder = SemanticComposite(
            all_embed_dim, self._params['dropout_rate'])
        self.right_encoder = SemanticComposite(
            all_embed_dim, self._params['dropout_rate'])

        # Interaction
        self.matching = Matching(matching_type='mul')

        # Feature Extraction
        self.conv = nn.Conv2d(
            in_channels=all_embed_dim,
            out_channels=int(all_embed_dim * self._params['first_scale_down_ratio']),
            kernel_size=1
        )
        self.dense_net = DenseNet(
            in_channels=int(all_embed_dim * self._params['first_scale_down_ratio']),
            nb_dense_blocks=self._params['nb_dense_blocks'],
            layers_per_dense_block=self._params['layers_per_dense_block'],
            growth_rate=self._params['growth_rate'],
            transition_scale_down_ratio=self._params['transition_scale_down_ratio'],
            conv_kernel_size=self._params['conv_kernel_size'],
            pool_kernel_size=self._params['pool_kernel_size']
        )
        self.max_pooling = nn.AdaptiveMaxPool2d((1, 1))

        # Output
        self.out_layer = self._make_output_layer(self.dense_net.out_channels)

        self.dropout = nn.Dropout(p=self._params['dropout_rate'])

    def forward(self, inputs):
        """Forward."""
        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   L = `input_word_left` sequence length
        #   R = `input_word_right` sequence length
        #   C = word length
        #   D1 = word embedding size
        #   D2 = character embedding size

        # shape = [B, L]
        # shape = [B, R]
        input_word_left, input_word_right = inputs['text_left'], inputs['text_right']
        mask_word_left = (input_word_left == self._params['mask_value'])
        mask_word_right = (input_word_right == self._params['mask_value'])

        # shape = [B, L, C]
        # shape = [B, R, C]
        input_char_left, input_char_right = inputs['ngram_left'], inputs['ngram_right']

        # shape = [B, L, D1]
        # shape = [B, R, D1]
        embed_word_left = self.dropout(self.embedding(input_word_left.long()))
        embed_word_right = self.dropout(self.embedding(input_word_right.long()))

        # shape = [B, L, D2]
        # shape = [B, R, D2]
        embed_char_left = self.dropout(self.char_embedding(input_char_left.long()))
        embed_char_right = self.dropout(self.char_embedding(input_char_right.long()))

        # shape = [B, L, 1]
        # shape = [B, R, 1]
        exact_match_left, exact_match_right = self.exact_maching(
            input_word_left, input_word_right)
        exact_match_left = exact_match_left.masked_fill(mask_word_left, 0)
        exact_match_right = exact_match_right.masked_fill(mask_word_right, 0)
        exact_match_left = torch.unsqueeze(exact_match_left, dim=-1)
        exact_match_right = torch.unsqueeze(exact_match_right, dim=-1)

        # shape = [B, L, D]
        # shape = [B, R, D]
        embed_left = torch.cat(
            [embed_word_left, embed_char_left, exact_match_left], dim=-1)
        embed_right = torch.cat(
            [embed_word_right, embed_char_right, exact_match_right], dim=-1)

        encode_left = self.left_encoder(embed_left)
        encode_right = self.right_encoder(embed_right)

        # shape = [B, L, R, D]
        interaction = self.matching(encode_left, encode_right)

        interaction = self.conv(self.dropout(interaction.permute(0, 3, 1, 2)))
        interaction = self.dense_net(interaction)
        interaction = self.max_pooling(interaction).squeeze(dim=-1).squeeze(dim=-1)

        output = self.out_layer(interaction)
        return output
