"""DeepRank Preprocessor."""

from tqdm import tqdm
import typing

from . import units
from matchzoo import DataPack
from matchzoo.engine.base_preprocessor import BasePreprocessor
from .build_vocab_unit import build_vocab_unit
from .build_unit_from_data_pack import build_unit_from_data_pack
from .chain_transform import chain_transform

tqdm.pandas()


class DeepRankPreprocessor(BasePreprocessor):
    """
    DeepRank model preprocessor helper.

        For pre-processing, all the words in documents and queries are white-space
        tokenized, lower-cased, and stemmed using the Krovetz stemmer.
        Stopword removal is performed on query and document words using the
        INQUERY stop list.
        Words occurred less than 5 times in the collection are removed from all
        the document.
        Querys are truncated below a max_length.

    :param filter_low_freq: Float, lower bound value used by
        :class:`FrequenceFilterUnit`.
    :param half_window_size: int, half of the match window size (not including
        the center word), so the real window size is 2 * half_window_size + 1
    :padding_token_index: int: vocabulary index of pad token, default 0

    Example:
        >>> import matchzoo as mz
        >>> train_data = mz.datasets.toy.load_data('train')
        >>> test_data = mz.datasets.toy.load_data('test')
        >>> preprocessor = mz.preprocessors.DeepRankPreprocessor(
        ...     filter_low_freq=5,
        ...     half_window_size=10,
        ...     padding_token_index=0,
        ... )
        >>> preprocessor = preprocessor.fit(train_data, verbose=0)
        >>> preprocessor.context['vocab_size']
        105
        >>> processed_train_data = preprocessor.transform(train_data,
        ...                                               verbose=0)
        >>> type(processed_train_data)
        <class 'matchzoo.data_pack.data_pack.DataPack'>
        >>> test_data_transformed = preprocessor.transform(test_data,
        ...                                                verbose=0)
        >>> type(test_data_transformed)
        <class 'matchzoo.data_pack.data_pack.DataPack'>

    """

    def __init__(self,
                 truncated_mode: str = 'pre',
                 truncated_length_left: int = None,
                 truncated_length_right: int = None,
                 filter_low_freq: float = 5,
                 half_window_size: int = 5,
                 padding_token_index: int = 0):
        """Initialization."""
        super().__init__()

        self._truncated_mode = truncated_mode
        self._truncated_length_left = truncated_length_left
        self._truncated_length_right = truncated_length_right
        if self._truncated_length_left:
            self._left_truncatedlength_unit = units.TruncatedLength(
                self._truncated_length_left, self._truncated_mode
            )
        if self._truncated_length_right:
            self._right_truncatedlength_unit = units.TruncatedLength(
                self._truncated_length_right, self._truncated_mode
            )

        self._counter_unit = units.FrequencyCounter()
        self._filter_unit = units.FrequencyFilter(
            low=filter_low_freq,
            mode="tf"
        )

        self._units = [
            units.Tokenize(),
            units.Lowercase(),
            units.Stemming(stemmer="krovetz"),
            units.stop_removal.StopRemoval(),  # todo: INQUERY stop list ?
        ]

        self._context["filter_low_freq"] = filter_low_freq
        self._context["half_window_size"] = half_window_size
        self._context['padding_unit'] = units.PaddingLeftAndRight(
            left_padding_num=self._context["half_window_size"],
            right_padding_num=self._context["half_window_size"],
            pad_value=padding_token_index,
        )

    def fit(self, data_pack: DataPack, verbose: int = 1):
        """
        Fit pre-processing context for transformation.

        :param data_pack: data_pack to be preprocessed.
        :param verbose: Verbosity.
        :return: class:`BasicPreprocessor` instance.
        """
        data_pack = data_pack.apply_on_text(chain_transform(self._units),
                                            verbose=verbose)
        fitted_counter_unit = build_unit_from_data_pack(self._counter_unit,
                                                        data_pack,
                                                        flatten=False,
                                                        mode='right',
                                                        verbose=verbose)
        self._context['counter_unit'] = fitted_counter_unit
        self._context['term_idf'] = fitted_counter_unit.context["idf"]

        fitted_filter_unit = build_unit_from_data_pack(self._filter_unit,
                                                       data_pack,
                                                       flatten=False,
                                                       mode='right',
                                                       verbose=verbose)
        data_pack = data_pack.apply_on_text(fitted_filter_unit.transform,
                                            mode='right', verbose=verbose)
        self._context['filter_unit'] = fitted_filter_unit

        vocab_unit = build_vocab_unit(data_pack, verbose=verbose)
        self._context['vocab_unit'] = vocab_unit

        vocab_size = len(vocab_unit.state['term_index'])
        self._context['vocab_size'] = vocab_size
        self._context['embedding_input_dim'] = vocab_size

        return self

    def transform(self, data_pack: DataPack, verbose: int = 1) -> DataPack:
        """
        Apply transformation on data.

        :param data_pack: Inputs to be preprocessed.
        :param verbose: Verbosity.

        :return: Transformed data as :class:`DataPack` object.
        """
        data_pack = data_pack.copy()
        # simple preprocessing
        data_pack.apply_on_text(chain_transform(self._units), inplace=True,
                                verbose=verbose)
        # filter
        data_pack.apply_on_text(self._context['filter_unit'].transform,
                                mode='right', inplace=True, verbose=verbose)
        # token to id
        data_pack.apply_on_text(self._context['vocab_unit'].transform,
                                mode='both', inplace=True, verbose=verbose)
        # truncate
        if self._truncated_length_left:
            data_pack.apply_on_text(self._left_truncatedlength_unit.transform,
                                    mode='left', inplace=True, verbose=verbose)
        if self._truncated_length_right:
            data_pack.apply_on_text(self._right_truncatedlength_unit.transform,
                                    mode='right', inplace=True,
                                    verbose=verbose)
        # add length
        data_pack.append_text_length(inplace=True, verbose=verbose)
        data_pack.drop_empty(inplace=True)
        # padding on left and right for matching window
        data_pack.apply_on_text(self._context['padding_unit'].transform,
                                mode='right', inplace=True, verbose=verbose)
        return data_pack
