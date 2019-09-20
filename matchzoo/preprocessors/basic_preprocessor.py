"""Basic Preprocessor."""

from tqdm import tqdm
import typing

from . import units
from matchzoo import DataPack
from matchzoo.engine.base_preprocessor import BasePreprocessor
from .build_vocab_unit import build_vocab_unit
from .build_unit_from_data_pack import build_unit_from_data_pack
from .chain_transform import chain_transform

tqdm.pandas()


class BasicPreprocessor(BasePreprocessor):
    """
    Baisc preprocessor helper.

    :param truncated_mode: String, mode used by :class:`TruncatedLength`.
        Can be 'pre' or 'post'.
    :param truncated_length_left: Integer, maximize length of :attr:`left`
        in the data_pack.
    :param truncated_length_right: Integer, maximize length of :attr:`right`
        in the data_pack.
    :param filter_mode: String, mode used by :class:`FrequenceFilterUnit`. Can
        be 'df', 'cf', and 'idf'.
    :param filter_low_freq: Float, lower bound value used by
        :class:`FrequenceFilterUnit`.
    :param filter_high_freq: Float, upper bound value used by
        :class:`FrequenceFilterUnit`.
    :param remove_stop_words: Bool, use :class:`StopRemovalUnit` unit or not.

    Example:
        >>> import matchzoo as mz
        >>> train_data = mz.datasets.toy.load_data('train')
        >>> test_data = mz.datasets.toy.load_data('test')
        >>> preprocessor = mz.preprocessors.BasicPreprocessor(
        ...     truncated_length_left=10,
        ...     truncated_length_right=20,
        ...     filter_mode='df',
        ...     filter_low_freq=2,
        ...     filter_high_freq=1000,
        ...     remove_stop_words=True
        ... )
        >>> preprocessor = preprocessor.fit(train_data, verbose=0)
        >>> preprocessor.context['vocab_size']
        226
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
                 filter_mode: str = 'df',
                 filter_low_freq: float = 1,
                 filter_high_freq: float = float('inf'),
                 remove_stop_words: bool = False,
                 ngram_size: typing.Optional[int] = None):
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
        self._filter_unit = units.FrequencyFilter(
            low=filter_low_freq,
            high=filter_high_freq,
            mode=filter_mode
        )
        self._units = self._default_units()
        if remove_stop_words:
            self._units.append(units.stop_removal.StopRemoval())
        self._ngram_size = ngram_size
        if ngram_size:
            self._context['ngram_process_unit'] = units.NgramLetter(
                ngram=ngram_size, reduce_dim=True
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

        if self._ngram_size:
            data_pack = data_pack.apply_on_text(
                self._context['ngram_process_unit'].transform,
                mode='both',
                verbose=verbose
            )
            ngram_unit = build_vocab_unit(data_pack, verbose=verbose)
            self._context['ngram_vocab_unit'] = ngram_unit
            self._context['ngram_vocab_size'] = len(
                ngram_unit.state['term_index'])
        return self

    def transform(self, data_pack: DataPack, verbose: int = 1) -> DataPack:
        """
        Apply transformation on data, create truncated length representation.

        :param data_pack: Inputs to be preprocessed.
        :param verbose: Verbosity.

        :return: Transformed data as :class:`DataPack` object.
        """
        data_pack = data_pack.copy()
        data_pack.apply_on_text(chain_transform(self._units), inplace=True,
                                verbose=verbose)

        data_pack.apply_on_text(self._context['filter_unit'].transform,
                                mode='right', inplace=True, verbose=verbose)
        data_pack.apply_on_text(self._context['vocab_unit'].transform,
                                mode='both', inplace=True, verbose=verbose)
        if self._truncated_length_left:
            data_pack.apply_on_text(self._left_truncatedlength_unit.transform,
                                    mode='left', inplace=True, verbose=verbose)
        if self._truncated_length_right:
            data_pack.apply_on_text(self._right_truncatedlength_unit.transform,
                                    mode='right', inplace=True,
                                    verbose=verbose)
        data_pack.append_text_length(inplace=True, verbose=verbose)

        data_pack.drop_empty(inplace=True)
        return data_pack
