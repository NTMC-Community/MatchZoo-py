from . import units
from .naive_preprocessor import NaivePreprocessor
from .basic_preprocessor import BasicPreprocessor
from .bert_preprocessor import BertPreprocessor
from .deeprank_preprocessor import DeepRankPreprocessor


def list_available() -> list:
    from matchzoo.engine.base_preprocessor import BasePreprocessor
    from matchzoo.utils import list_recursive_concrete_subclasses
    return list_recursive_concrete_subclasses(BasePreprocessor)
