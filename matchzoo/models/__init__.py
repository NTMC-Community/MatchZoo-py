from .dense_baseline import DenseBaseline
from .dssm import DSSM
from .cdssm import CDSSM
from .drmm import DRMM
from .drmmtks import DRMMTKS
from .esim import ESIM
from .knrm import KNRM
from .conv_knrm import ConvKNRM


def list_available() -> list:
    from matchzoo.engine.base_model import BaseModel
    from matchzoo.utils import list_recursive_concrete_subclasses
    return list_recursive_concrete_subclasses(BaseModel)
