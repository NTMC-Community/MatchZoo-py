import torch
import pytest

from matchzoo.modules import Matching


def test_matching():
    x = torch.randn(2, 3, 2)
    y = torch.randn(2, 4, 2)
    z = torch.randn(2, 3, 3)
    for matching_type in ['dot', 'mul', 'plus', 'minus', 'concat']:
        Matching(matching_type=matching_type)(x, y)
    with pytest.raises(ValueError):
        Matching(matching_type='error')
    with pytest.raises(RuntimeError):
        Matching()(x, z)
