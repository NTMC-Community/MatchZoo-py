import torch
import pytest

from matchzoo.modules import Matching


def test_matching():
    x = torch.randn(2, 3, 2)
    y = torch.randn(2, 4, 2)
    z = torch.randn(2, 3, 3)
    for matching_type in ['dot', 'mul', 'plus', 'minus', 'concat']:
        x_mask = (torch.randint(low=0, high=2, size=(2,3)) == 0)
        y_mask = (torch.randint(low=0, high=2, size=(2,4)) == 0)
        Matching(matching_type=matching_type)(x, y)
        Matching(matching_type=matching_type)(x, y, x_mask=x_mask, y_mask=y_mask)
    with pytest.raises(ValueError):
        Matching(matching_type='error')
    with pytest.raises(RuntimeError):
        Matching()(x, z)
