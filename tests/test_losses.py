import torch
import numpy as np

from matchzoo import losses


def test_hinge_loss():
    true_value = torch.Tensor([[1.2], [1], [1], [1]])
    pred_value = torch.Tensor([[1.2], [0.1], [0], [-0.3]])
    expected_loss = torch.Tensor([(0 + 1 - 0.3 + 0) / 2.0])
    loss = losses.RankHingeLoss()(pred_value, true_value)
    assert torch.isclose(expected_loss, loss)
    expected_loss = torch.Tensor(
        [(2 + 0.1 - 1.2 + 2 - 0.3 + 0) / 2.0])
    loss = losses.RankHingeLoss(margin=2)(pred_value, true_value)
    assert torch.isclose(expected_loss, loss)
    true_value = torch.Tensor(
        [[1.2], [1], [0.8], [1], [1], [0.8]])
    pred_value = torch.Tensor(
        [[1.2], [0.1], [-0.5], [0], [0], [-0.3]])
    expected_loss = torch.Tensor(
        [(0 + 1 - 0.15) / 2.0])
    loss = losses.RankHingeLoss(num_neg=2, margin=1)(
        pred_value, true_value)
    assert torch.isclose(expected_loss, loss)


def test_rank_crossentropy_loss():
    losses.neg_num = 1

    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    true_value = torch.Tensor([[1.], [0.], [0.], [1.]])
    pred_value = torch.Tensor([[0.8], [0.1], [0.8], [0.1]])
    expected_loss = torch.Tensor(
        [(-np.log(softmax([0.8, 0.1])[0]) - np.log(
            softmax([0.8, 0.1])[1])) / 2])
    loss = losses.RankCrossEntropyLoss()(pred_value, true_value)
    assert torch.isclose(expected_loss, loss)
    true_value = torch.Tensor([[1.], [0.], [0.], [0.], [1.], [0.]])
    pred_value = torch.Tensor([[0.8], [0.1], [0.1], [0.8], [0.1], [0.1]])
    expected_loss = torch.Tensor(
        [(-np.log(softmax([0.8, 0.1, 0.1])[0]) - np.log(
            softmax([0.8, 0.1, 0.1])[1])) / 2])
    loss = losses.RankCrossEntropyLoss(num_neg=2)(
        pred_value, true_value)
    assert torch.isclose(expected_loss, loss)
