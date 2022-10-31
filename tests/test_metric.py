import numpy as np
import torch
from sklearn.metrics import mean_squared_error

from src.model_finetuning.metric import MCRMSELoss


def test_sklearn_metric_matches_torch():
    a = torch.randn(10, 6)
    b = torch.randn(10, 6)

    sklearn_loss = []
    for ii in range(6):
        loss = mean_squared_error(a[:, ii], b[:, ii], squared=False)
        sklearn_loss.append(loss)

    assert np.isclose(np.mean(sklearn_loss), MCRMSELoss().forward(a, b))
