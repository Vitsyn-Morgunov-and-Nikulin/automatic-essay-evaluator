import torch
import torch.nn as nn
from torch import Tensor


class MCRMSELoss(nn.Module):

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, y_pred: Tensor, y_true: Tensor):
        mse = self.mse(y_pred, y_true).mean(0)  # column-wise mean
        rmse = torch.sqrt(mse + 1e-7)

        return rmse.mean()
