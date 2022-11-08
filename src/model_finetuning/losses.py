import torch
import torch.nn as nn
from torch import Tensor


class MCRMSELoss(nn.Module):

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, y_pred: Tensor, y_true: Tensor):
        """Calculate mean column-wise rmse on columns

        :param y_pred: tensor of shape (bs, 6)
        :param y_true: tensor of shape (bs, 6)
        :return: tensor of shape 0 (scalar with grad)
        """

        mse = self.mse(y_pred, y_true).mean(0)  # column-wise mean
        rmse = torch.sqrt(mse + 1e-7)

        return rmse.mean()

    def class_mcrmse(self, y_pred: Tensor, y_true: Tensor):
        mse = self.mse(y_pred, y_true).mean(0)  # column-wise mean
        rmse = torch.sqrt(mse + 1e-7)

        return rmse.squeeze()
