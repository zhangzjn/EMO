import torch.nn as nn
from . import LOSS

__all__ = ['L1Loss', 'MSELoss']


@LOSS.register_module
class L1Loss(nn.L1Loss):
    def __init__(self, lam=1):
        super(L1Loss, self).__init__()
        self.lam = lam

    def forward(self, input, target):
        return super(L1Loss, self).forward(input, target) * self.lam


@LOSS.register_module
class MSELoss(nn.MSELoss):
    def __init__(self, lam=1):
        super(MSELoss, self).__init__()
        self.lam = lam

    def forward(self, input, target):
        return super(MSELoss, self).forward(input, target) * self.lam
