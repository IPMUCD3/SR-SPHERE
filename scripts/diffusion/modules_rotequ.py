
"""
implement From https://arxiv.org/abs/1911.09040
"""

import torch
import torch.nn as nn

class ReLU_rotequ(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.abs(x)/torch.max(torch.abs(x), torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))) * x
    
class BatchNorm_rotequ(nn.Module):
    def __init__(self, var_min=1e-8):
        super().__init__()
        self.var_min = var_min

    def forward(self, x):
        return x / torch.sqrt(torch.var(x, dim=0, keepdim=True) + self.var_min)