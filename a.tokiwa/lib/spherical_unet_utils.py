"""Layers used in both Encoder and Decoder.
"""
# pylint: disable=W0221
import torch.nn.functional as F
import torch
from torch import nn

from cheby_shev import SphericalChebConv

class SphericalCheb(nn.Module):
    def __init__(self, in_channels, out_channels, lap, kernel_size):
        super().__init__()
        self.spherical_cheb = SphericalChebConv(in_channels, out_channels, lap, kernel_size)
        self.ReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.spherical_cheb(x)
        x = self.ReLU(x)
        return x

class SphericalChebPool(nn.Module):
    def __init__(self, in_channels, out_channels, lap, pooling, kernel_size):
        super().__init__()
        self.pooling = pooling
        self.spherical_cheb = SphericalCheb(in_channels, out_channels, lap, kernel_size)

    def forward(self, x):
        x = self.pooling(x)
        x = self.spherical_cheb(x)
        return x
    
class SphericalChebPoolConcat(nn.Module):
    def __init__(self, in_channels, out_channels, lap, pooling, kernel_size):
        super().__init__()
        self.spherical_cheb_pool = SphericalChebPool(in_channels, out_channels, lap, pooling, kernel_size)
        self.spherical_cheb = SphericalCheb(2* out_channels, out_channels, lap, kernel_size)

    def forward(self, x, concat_data):
        x = self.spherical_cheb_pool(x)
        x = torch.cat((x, concat_data), dim=2)
        x = self.spherical_cheb(x)
        return x

class SphericalChebBN(nn.Module):
    """Building Block with a Chebyshev Convolution, Batchnormalization, and ReLu activation.
    """

    def __init__(self, in_channels, out_channels, lap, kernel_size):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            kernel_size (int, optional): polynomial degree. Defaults to 3.
        """
        super().__init__()
        self.spherical_cheb = SphericalChebConv(in_channels, out_channels, lap, kernel_size)
        self.batchnorm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.tensor`: output [batch x vertices x channels/features]
        """
        x = self.spherical_cheb(x)
        x = self.batchnorm(x.permute(0, 2, 1))
        x = F.relu(x.permute(0, 2, 1))
        return x


class SphericalChebBNPool(nn.Module):
    """Building Block with a pooling/unpooling, a calling the SphericalChebBN block.
    """

    def __init__(self, in_channels, out_channels, lap, pooling, kernel_size):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            pooling (:obj:`torch.nn.Module`): pooling/unpooling module.
            kernel_size (int, optional): polynomial degree. Defaults to 3.
        """
        super().__init__()
        self.pooling = pooling
        self.spherical_cheb_bn = SphericalChebBN(in_channels, out_channels, lap, kernel_size)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.tensor`: output [batch x vertices x channels/features]
        """
        x = self.pooling(x)
        x = self.spherical_cheb_bn(x)
        return x