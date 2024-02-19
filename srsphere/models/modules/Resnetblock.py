#Description: This file contains the implementation of the ResnetBlock class, which is used to define the ResNet architecture.

from torch import nn
import torch.nn.functional as F
from functools import partial
from deepsphere.cheby_shev import SphericalChebConv
from srsphere.models.modules.normalization import Norms
from srsphere.models.modules.activation import Acts
from srsphere.models.modules.timeembedding import TimeEmbed
from srsphere.diffusion.utils import exists

'''
The code defines various neural network blocks, specifically for ResNet architecture.
These blocks include standard convolutional layers, normalization, and activation functions.
Additionally, it includes a specialized block for BigGAN's upsampling/downsampling operations.
The implementation is designed to be modular, allowing easy integration into larger neural network models.
'''

class Upsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = partial(F.interpolate, scale_factor=4, mode="nearest")

    def forward(self, x):
        return self.up(x.permute(0, 2, 1)).permute(0, 2, 1)


class Downsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = partial(F.avg_pool1d, kernel_size=4)

    def forward(self, x):
        return self.down(x.permute(0, 2, 1)).permute(0, 2, 1)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, 
                laplacian, kernel_size=20, 
                norm_type="group", act_type="silu"):
        super().__init__()
        self.conv = SphericalChebConv(in_channels, out_channels, laplacian, kernel_size)
        self.norm = Norms(in_channels, norm_type)
        self.act = Acts(act_type) if out_channels > 1 else nn.Identity()

    def forward(self, x):
        return self.conv(self.act(self.norm(x)))

class ResnetBlock_BG(nn.Module):
    """
    Up/Downsampling Residual block of BigGAN. https://arxiv.org/abs/1809.11096
    """
    def __init__(self, in_channels, out_channels, 
                laplacian, pooling="identity", kernel_size=8, time_emb_dim=None, 
                norm_type="group", act_type="silu"):
        super().__init__()
        if exists(time_emb_dim):
            self.mlp1 = TimeEmbed(time_emb_dim, in_channels * 2)
            self.mlp2 = TimeEmbed(time_emb_dim, out_channels * 2)

        self.norm1 = Norms(in_channels, norm_type)
        self.norm2 = Norms(out_channels, norm_type)

        self.act1 = Acts(act_type) 
        self.act2 = Acts(act_type)

        if pooling == "pooling":
            self.pooling = Downsample()
        elif pooling == "unpooling":
            self.pooling = Upsample()
        elif pooling == "identity":
            self.pooling = nn.Identity()
        else:
            raise ValueError("must be pooling, unpooling or identity")
        #self.conv_res = SphericalChebConv(in_channels, out_channels, laplacian, kernel_size) if in_channels != out_channels else nn.Identity()
        self.conv_res = nn.Linear(in_channels, out_channels)
        self.conv1 = SphericalChebConv(in_channels, out_channels, laplacian, kernel_size)
        self.conv2 = SphericalChebConv(out_channels, out_channels, laplacian, kernel_size)

    def forward(self, x, time_emb=None):
        h = x
        res = self.conv_res(self.pooling(x))

        if exists(time_emb):
            t_e = self.mlp1(time_emb)
            scale, shift = t_e.chunk(2, dim = 2)
            h = h * (scale + 1) + shift

        h = self.norm1(h)
        h = self.act1(h)
        h = self.pooling(h)
        h = self.conv1(h)

        if exists(time_emb):
            t_e = self.mlp2(time_emb)
            scale, shift = t_e.chunk(2, dim = 2)
            h = h * (scale + 1) + shift

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        return h + res
    
class ResnetBlock(nn.Module):
    """
    Up/Downsampling Residual block implemented from https://arxiv.org/abs/2311.05217.
    Originally https://arxiv.org/abs/1512.03385.
    """
    def __init__(self, in_channels, out_channels, laplacian, 
                time_emb_dim, kernel_size=20, dropout=0.0, 
                norm_type="group", act_type="silu", 
                use_scale_shift_norm=False, use_conv=False, up=False, down=False):
        super().__init__()
        
        self.use_scale_shift_norm = use_scale_shift_norm
        self.emb = TimeEmbed(time_emb_dim, out_channels * 2 if use_scale_shift_norm else out_channels)

        self.in_layers = nn.Sequential(
            Norms(in_channels, norm_type),
            Acts(act_type),
            SphericalChebConv(in_channels, out_channels, laplacian, kernel_size),
        )

        self.updown = up or down
        if up:
            self.pooling = Upsample()
        elif down:
            self.pooling = Downsample()
        else:
            self.pooling = nn.Identity()

        self.out_layers = nn.Sequential(
            Norms(out_channels, norm_type),
            Acts(act_type),
            nn.Dropout(p=dropout),
            SphericalChebConv(out_channels, out_channels, laplacian, kernel_size),
        )

        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = SphericalChebConv(in_channels, out_channels, laplacian, kernel_size)
        else:
            self.skip_connection = nn.Linear(in_channels, out_channels)

    def forward(self, x, time_emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.pooling(h)
            h = in_conv(h)
            x = self.pooling(x)
        else:
            h = self.in_layers(x)
            
        emb_out = self.emb(time_emb)
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = emb_out.chunk(2, dim=2)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h