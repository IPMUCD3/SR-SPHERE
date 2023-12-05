from torch import nn
from scripts.layers.cheby_shev import SphericalChebConv
from scripts.layers.normalization import Norms
from scripts.layers.activation import Acts
from scripts.layers.timeembedding import TimeEmbed
from scripts.utils.healpix_pool_unpool import Healpix
from scripts.utils.diffusion_utils import exists

'''
The code defines various neural network blocks, specifically for ResNet architecture.
These blocks include standard convolutional layers, normalization, and activation functions.
Additionally, it includes a specialized block for BigGAN's upsampling/downsampling operations.
The implementation is designed to be modular, allowing easy integration into larger neural network models.
'''

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, 
                laplacian, kernel_size=8, 
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
            self.mlp1 = TimeEmbed(time_emb_dim, in_channels)
            self.mlp2 = TimeEmbed(time_emb_dim, out_channels)

        self.norm1 = Norms(in_channels, norm_type)
        self.norm2 = Norms(out_channels, norm_type)

        self.act1 = Acts(act_type) 
        self.act2 = Acts(act_type)

        if pooling == "pooling":
            self.pooling = Healpix().pooling
        elif pooling == "unpooling":
            self.pooling = Healpix().unpooling
        elif pooling == "identity":
            self.pooling = nn.Identity()
        else:
            raise ValueError("must be pooling, unpooling or identity")
        self.conv_res = SphericalChebConv(in_channels, out_channels, laplacian, kernel_size) if in_channels != out_channels else nn.Identity()
        self.conv1 = SphericalChebConv(in_channels, out_channels, laplacian, kernel_size)
        self.conv2 = SphericalChebConv(out_channels, out_channels, laplacian, kernel_size)

    def forward(self, x, time_emb=None):
        h = x
        res = self.conv_res(self.pooling(x))

        if exists(time_emb):
            h = self.mlp1(time_emb) + h

        h = self.norm1(h)
        h = self.act1(h)
        h = self.pooling(h)
        h = self.conv1(h)

        if exists(time_emb):
            h = self.mlp2(time_emb) + h

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        return h + res
    
class ResnetBlock(nn.Module):
    """
    Up/Downsampling Residual block implemented from https://arxiv.org/abs/2311.05217.
    Originally https://arxiv.org/abs/1512.03385.
    """
    def __init__(self, in_channels, out_channels, 
                laplacian, pooling="identity", kernel_size=8, time_emb_dim=None, 
                norm_type="group", act_type="silu"):
        super().__init__()
        if exists(time_emb_dim):
            self.mlp = TimeEmbed(time_emb_dim, out_channels)

        if pooling == "pooling":
            self.pooling = Healpix().pooling
        elif pooling == "unpooling":
            self.pooling = Healpix().unpooling
        elif pooling == "identity":
            self.pooling = nn.Identity()
        else:
            raise ValueError("must be pooling, unpooling or identity")
        self.conv_res = SphericalChebConv(in_channels, out_channels, laplacian, kernel_size) if in_channels != out_channels else nn.Identity()
        self.block1 = Block(in_channels, out_channels, laplacian, kernel_size, norm_type, act_type)
        self.block2 = Block(out_channels, out_channels, laplacian, kernel_size, norm_type, act_type)

    def forward(self, x, time_emb=None):
        h = x
        res = self.conv_res(self.pooling(x))

        h = self.block1(h)
        h = self.pooling(h)

        if exists(time_emb):
            h = self.mlp(time_emb) + h

        h = self.block2(h)
        return h + res
