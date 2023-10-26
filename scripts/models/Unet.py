import torch
import torch.nn as nn
from utils.cheby_shev import SphericalChebConv
from models.model_template import template
from utils.partial_laplacians import get_partial_laplacians
from utils.healpix_pool_unpool import Healpix


class SphericalCheb(nn.Module):
    """
    Spherical Chebyshev Convolution layer followed by an activation.
    """
    def __init__(self, in_channels, out_channels, lap, kernel_size):
        super().__init__()
        self.conv = SphericalChebConv(in_channels, out_channels, lap, kernel_size)
        self.act = nn.PReLU() if out_channels > 1 else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))


class SphericalChebBlock(nn.Module):
    """
    Two subsequent Spherical Chebyshev Convolution layers.
    """
    def __init__(self, in_channels, middle_channels, out_channels, lap, kernel_size):
        super().__init__()
        self.layer1 = SphericalCheb(in_channels, middle_channels, lap, kernel_size)
        self.layer2 = SphericalCheb(middle_channels, out_channels, lap, kernel_size)

    def forward(self, x):
        return self.layer2(self.layer1(x))


class SphericalChebPool(nn.Module):
    """
    Spherical Chebyshev Convolution layer followed by pooling.
    """
    def __init__(self, in_channels, out_channels, lap, pooling, kernel_size):
        super().__init__()
        self.conv = SphericalCheb(in_channels, out_channels, lap, kernel_size)
        self.pool = pooling

    def forward(self, x):
        return self.conv(self.pool(x))


class SphericalChebPoolConcat(nn.Module):
    """
    Spherical Chebyshev Convolution layer followed by pooling and concatenation.
    """
    def __init__(self, in_channels, out_channels, lap, pooling, kernel_size):
        super().__init__()
        self.conv_pool = SphericalChebPool(in_channels, in_channels, lap, pooling, kernel_size)
        self.conv = SphericalCheb(2 * in_channels, out_channels, lap, kernel_size)

    def forward(self, x, concat_data):
        return self.conv(torch.cat((self.conv_pool(x), concat_data), dim=2))


class SphericalChebPoolCheb(nn.Module):
    """
    Spherical Chebyshev Convolution layer followed by pooling and another Convolution layer.
    """
    def __init__(self, in_channels, middle_channels, out_channels, lap, pooling, kernel_size):
        super().__init__()
        self.conv_pool = SphericalChebPool(in_channels, middle_channels, lap, pooling, kernel_size)
        self.conv = SphericalCheb(middle_channels, out_channels, lap, kernel_size)

    def forward(self, x):
        return self.conv(self.conv_pool(x))


class Encoder(nn.Module):
    """
    Encoder part of the U-Net architecture.
    """
    def __init__(self, pooling, laps, kernel_size):
        super().__init__()
        self.enc_l3 = SphericalChebBlock(1, 32, 64, laps[3], kernel_size)
        self.enc_l2 = SphericalChebPool(64, 128, laps[2], pooling, kernel_size)
        self.enc_l1 = SphericalChebPool(128, 256, laps[1], pooling, kernel_size)
        self.enc_l0 = SphericalChebPoolCheb(256, 512, 256, laps[0], pooling, kernel_size)

    def forward(self, x):
        x_enc3 = self.enc_l3(x)
        x_enc2 = self.enc_l2(x_enc3)
        x_enc1 = self.enc_l1(x_enc2)
        x_enc0 = self.enc_l0(x_enc1)
        return x_enc0, x_enc1, x_enc2, x_enc3


class Decoder(nn.Module):
    """
    Decoder part of the U-Net architecture.
    """
    def __init__(self, unpooling, laps, kernel_size):
        super().__init__()
        self.dec_l1 = SphericalChebPoolConcat(256, 128, laps[1], unpooling, kernel_size)
        self.dec_l2 = SphericalChebPoolConcat(128, 64, laps[2], unpooling, kernel_size)
        self.dec_l3 = SphericalChebPoolConcat(64, 64, laps[3], unpooling, kernel_size)
        self.dec_out = SphericalChebBlock(64, 32, 1, laps[3], kernel_size)

    def forward(self, x_enc0, x_enc1, x_enc2, x_enc3):
        x = self.dec_l1(x_enc0, x_enc1)
        x = self.dec_l2(x, x_enc2)
        x = self.dec_l3(x, x_enc3)
        return self.dec_out(x)


class SphericalUNet(template):
    """
    Full U-Net architecture composed of an encoder and a decoder.
    """
    def __init__(self, params):
        super().__init__(params)
        self.kernel_size = params["kernel_size"]
        pooling_class = Healpix()
        self.laps = get_partial_laplacians(params['nside_lr'], 4, params['order'], 'normalized')

        self.encoder = Encoder(pooling_class.pooling, self.laps, self.kernel_size)
        self.decoder = Decoder(pooling_class.unpooling, self.laps, self.kernel_size)

    def forward(self, x):
        return self.decoder(*self.encoder(x))
