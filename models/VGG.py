# Refactored code by ChatGPT-4
import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
from srsphere.utils.partial_laplacians import get_partial_laplacians
from srsphere.utils.healpix_pool_unpool import Healpix
from srsphere.models.ResUnet import Block
from srsphere.utils.cheby_shev import SphericalChebConv

class VGG(pl.LightningModule):
    def __init__(self, nside=64, order=2, depth=5, kernel_size=30, model="VGG16"):
        super().__init__()

        # Initialize parameters
        self.nside = nside
        self.order = order
        self.depth = depth
        self.kernel_size = kernel_size

        # Compute partial laplacians
        self.laps = get_partial_laplacians(self.nside, self.depth, self.order, 'normalized')

        # Define blocks of layers
        if model == "VGG16":
            self.blocks = nn.ModuleList([
                self._make_block(1, 64, self.laps[4]),
                self._make_block(64, 128, self.laps[3]),
                self._make_block(128, 256, self.laps[2], num_layers=3),
                self._make_block(256, 512, self.laps[1], num_layers=3),
                self._make_block(512, 512, self.laps[0], num_layers=3),
            ])
        elif model == "VGG19":
            self.blocks = nn.ModuleList([
                self._make_block(1, 64, self.laps[4]),
                self._make_block(64, 128, self.laps[3]),
                self._make_block(128, 256, self.laps[2], num_layers=4),
                self._make_block(256, 512, self.laps[1], num_layers=4),
                self._make_block(512, 512, self.laps[0], num_layers=4),
            ])

        # Define pooling operation
        self.pool = Healpix().pooling

    def _make_block(self, in_channels, out_channels, laplacian, num_layers=2):
        # Create a block of layers with optional pooling
        layers = [
            Block(in_channels, out_channels, laplacian, self.kernel_size) if (n==0) 
            else Block(out_channels, out_channels, laplacian, self.kernel_size) for n in range(num_layers)
            ]
        return nn.Sequential(*layers)

    def forward(self, x):
        # Pass input through each block and pool the output
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)

        # Ensure we have a valid list of features
        assert features, "No features extracted. Check the model architecture."

        return features
    
class VGG_cosmo(VGG):
    def __init__(self, params):
        self.nside = params["nside_hr"]
        self.order = params["order"]
        self.kernel_size = params["kernel_size"]
        self.depth = 5
        
        super().__init__(self.nside, self.order, self.depth, self.kernel_size, model="VGG16")
        self.n_params = 4 # Omega_m, Omega_b, sigma_8, h
        self.patch_size = (self.nside // self.order) ** 2
        self.final_size = self.patch_size // (4 ** (self.depth-1))

        self.block1 = Block(512, 512, self.laps[0], self.kernel_size)
        self.block2 = Block(512, 512, self.laps[0], self.kernel_size)
        self.sconv = SphericalChebConv(512, 1, self.laps[0], self.kernel_size)

        self.econv = nn.Linear(self.final_size, self.n_params)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
            # Do not pool after the last block
            if block != self.blocks[-1]:
                x = self.pool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.sconv(x)
        x = torch.squeeze(x, dim=2)
        x = self.econv(x)
        return x

class VGG_Unet(pl.LightningModule):
    def __init__(self, nside=64, order=2, depth=5, kernel_size=30, model="VGG16"):
        super().__init__()

        # Initialize parameters
        self.nside = nside
        self.order = order
        self.depth = depth
        self.kernel_size = kernel_size

        # Compute partial laplacians
        self.laps = get_partial_laplacians(self.nside, self.depth, self.order, 'normalized')

        # Define blocks of layers
        if model == "VGG16":
            self.encoder = nn.ModuleList([
                self._make_block(1, 64, self.laps[4]),
                self._make_block(64, 128, self.laps[3]),
                self._make_block(128, 256, self.laps[2], num_layers=3),
                self._make_block(256, 512, self.laps[1], num_layers=3),
                self._make_block(512, 512, self.laps[0], num_layers=3),
            ])
            self.decoder = nn.ModuleList([
                self._make_block(512, 512, self.laps[0], num_layers=3),
                self._make_block(512, 256, self.laps[1], num_layers=3),
                self._make_block(256, 128, self.laps[2], num_layers=3),
                self._make_block(128, 64, self.laps[3]),
                self._make_block(64, 1, self.laps[4]),
            ])
        elif model == "VGG19":
            self.encoder = nn.ModuleList([
                self._make_block(1, 64, self.laps[4]),
                self._make_block(64, 128, self.laps[3]),
                self._make_block(128, 256, self.laps[2], num_layers=4),
                self._make_block(256, 512, self.laps[1], num_layers=4),
                self._make_block(512, 512, self.laps[0], num_layers=4),
            ])
            self.decoder = nn.ModuleList([
                self._make_block(512*2, 512, self.laps[0], num_layers=4),
                self._make_block(512*2, 256, self.laps[1], num_layers=4),
                self._make_block(256*2, 128, self.laps[2], num_layers=4),
                self._make_block(128*2, 64, self.laps[3]),
                self._make_block(64, 1, self.laps[4]),
            ])

        self.bottleneck = self._make_block(512, 512, self.laps[0], num_layers=2)

        # Define pooling operation
        self.pooling = Healpix().pooling
        self.unpooling = Healpix().unpooling

    def _make_block(self, in_channels, out_channels, laplacian, num_layers=2):
        # Create a block of layers with optional pooling
        layers = [
            Block(in_channels, out_channels, laplacian, self.kernel_size) if (n==0) 
            else Block(out_channels, out_channels, laplacian, self.kernel_size) for n in range(num_layers)
            ]
        return nn.Sequential(*layers)

    def forward(self, x):
        skip_connections = []

        # downsample
        for e1, e2, e3, e4, e5 in self.encoder:
            x = e1(x)
            skip_connections.append(x)
            x = self.pooling(x)
            x = e2(x)
            skip_connections.append(x)
            x = self.pooling(x)
            x = e3(x)
            skip_connections.append(x)
            x = self.pooling(x)
            x = e4(x)
            skip_connections.append(x)
            x = self.pooling(x)
            x = e5(x)
            skip_connections.append(x)

        # bottleneck
        x = self.bottleneck(x)

        # upsample
        for d1, d2, d3, d4, d5 in self.decoder:
            tmp_connection = skip_connections.pop()
            x = torch.cat([x, tmp_connection], dim=2)
            x = self.unpooling(x)
            x = d1(x)
            tmp_connection = skip_connections.pop()
            x = torch.cat([x, tmp_connection], dim=2)
            x = self.unpooling(x)
            x = d2(x)
            tmp_connection = skip_connections.pop()
            x = torch.cat([x, tmp_connection], dim=2)
            x = self.unpooling(x)
            x = d3(x)
            tmp_connection = skip_connections.pop()
            x = torch.cat([x, tmp_connection], dim=2)
            x = self.unpooling(x)
            x = d4(x)
            tmp_connection = skip_connections.pop()
            x = torch.cat([x, tmp_connection], dim=2)
            x = self.unpooling(x)
            x = d5(x)


        return self.final_conv(x)