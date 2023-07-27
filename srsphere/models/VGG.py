# Refactored code by ChatGPT-4
import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
from srsphere.utils.partial_laplacians import get_partial_laplacians
from srsphere.utils.healpix_pool_unpool import Healpix
from srsphere.models.ResUnet import Block


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