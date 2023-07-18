# Refactored code by ChatGPT-4
import torch
from torch import nn
import pytorch_lightning as pl
from srsphere.utils.partial_laplacians import get_partial_laplacians
from srsphere.utils.healpix_pool_unpool import Healpix
from srsphere.models.ResUnet import Block


class VGG16(pl.LightningModule):
    def __init__(self, nside=64, order=2, depth=5, kernel_size=30):
        super().__init__()

        # Initialize parameters
        self.nside = nside
        self.order = order
        self.depth = depth
        self.kernel_size = kernel_size

        # Compute partial laplacians
        self.laps = get_partial_laplacians(self.nside, self.depth, self.order, 'normalized')

        # Define blocks of layers
        self.blocks = nn.ModuleList([
            self._make_block(1, 64, self.laps[4]),
            self._make_block(64, 128, self.laps[3]),
            self._make_block(128, 256, self.laps[2], num_layers=3),
            self._make_block(256, 512, self.laps[1], num_layers=3),
            self._make_block(512, 512, self.laps[0], num_layers=3),
        ])

        # Define pooling operation
        self.pool = Healpix().pooling

    def _make_block(self, in_channels, out_channels, laplacian, num_layers=2):
        # Create a block of layers with optional pooling
        layers = [Block(in_channels, out_channels, laplacian, self.kernel_size) for _ in range(num_layers)]
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


class PerceptualLoss(nn.Module):
    def __init__(self, ckpt_path=None):
        super().__init__()

        # Initialize VGG16 model
        self.model = VGG16()

        # Load checkpoint if provided
        if ckpt_path is not None:
            self.model.load_state_dict(torch.load(ckpt_path)["state_dict"], strict=False)
        else:
            # Initialize the model parameters randomly
            for param in self.model.parameters():
                param.data.uniform_(-1, 1)

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Define loss function
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target):
        # Extract features from output and target
        features_out = self.model(output)
        features_target = self.model(target)

        # Ensure we have a valid list of features
        assert features_out and features_target, "No features extracted. Check the input and model architecture."

        # Compute loss for each pair of feature maps
        losses = [self.mse_loss(out, tgt) for out, tgt in zip(features_out, features_target)]

        # Return the average loss
        return sum(losses) / len(losses)

