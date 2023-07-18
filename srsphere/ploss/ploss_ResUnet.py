
import torch
from torch import nn
import pytorch_lightning as pl
import sys
sys.path.append('/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/')

from srsphere.models.ResUnet import Unet

class ploss_ResUnet(Unet):
    def __init__(self, params):
        super().__init__(params)

    def forward(self, x):
        x = self.init_conv(x)

        features = []

        # downsample
        for block1, block2, downsample in self.downs:
            x = block1(x)
            x = block2(x)
            features.append(x)  # Save the feature maps
            x = downsample(x)

        # bottleneck:
        x = self.mid_block1(x)
        features.append(x)
        return features  # Return the final output and the feature maps


class PerceptualLoss(nn.Module):
    def __init__(self, params, ckpt_path=None):
        super().__init__()
        self.model = ploss_ResUnet(params)
        if ckpt_path is not None:
            self.model.load_state_dict(torch.load(ckpt_path)["state_dict"], strict=False)
            for param in self.model.parameters():
                param.requires_grad = False
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target):
        # Pass the output and target through the model to get the feature maps
        features_out = self.model(output)
        features_target = self.model(target)

        # Compute the MSE loss for each pair of feature maps
        losses = [self.mse_loss(out, tgt) for out, tgt in zip(features_out, features_target)]

        # Return the average loss
        return sum(losses) / len(losses) if losses else 0

