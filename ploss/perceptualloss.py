from torch import nn
import numpy as np
import sys

sys.path.append('/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/')
from srsphere.models.VGG import VGG

def weights_init(m, patch_size=1024):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname=='ChebConv':
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.normal_(mean=0.0, std=np.sqrt(2/(m.in_channels * patch_size)))
        m.bias.data.fill_(0)

class PerceptualLoss(nn.Module):
    def __init__(self, model="VGG16", loss="mse"):
        super().__init__()

        # Initialize VGG16 model
        self.model = VGG(model=model)

        # Initialize the model parameters randomly
        self.model.apply(weights_init)

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Define loss function
        if loss == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss == "l1":
            self.loss_fn = nn.L1Loss()

    def forward(self, output, target):
        # Extract features from output and target
        features_out = self.model(output)
        features_target = self.model(target)

        # Ensure we have a valid list of features
        assert features_out and features_target, "No features extracted. Check the input and model architecture."

        # Compute loss for each pair of feature maps
        losses = [self.loss_fn(out, tgt) for out, tgt in zip(features_out, features_target)]

        # Return the average loss
        return sum(losses) / len(losses)
    
class PerceptualLoss_plus(nn.Module):
    def __init__(self, model="VGG16", add_loss="mse", lambda_=0.1):
        super().__init__()
        self.ploss = PerceptualLoss(model=model, loss=add_loss)
        self.lambda_ = lambda_
        if add_loss == "mse":
            self.add_loss = nn.MSELoss()
        elif add_loss == "l1":
            self.add_loss = nn.L1Loss()

    def forward(self, output, target):
        ploss = self.ploss(output, target)
        add_loss = self.add_loss(output, target)
        loss = self.lambda_ * ploss + add_loss
        return loss