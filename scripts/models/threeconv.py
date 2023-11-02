
import pytorch_lightning as pl
from torch import nn
from torch import optim

from scripts.utils import SphericalChebConv, get_partial_laplacians

class ThreeConv(pl.LightningModule):
    def __init__(self, 
                 in_channels=1,
                 inner_channels=64,
                 nside=512,
                 order=4, 
                 kernel_size=8,
                 slope=0.1,
                 learning_rate=1e-3,
                 gamma=0.99):
        super().__init__()
        laps = get_partial_laplacians(nside, 1, order, 'normalized')
        self.conv_in = SphericalChebConv(in_channels, inner_channels, laps[0], kernel_size)
        self.conv_mid = SphericalChebConv(inner_channels, inner_channels, laps[0], kernel_size)
        self.conv_out = SphericalChebConv(inner_channels, in_channels, laps[0], kernel_size)
        self.ReLU = nn.LeakyReLU(slope)
        self.loss_fn = nn.MSELoss()
        self.learning_rate = learning_rate
        self.gamma = gamma
    
    def forward(self, x):
        x = self.ReLU(self.conv_in(x))
        x = self.ReLU(self.conv_mid(x))
        x = self.conv_out(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss =  self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss =  self.loss_fn(y_hat, y)
        self.log('val_loss', loss, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}  