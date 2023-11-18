
import pytorch_lightning as pl
from torch import nn
from torch import optim
import torch

from scripts.utils import SphericalChebConv, get_partial_laplacians

class ThreeConv(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)
        laps = get_partial_laplacians(params["data"]["nside"], 1, params["data"]["order"], 'normalized')
        self.conv_in = SphericalChebConv(params["architecture"]["dim_in"], params["architecture"]["inner_dim"], laps[0], params["architecture"]["kernel_size"])
        self.conv_mid = SphericalChebConv(params["architecture"]["inner_dim"], params["architecture"]["inner_dim"], laps[0], params["architecture"]["kernel_size"])
        self.conv_out = SphericalChebConv(params["architecture"]["inner_dim"], params["architecture"]["dim_out"], laps[0], params["architecture"]["kernel_size"])
        self.norm_in = nn.BatchNorm1d(params["architecture"]["inner_dim"])
        self.norm_mid = nn.BatchNorm1d(params["architecture"]["inner_dim"])
        self.act = torch.nn.Mish()
        self.loss_fn = nn.MSELoss()
        self.learning_rate = params["train"]["learning_rate"]
        self.gamma = params["train"]["gamma"]
    
    def forward(self, x):
        x = self.conv_in(x)
        x = self.act(self.norm_in(x.permute(0, 2, 1)).permute(0, 2, 1))
        x = self.conv_mid(x)
        x = self.act(self.norm_mid(x.permute(0, 2, 1)).permute(0, 2, 1))
        x = self.conv_out(x)
        return x
    
    def training_step(self, batch, batch_idx):
        y, x = batch
        y_hat = self(x)
        loss =  self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y, x = batch
        y_hat = self(x)
        loss =  self.loss_fn(y_hat, y)
        self.log('val_loss', loss, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}  