"""Model template for PyTorch Lightning"""

import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

class template(pl.LightningModule):    
    def __init__(self, params):
        super().__init__()
        self.lr_init = params['lr_init']
        self.lr_max = params['lr_max']
        self.epochs = params['num_epochs']
        self.steps_per_epoch = params['steps_per_epoch']
        self.loss_fn = nn.MSELoss()

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
        optimizer = optim.Adam(self.parameters(), lr=self.lr_init)
        #scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr_max, epochs = self.epochs, steps_per_epoch=self.steps_per_epoch)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}  