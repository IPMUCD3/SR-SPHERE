import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl
from cheby_shev import SphericalChebConv
from healpix_pool_unpool import Healpix
from laplacian_funcs import get_healpix_laplacians
from maploader import MapDataset

class paramset():
    def __init__(self, MapDataset, depth, kernel_size, max_lr, epochs, batch_size):
        self.len_train = MapDataset.len_train
        self.len_val = MapDataset.len_val
        self.depth = depth
        self.kernel_size = kernel_size
        self.max_lr = max_lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.steps_per_epoch = MapDataset.n_maps * MapDataset.split_order**2 // batch_size
        self.npix_hr = MapDataset.npix_hr
        self.npix_lr = MapDataset.npix_lr

class template(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.kernel_size = params.kernel_size
        self.len_val = params.len_val
        self.len_train = params.len_train
        self.laps = get_healpix_laplacians(params.npix_hr, params.depth, "normalized")
        self.pooling = Healpix().pooling
        self.unpooling = Healpix().unpooling
        self.loss_fn = nn.MSELoss()
        self.max_lr = params.max_lr
        self.epochs = params.epochs
        self.steps_per_epoch = params.steps_per_epoch
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return {'loss': loss}

    def validation_epoch_end(self, val_step_outputs):
        epoch_vloss = sum([value['val_batch_loss'] for value in val_step_outputs])/self.len_val
        print('valid Loss: {:e}'.format(epoch_vloss))     
        epoch = self.current_epoch+1
        # Save state dict of model [10, 50, 100, 500, 1000] epoch
        if epoch in [10, 50, 100, 500, 1000]:  
            torch.save(self.state_dict(), 'model_{}.ckpt'.format(epoch))
            
    def training_epoch_end(self, train_step_outputs):
        epoch_tloss = sum([value['train_batch_loss'] for value in train_step_outputs])/self.len_train
        print('-------- Current Epoch {} --------'.format(self.current_epoch + 1))
        print('train Loss: {:e}'.format(epoch_tloss))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.002)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.max_lr, epochs = self.epochs, steps_per_epoch=self.steps_per_epoch)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}  