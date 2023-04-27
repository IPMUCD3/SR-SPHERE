from lib.maploader import MapDataset, get_loaders
from lib.partiallib import get_partial_laplacians
from lib.cheby_shev import SphericalChebConv
from lib.healpix_pool_unpool import Healpix
from lib.spherical_unet_utils import SphericalCheb
from lib.healpix_pixelshuffle import patch_PSUB

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
import torch
import torch.nn as nn
import torch.optim as optim


def get_params(basedir = "./data/" ,order=2, verbose=True):
    """Parameters for the models"""

    params = dict()
    
    #data info
    params["hrmaps_dir"]=basedir + "HR/map/"
    params["lrmaps_dir"]=basedir + "LR/map/"
    params['nside_hr'] = 64
    params['nside_lr'] = 32
    
    params['rate_train'] = 0.8
    params['n_maps'] = 100
    params['N_train'] = int(params['n_maps']*params['rate_train'])
    params['N_val'] = int(params['n_maps'] - params['N_train'])
    
    #spliting data
    params['order'] = 2
    params['issplit'] = True
    
    #Architecture
    params['depth'] = 2
    params['kernel_size'] = 30
    
    # Training.
    params['num_epochs'] = 1000  # Number of passes through the training data.
    params['batch_size'] = 3*16   # Constant quantity of information (#pixels) per step (invariant to sample size).
    params['steps_per_epoch'] = params['num_epochs'] * params['N_train'] // params['batch_size']
    params['lr_init'] = 2*10**-4
    params['lr_max'] = 1*10**-3
    
    if params['issplit']:
        params['N_train'] = int(params['n_maps']*(12 * order**2)*params['rate_train'])
        params['N_val'] = int(params['n_maps'] - params['N_train'])
        params['batch_size'] = 3*16 * order ** 2
    
    if verbose:
        print('#LRsides: {0}, HRsides: {1} '.format(params['nside_lr'], params['nside_hr']))
        print('#LRpixels: {0} / {1} = {2}'.format(12*params['nside_lr']**2, 12*order**2, (params['nside_lr']//order)**2))
        # Number of pixels on the full sphere: 12 * nside**2.

        print('#samples per batch: {}'.format(params['batch_size']))
        print('=> #pixels per batch (input): {:,}'.format(params['batch_size']*(params['nside_lr']//order)**2))
        print('=> #pixels for training (input): {:,}'.format(params['num_epochs']*params['N_train']*(params['nside_lr']//order)**2))
        print('Learning rate will start at {0:.1e} and reach maximum at {1:.1e}.'.format(params['lr_init'], params['lr_max']))

    return params

class simplenetwork(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        
        self.kernel_size = params['kernel_size']
        self.N_val = params['N_val']
        self.N_train = params['N_train']
        self.epochs = params['num_epochs']
        self.steps_per_epoch = params['steps_per_epoch']
        self.order = params['order']
        self.lr_init = params['lr_init']
        self.lr_max = params['lr_max']
    
        
        self.laps = get_partial_laplacians(params['nside_hr'], params['depth'], self.order, "normalized")
        self.pooling = Healpix().pooling
        self.unpooling = Healpix().unpooling
        self.loss_fn = nn.MSELoss()
        
        self.spherical_cheb_in = SphericalCheb(1, 64, self.laps[0], self.kernel_size)
        self.PSUB = patch_PSUB(params['nside_hr'], 64, self.laps[0], self.laps[1], self.kernel_size, self.order)
        self.spherical_cheb_out = SphericalChebConv(64, 1, self.laps[1], self.kernel_size)
        
    def forward(self, x):
        x = self.spherical_cheb_in(x)
        x = self.PSUB(x)
        x = self.spherical_cheb_out(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, on_epoch=True)
        return {'loss': loss}
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr_init)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr_max, epochs = self.epochs, steps_per_epoch=self.steps_per_epoch)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}  

"""
    def on_validation_epoch_end(self):
        epoch_vloss = sum([value['val_batch_loss'] for value in self.validation_step_outputs])/self.N_val
        print('valid Loss: {:e}'.format(epoch_vloss))     
        epoch = self.current_epoch+1
        # Save state dict of model [10, 50, 100, 500, 1000] epoch
        if epoch in [10, 50, 100, 500, 1000]:  
            torch.save(self.state_dict(), 'model_{}.ckpt'.format(epoch))
            
    def on_train_epoch_end(self):
        epoch_tloss = sum([value['train_batch_loss'] for value in self.training_step_outputs])/self.N_train
        print('-------- Current Epoch {} --------'.format(self.current_epoch + 1))
        print('train Loss: {:e}'.format(epoch_tloss))
"""

    

if __name__ == "__main__":
    params = get_params()
    train_loader, val_loader = get_loaders(params['hrmaps_dir'], params['lrmaps_dir'], params['n_maps'], 
                                       params['nside_hr'], params['nside_lr'], params['rate_train'], 
                                       params['batch_size'], params['issplit'], params['order'])
    model_dir = "./pl-checkpoints"
    model = simplenetwork(params)
    early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=10,
    verbose=1,
    mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
    dirpath=model_dir,
    filename="best",
    save_top_k=1,
    monitor="val_loss",
    mode="min"
    )

    # Define the PyTorch-Lightning trainer
    trainer = pl.Trainer(
        max_epochs=params['num_epochs'],
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0
    )
    trainer.fit(model, train_loader, val_loader)