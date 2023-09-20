
import os
import sys
import glob
import healpy as hp
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.transforms import Compose, ToTensor, Normalize

# Local module imports
sys.path.append('/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/')
from srsphere.data.maploader import get_minmax_transform, hp_split
from srsphere.tests.params import get_params
from srsphere.models.VGG import VGG_cosmo
from srsphere.models.model_template import template


def setup_trainer(params, logger=None):
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=30,
        verbose=0,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        save_last=False,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=params['num_epochs'],
        callbacks=[checkpoint_callback, early_stop_callback],
        num_sanity_val_steps=0,
        accelerator='gpu', devices=1,
        logger=logger
    )
    return trainer

def train_model(model_class, params, logger=None):
    train_loader, val_loader = setloaders(params)
    model = model_for_pl(model_class, params)
    if logger is None:
        logger = TensorBoardLogger(save_dir='.')
    trainer = setup_trainer(params, logger)
    trainer.fit(model, train_loader, val_loader)
    return model

def setloaders(params):
    maps = sorted(glob.glob(f'{params["hrmaps_dir"]}*.fits'))
    dmaps = [hp.read_map(dmap) for dmap in maps]
    data = np.vstack([dmap for dmap in dmaps])
    data = np.vstack([hp_split(el, order=params["order"]) for el in data])
    shape = (params["n_maps"]*params["order"]**2*12, (params["nside_hr"]//params["order"])**2, 1)
    data_hr = ToTensor()(data).view(*shape).float()

    RANGE_MIN, RANGE_MAX = data_hr.min().clone().detach(), data_hr.max().clone().detach()
    transforms, inverse_transforms = get_minmax_transform(RANGE_MIN, RANGE_MAX)

    cosmofile = params["cosmofile"]
    with open(cosmofile, "r") as f:
        lines = f.readlines()

    cosmo=[]
    seed_list=[]
    for line in lines:
        seed, Omega0_b, Omega0_cdm, h, sigma8 = map(float, line.strip().split())
        seed_list.append(seed)
        cosmo.append([Omega0_b, Omega0_cdm, h, sigma8])

    cosmo = np.array(cosmo)
    cosmo = cosmo[np.argsort(seed_list)]
    cosmotensor = torch.from_numpy(cosmo).float().repeat_interleave(params["order"]**2*12, dim=0)

    combined_dataset = torch.utils.data.TensorDataset(transforms(data_hr), cosmotensor)
    train, val = torch.utils.data.random_split(combined_dataset, [int(params["rate_train"]*len(combined_dataset)), len(combined_dataset)-int(params["rate_train"]*len(combined_dataset))])
    train_loader = torch.utils.data.DataLoader(train, batch_size=params["batch_size"], shuffle=True, num_workers=os.cpu_count())
    val_loader = torch.utils.data.DataLoader(val, batch_size=params["batch_size"], shuffle=False, num_workers=os.cpu_count())

    return train_loader, val_loader

class model_for_pl(pl.LightningModule):
    def __init__(self, model_class, params):
        super().__init__()
        self.lr_init = params['lr_init']
        self.lr_max = params['lr_max']
        self.epochs = params['num_epochs']
        self.steps_per_epoch = params['steps_per_epoch']
        self.loss_fn = nn.MSELoss()
        self.model = model_class(params)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss =  self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss =  self.loss_fn(y_hat, y)
        self.log('val_loss', loss, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr_init)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}  

if __name__ == "__main__":
    datadir = "/gpfs02/work/akira.tokiwa/gpgpu/FastPM" 

    params = get_params()
    params["hrmaps_dir"] = f"{datadir}/healpix/nc256/"
    params["lrmap_dir"] = f"{datadir}/healpix/nc128/"
    params["nside_hr"] = 512
    params["nside_lr"] = 512
    params["cosmofile"] = f"{datadir}/cosmoparams.txt"
    params["order"] = 2
    params['batch_size'] = 1

    train_model(VGG_cosmo, params)