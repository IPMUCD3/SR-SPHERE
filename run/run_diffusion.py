'''
Code snippets ported from:
https://huggingface.co/blog/annotated-diffusion
https://github.com/lucidrains/denoising-diffusion-pytorch
https://github.com/hojonathanho/diffusion
'''

import os   
import sys
import yaml
import torch
import datetime
import pickle
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.append('/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/scripts')
from diffusion.diffusionclass import Diffusion
from diffusion.schedules import TimestepSampler, linear_beta_schedule
from diffusion.ResUnet_timeembed import Unet
from maploader.maploader import MapDataset, get_minmax_transform


def prepare_data(params):
    lr_set = MapDataset(params['lrmaps_dir'], params['n_maps'],  params['nside_lr'], params['order'], params['issplit'], params["normalize"])
    hr_set = MapDataset(params['hrmaps_dir'], params['n_maps'],  params['nside_hr'], params['order'],params['issplit'], params["normalize"])
    lr_set.maps = sorted(lr_set.maps)
    hr_set.maps = sorted(hr_set.maps)
    lr, hr = lr_set.__getitem__(0), hr_set.__getitem__(0)
    return lr, hr

def normalize_data(input_data):
    RANGE_MIN, RANGE_MAX = input_data.min().clone().detach(), input_data.max().clone().detach()
    transforms, inverse_transforms = get_minmax_transform(RANGE_MIN, RANGE_MAX)
    norm_input = transforms(input_data)
    return norm_input, transforms, inverse_transforms, RANGE_MIN, RANGE_MAX

def setup_trainer(params, logger=None):
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=0,
        mode="min"
    )

    dt = datetime.datetime.now()
    name = dt.strftime('Run_%m-%d_%H-%M')

    checkpoint_callback = ModelCheckpoint(
        filename= name + "{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        save_last=False,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=params["train"]['num_epochs'],
        callbacks=[checkpoint_callback, early_stop_callback],
        num_sanity_val_steps=0,
        accelerator='gpu', devices=1,
        logger=logger
    )
    return trainer

def initialize(config_file):
    with open(config_file, 'r') as stream:
        config_dict = yaml.safe_load(stream)

    config_dict["train"].update({"steps_per_epoch": int(config_dict['train']['num_epochs']) * int(config_dict["data"]['n_maps']) // int(config_dict['train']['batch_size'])})
    return config_dict

class Unet_pl(pl.LightningModule):
    def __init__(self, model, params, loss_type="huber", sampler=None, conditional=False):
        super().__init__()
        self.model = model(params)
        self.batch_size = params["train"]['batch_size']
        self.lr_init = float(params["train"]['lr_init'])
        self.lr_max = float(params["train"]['lr_max'])
        self.num_epochs = params["train"]['num_epochs']
        self.steps_per_epoch = params["train"]['steps_per_epoch']

        timesteps = int(params['diffusion']['timesteps'])
        beta_func = linear_beta_schedule
        beta_args = params['diffusion']['schedule_args']
        betas = beta_func(timesteps=timesteps, **beta_args)
        self.diffusion = Diffusion(betas)

        self.loss_type = loss_type
        self.sampler = sampler
        self.conditional = conditional
        self.loss_spike_flg = 0

    def training_step(self, batch, batch_idx):
        self.model.train(mode=True)

        if self.conditional:
            x, x_lr, labels = batch
        else:
            x, x_lr = batch

        t = self.sampler.get_timesteps(self.batch_size, self.current_epoch)
        loss = self.diffusion.p_losses(self.model, x, t, x_lr, loss_type=self.loss_type, labels=labels if self.conditional else None)
        self.log('train_loss', loss)

        if self.sampler.type == 'loss_aware':
            loss_timewise = self.diffusion.timewise_loss(self.model, x, t, x_lr, loss_type=self.loss_type, labels=labels if self.conditional else None)
            self.sampler.update_history(t, loss_timewise)

        if loss.item() > 0.1 and self.current_epoch > 300 and (self.loss_spike_flg < 2):
            badbdict = {'batch': batch.detach().cpu().numpy(), 'itn': self.current_epoch, 't': t.detach().cpu().numpy(), 'loss': loss.item()}
            pickle.dump(badbdict, open(f'largeloss_{self.current_epoch}.pkl', 'wb'))
            self.loss_spike_flg += 1

        return loss
    
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        if self.conditional:
            x, x_lr, labels = batch
        else:
            x, x_lr = batch

        t = self.sampler.get_timesteps(self.batch_size, self.current_epoch)
        loss = self.diffusion.p_losses(self.model, x, t, x_lr, loss_type=self.loss_type, labels=labels if self.conditional else None)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_init)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}  

    def on_train_epoch_end(self):
        if (self.current_epoch % 2000 == 0) & (self.current_epoch > 0):
            torch.save(self.state_dict(), f'checkpoint_{self.current_epoch}.pt')

if __name__ == '__main__':
    config_file = "/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/config/config_diffusion.yaml"
    config_dict = initialize(config_file)

    pl.seed_everything(1234)

    ### get training data
    config_dict['data']['lrmaps_dir'] = "/gpfs02/work/akira.tokiwa/gpgpu/FastPM/healpix/nc128/"
    config_dict['data']['hrmaps_dir'] = "/gpfs02/work/akira.tokiwa/gpgpu/FastPM/healpix/nc256/"
    config_dict['data']['nside_lr'] = 512
    config_dict['data']['nside_hr'] = 512
    config_dict['data']["normalize"] = False
    config_dict['data']['order'] = 4
    config_dict['train']['batch_size'] = 24

    CONDITIONAL = bool(config_dict['data']['conditional'])
    BATCH_SIZE = config_dict['train']['batch_size']

    lr, hr = prepare_data(config_dict['data'])
    input_data, transforms, inverse_transforms, RANGE_MIN, RANGE_MAX = normalize_data(hr - lr)
    condition = lr
    combined_dataset = torch.utils.data.TensorDataset(input_data, condition)
    len_train = int(config_dict['data']['rate_train'] * len(condition))
    len_val = len(condition) - len_train
    train, val = torch.utils.data.random_split(combined_dataset, [len_train, len_val])
    loaders = {x: torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=x=='train', num_workers=os.cpu_count()) for x, ds in zip(('train', 'val'), (train, val))}
    train_loader, val_loader= loaders['train'], loaders['val']

    #get sampler type
    sampler = TimestepSampler(timesteps=int(config_dict['diffusion']['timesteps']), **config_dict['diffusion']['sampler_args'])

    #get model
    model = Unet_pl(Unet, config_dict, sampler = sampler)

    logger = TensorBoardLogger(save_dir='/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/ckpt_logs/diffusion', name='HR_LR_normalized')
    trainer = setup_trainer(config_dict, logger=logger)
    trainer.fit(model, train_loader, val_loader)
    

