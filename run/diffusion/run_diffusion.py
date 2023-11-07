'''
Code snippets ported from:
https://huggingface.co/blog/annotated-diffusion
https://github.com/lucidrains/denoising-diffusion-pytorch
https://github.com/hojonathanho/diffusion
'''

import argparse
import torch
from glob import glob
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from scripts.diffusion.diffusionclass import Diffusion
from scripts.diffusion.schedules import TimestepSampler, cosine_beta_schedule, linear_beta_schedule
from scripts.diffusion.ResUnet_timeembed import Unet
from scripts.maploader.maploader import get_data, get_minmaxnormalized_data, get_loaders
from scripts.utils.run_utils import setup_trainer

def set_params(target="difference"):
    params = {}
    params["data"] = {}
    params["architecture"] = {}
    params["train"] = {}
    params["diffusion"] = {}

    params["data"]["HR_dir"]: str = "/gpfs02/work/akira.tokiwa/gpgpu/FastPM/healpix/nc256/"
    params["data"]["LR_dir"]: str = "/gpfs02/work/akira.tokiwa/gpgpu/FastPM/healpix/nc128/"
    params["data"]["n_maps"]: int = len(glob(params["data"]["LR_dir"] + "*.fits"))
    params["data"]["nside"]: int = 512
    params["data"]["order"]: int = 2

    params["architecture"]["conditional"]: bool = True
    params["architecture"]["kernel_size"]: int = 8 
    params["architecture"]["norm_groups"]: int = 8
    params["architecture"]["dim_in"]: int = 1
    params["architecture"]["inner_dim"]: int = 64
    params["architecture"]["mults"] = [1, 2, 4, 8]

    params['diffusion']['timesteps']: int = 1000
    params['diffusion']['loss_type']: str = "huber"
    params['diffusion']['schedule']: str = "linear"
    params['diffusion']['sampler_type']: str = "uniform"

    params["train"]['target']: str = target
    params["train"]['train_rate']: float = 0.8
    params["train"]['batch_size']: int = 6
    params["train"]['learning_rate'] = 10**-6
    params["train"]['n_epochs']: int = 300
    params["train"]['gamma']: float = 0.9999
    params["train"]['save_dir']: str = f"/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/ckpt_logs/diffusion/{params['train']['target']}/"
    params["train"]['log_name']: str = f"{params['train']['target']}_{params['diffusion']['schedule']}_o{params['data']['order']}_b{params['train']['batch_size']}"
    params["train"]['patience']: int = 3
    params["train"]['save_top_k']: int = 3

    return params

class DDPM(pl.LightningModule):
    def __init__(self, model, params, sampler=None):
        super().__init__()
        self.save_hyperparameters(params)
        self.model = model(params)
        self.batch_size = params["train"]['batch_size']
        self.learning_rate = params["train"]['learning_rate']
        self.gamma = params["train"]['gamma']

        timesteps = int(params['diffusion']['timesteps'])
        if params['diffusion']['schedule'] == "cosine":
            betas = cosine_beta_schedule(timesteps=timesteps, s=0.015)
        elif params['diffusion']['schedule'] == "linear":
            betas = linear_beta_schedule(timesteps=timesteps, beta_start=0.0001, beta_end=0.02)
        self.diffusion = Diffusion(betas)
        self.loss_type = params['diffusion']['loss_type']

        self.sampler = sampler

    def training_step(self, batch, batch_idx):
        x, cond = batch
        t = self.sampler.get_timesteps(self.batch_size, self.current_epoch)
        loss = self.diffusion.p_losses(self.model, x, t, condition=cond, loss_type=self.loss_type)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, cond = batch
        t = self.sampler.get_timesteps(self.batch_size, self.current_epoch)
        loss = self.diffusion.p_losses(self.model, x, t, condition=cond, loss_type=self.loss_type)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}  

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--target', type=str, default='difference')
    args = args.parse_args()

    pl.seed_everything(1234)
    parms = set_params(target=args.target)

    lr = get_data(parms["data"]["LR_dir"], parms["data"]["n_maps"], parms["data"]["nside"], parms["data"]["order"], issplit=True)
    hr = get_data(parms["data"]["HR_dir"], parms["data"]["n_maps"], parms["data"]["nside"], parms["data"]["order"], issplit=True)

    hr, transforms_hr, inverse_transforms_hr, range_min_hr, range_max_hr = get_minmaxnormalized_data(hr)
    print("HR data loaded. min: {}, max: {}".format(range_min_hr, range_max_hr))
    print("HR data normalized. min: {}, max: {}".format(hr.min(), hr.max()))

    if args.target == 'difference':
        lr = transforms_hr(lr)
        print("LR data normalized by HR range. min: {}, max: {}".format(lr.min(), lr.max()))
        data_input, data_condition = hr-lr, lr
    elif args.target == 'HR':
        lr, transform_lr, inverse_transform_lr, range_min_lr, range_max_lr = get_minmaxnormalized_data(lr)
        print("LR data loaded. min: {}, max: {}".format(range_min_lr, range_max_lr))
        data_input, data_condition = hr, lr
    else:
        raise ValueError("target must be 'difference' or 'HR'")

    train_loader, val_loader = get_loaders(data_input, data_condition, parms["train"]['train_rate'], parms["train"]['batch_size'])

    #get sampler type
    sampler = TimestepSampler(timesteps=int(parms['diffusion']['timesteps']), sampler_type=parms['diffusion']['sampler_type'])

    #get model
    model = DDPM(Unet, parms, sampler = sampler)

    logger = TensorBoardLogger(save_dir=parms["train"]['save_dir'], name=parms["train"]['log_name'])
    trainer = setup_trainer(logger=logger, fname=None, save_top_k=int(parms['train']['save_top_k']), max_epochs=int(parms['train']['n_epochs']), patience=int(parms['train']['patience']))
    trainer.fit(model, train_loader, val_loader)
    

