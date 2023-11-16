'''
Code snippets ported from:
https://huggingface.co/blog/annotated-diffusion
https://github.com/lucidrains/denoising-diffusion-pytorch
https://github.com/hojonathanho/diffusion
'''

import argparse
from math import log
import torch
from glob import glob
import pytorch_lightning as pl
import healpy as hp
from pytorch_lightning.loggers import TensorBoardLogger

from scripts.diffusion.diffusionclass import Diffusion
from scripts.diffusion.schedules import TimestepSampler, cosine_beta_schedule, linear_beta_schedule
from scripts.diffusion.ResUnet_timeembed import Unet, Unet_ref
from scripts.maploader.maploader import get_data, get_normalized_data, get_loaders, get_log2linear_transform
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
    params["data"]["transform_type"]: str = "both"
    params["data"]["upsample_scale"]: float = 2.0

    params["architecture"]["conditional"]: bool = True
    params["architecture"]["kernel_size"]: int = 8 
    params["architecture"]["dim_in"]: int = 1
    params["architecture"]["dim_out"]: int = 1
    params["architecture"]["inner_dim"]: int = 64
    params["architecture"]["mults"] = [1, 2, 4, 8]
    params["architecture"]["mask"]: bool = True

    params['diffusion']['timesteps']: int = 1000
    params['diffusion']['loss_type']: str = "huber"
    params['diffusion']['schedule']: str = "linear"
    params['diffusion']['linear_beta_start']: float = 10**(-6)
    params['diffusion']['linear_beta_end']: float = 10**(-2)
    params['diffusion']['cosine_beta_s']: float = 0.015
    params['diffusion']['sampler_type']: str = "uniform"

    params["train"]['target']: str = target
    params["train"]['train_rate']: float = 0.8
    params["train"]['batch_size']: int = 6
    params["train"]['learning_rate'] = 10**-4
    params["train"]['n_epochs']: int = 3000
    params["train"]['gamma']: float = 0.9999
    params["train"]['save_dir']: str = f"/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/ckpt_logs/diffusion/{params['train']['target']}/"
    params["train"]['log_name']: str = f"{params['train']['target']}_{params['diffusion']['schedule']}_{params['data']['transform_type']}_o{params['data']['order']}_b{params['train']['batch_size']}"
    params["train"]['patience']: int = 30
    params["train"]['save_top_k']: int = 1

    return params

def mask_with_gaussian(x, device, num_chunk=4):
    x = torch.chunk(x, num_chunk, dim=1)
    flags = torch.randint(2, size=(num_chunk,)).bool()
    # make sure at least one chunk is masked
    while flags.all():
        flags = torch.randint(2, size=(num_chunk,)).bool()
    x = [x[i] if flags[i] else torch.randn_like(x[i], device=device) for i in range(num_chunk)]
    x = torch.cat(x, dim=1)
    return x

class DDPM(pl.LightningModule):
    def __init__(self, model, params, sampler=None):
        super().__init__()
        self.save_hyperparameters(params)
        self.model = model(params)
        self.batch_size = params["train"]['batch_size']
        self.learning_rate = params["train"]['learning_rate']
        self.gamma = params["train"]['gamma']
        print("We are using Adam with lr = {}, gamma = {}".format(self.learning_rate, self.gamma))
        self.ifmask = params["architecture"]["mask"]

        timesteps = int(params['diffusion']['timesteps'])
        if params['diffusion']['schedule'] == "cosine":
            betas = cosine_beta_schedule(timesteps=timesteps, s=params['diffusion']['cosine_beta_s'])
            print("The schedule is cosine with s = {}".format(params['diffusion']['cosine_beta_s']))
        elif params['diffusion']['schedule'] == "linear":
            betas = linear_beta_schedule(timesteps=timesteps, beta_start=params['diffusion']['linear_beta_start'], beta_end=params['diffusion']['linear_beta_end'])
            print("The schedule is linear with beta_start = {}, beta_end = {}".format(params['diffusion']['linear_beta_start'], params['diffusion']['linear_beta_end']))
        self.diffusion = Diffusion(betas)
        self.loss_type = params['diffusion']['loss_type']

        self.sampler = sampler

    def training_step(self, batch, batch_idx):
        x, cond = batch
        if self.ifmask:
            x = mask_with_gaussian(x, device=self.device)
        t = self.sampler.get_timesteps(self.batch_size, self.current_epoch)
        loss = self.diffusion.p_losses(self.model, x, t, condition=cond, loss_type=self.loss_type)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, cond = batch
        if self.ifmask:
            x = mask_with_gaussian(x, device=self.device)
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
    print("LR data loaded from {}.  Number of maps: {}".format(parms["data"]["LR_dir"], parms["data"]["n_maps"]))

    hr = get_data(parms["data"]["HR_dir"], parms["data"]["n_maps"], parms["data"]["nside"], parms["data"]["order"], issplit=True)
    print("HR data loaded from {}.  Number of maps: {}".format(parms["data"]["HR_dir"], parms["data"]["n_maps"]))

    lr, transform_lr, inverse_transform_lr, range_min_lr, range_max_lr = get_normalized_data(lr, transform_type=parms["data"]["transform_type"])
    print("LR data normalized to [{},{}] by {} transform.".format(lr.min(), lr.max(), parms["data"]["transform_type"]))

    if args.target == 'difference':
        log2linear_transform, inverse_log2linear_transform = get_log2linear_transform()
        diff = log2linear_transform(hr) - log2linear_transform(inverse_transform_lr(lr))*(parms["data"]["upsample_scale"]**3)
        print("Difference data calculated from HR - LR*upsample_scale^3. min: {}, max: {}".format(diff.min(), diff.max()))
        diff, transforms_diff, inverse_transforms_diff, range_min_diff, range_max_diff = get_normalized_data(diff, transform_type=parms["data"]["transform_type"])
        print("Difference data normalized to [{},{}] by {} transform.".format(diff.min(), diff.max(), parms["data"]["transform_type"]))
        data_input, data_condition = diff, lr
    elif args.target == 'HR':
        hr, transforms_hr, inverse_transforms_hr, range_min_hr, range_max_hr = get_normalized_data(hr, transform_type=parms["data"]["transform_type"])
        print("HR data normalized to [{},{}] by {} transform.".format(hr.min(), hr.max(), parms["data"]["transform_type"]))
        data_input, data_condition = hr, lr
    else:
        raise ValueError("target must be 'difference' or 'HR'")
    
    print("data nside: {}, divided into {} patches, each patch has {} pixels.".format(parms["data"]["nside"], 12 * parms["data"]["order"]**2, hp.nside2npix(parms["data"]["nside"])//(12 * parms["data"]["order"]**2)))

    train_loader, val_loader = get_loaders(data_input, data_condition, parms["train"]['train_rate'], parms["train"]['batch_size'])
    print("train:validation = {}:{}, batch_size: {}".format(len(train_loader), len(val_loader), parms["train"]['batch_size']))

    #get sampler type
    sampler = TimestepSampler(timesteps=int(parms['diffusion']['timesteps']), sampler_type=parms['diffusion']['sampler_type'])
    print("sampler type: {}, timesteps: {}".format(parms['diffusion']['sampler_type'], parms['diffusion']['timesteps']))

    #get model
    model = DDPM(Unet_ref, parms, sampler = sampler)
    parms["train"]['log_name'] = parms["train"]['log_name'] + "_ref"

    logger = TensorBoardLogger(save_dir=parms["train"]['save_dir'], name=parms["train"]['log_name'])
    print("data saved in {}".format(parms["train"]['save_dir']))
    print("data name: {}".format(parms["train"]['log_name']))

    trainer = setup_trainer(logger=logger, fname=None, save_top_k=int(parms['train']['save_top_k']), max_epochs=int(parms['train']['n_epochs']), patience=int(parms['train']['patience']))
    trainer.fit(model, train_loader, val_loader)
    

