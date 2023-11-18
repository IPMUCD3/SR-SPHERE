'''
Code snippets ported from:
https://huggingface.co/blog/annotated-diffusion
https://github.com/lucidrains/denoising-diffusion-pytorch
https://github.com/hojonathanho/diffusion
'''

import argparse
import torch
import pytorch_lightning as pl

from scripts.diffusion.diffusionclass import Diffusion
from scripts.diffusion.schedules import TimestepSampler, cosine_beta_schedule, linear_beta_schedule
from scripts.diffusion.ResUnet_timeembed import Unet, Unet_bg
from scripts.maploader.maploader import get_loaders_from_params
from scripts.utils.run_utils import setup_trainer, set_params
from scripts.utils.diffusion_utils import mask_with_gaussian

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
    args.add_argument('--model', type=str, default='diffusion')
    args.add_argument('--target', type=str, default='difference')
    args.add_argument('--scheduler', type=str, default='linear')
    args = args.parse_args()

    pl.seed_everything(1234)
    parms = set_params(target=args.target, model=args.model, scheduler=args.scheduler)

    ### get training data
    train_loader, val_loader = get_loaders_from_params(parms)

    #get sampler type
    sampler = TimestepSampler(
        timesteps=int(parms['diffusion']['timesteps']), 
        sampler_type=parms['diffusion']['sampler_type'])
    print("sampler type: {}, timesteps: {}".format(parms['diffusion']['sampler_type'], parms['diffusion']['timesteps']))

    #get model
    model = DDPM(Unet_bg, parms, sampler = sampler)

    trainer = setup_trainer(parms)
    trainer.fit(model, train_loader, val_loader)
    

