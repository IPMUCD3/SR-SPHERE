
import torch
import pytorch_lightning as pl

from scripts.utils.diffusion_utils import mask_with_gaussian
from scripts.diffusion.diffusionclass import Diffusion
from scripts.diffusion.schedules import cosine_beta_schedule, linear_beta_schedule

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
        t = self.sampler.get_timesteps(x.shape[0], self.current_epoch)
        loss = self.diffusion.p_losses(self.model, x, t, condition=cond, loss_type=self.loss_type)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, cond = batch
        if self.ifmask:
            x = mask_with_gaussian(x, device=self.device)
        t = self.sampler.get_timesteps(x.shape[0], self.current_epoch)
        loss = self.diffusion.p_losses(self.model, x, t, condition=cond, loss_type=self.loss_type)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler} 