
import torch
import pytorch_lightning as pl
import random

from srsphere.diffusion.utils import extract_masked_region, hr_or_sample
from srsphere.diffusion.diffusion import Diffusion

class DDPM(pl.LightningModule):
    def __init__(self, model, sampler, **args):
        super().__init__()
        self.learning_rate = args['learning_rate']
        self.gamma = args['gamma']
        #print("We are using Adam with lr = {}, gamma = {}".format(self.learning_rate, self.gamma))
        
        self.ifmask = args["mask"]
        if self.ifmask:
            self.masks = [[1, 1, 1, 1], [0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 0, 1]]

        #self.save_hyperparameters()
        self.diffusion = Diffusion(**args)
        self.model = model
        self.sampler = sampler

    def training_step(self, batch, batch_idx):
        cond, x = batch
        t = self.sampler.get_timesteps(x.shape[0], self.current_epoch)
        if self.ifmask:
            mask = random.choice(self.masks)
            noise = torch.randn_like(x)
            x_t = self.diffusion.q_sample(x_start=x, t=t, noise=noise)
            x_masked = hr_or_sample(x, x_t, mask)
            predicted_noise = self.model(x_masked, t, condition=cond)
            corr_noise = extract_masked_region(noise, mask)
            corr_pred = extract_masked_region(predicted_noise, mask)
            loss = self.diffusion.loss_fn(corr_noise, corr_pred)
        else:
            loss = self.diffusion.p_losses(self.model, x, t, condition=cond)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        cond, x = batch
        t = self.sampler.get_timesteps(x.shape[0], self.current_epoch)
        if self.ifmask:
            mask = random.choice(self.masks)
            noise = torch.randn_like(x)
            x_t = self.diffusion.q_sample(x_start=x, t=t, noise=noise)
            x_masked = hr_or_sample(x, x_t, mask)
            predicted_noise = self.model(x_masked, t, condition=cond)
            corr_noise = extract_masked_region(noise, mask)
            corr_pred = extract_masked_region(predicted_noise, mask)
            loss = self.diffusion.loss_fn(corr_noise, corr_pred)
        else:
            loss = self.diffusion.p_losses(self.model, x, t, condition=cond)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler} 