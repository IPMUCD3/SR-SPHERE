
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import random
from tqdm.auto import tqdm

from scripts.utils.diffusion_utils import extract_masked_region, hr_or_sample
from scripts.utils.diffusion_utils import extract

'''
Code snippets ported from:
https://huggingface.co/blog/annotated-diffusion
'''

def linear_beta_schedule(timesteps, beta_start, beta_end):
    #beta_start = 0.0001
    #beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps, beta_start, beta_end):
    #beta_start = 0.0001
    #beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps, beta_start, beta_end):
    #beta_start = 0.0001
    #beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def cosine_beta_schedule(timesteps, s=0.015):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0, 0.999) 

#based off https://github.com/openai/improved-diffusion/blob/783b6740edb79fdb7d063250db2c51cc9545dcd1/improved_diffusion/resample.py
class TimestepSampler():
    def __init__(self, timesteps, sampler_type="uniform"):
        self.timesteps = timesteps
        self.sampler_type = sampler_type
        print("sampler type: {}, timesteps: {}".format(self.sampler_type, self.timesteps))

    def get_timesteps(self, batch_size, iteration):
        return torch.randint(0, self.timesteps, (batch_size,), device='cuda').long()

class Diffusion():
    def __init__(self, **args):
        self.timesteps = args['timesteps']
        if args['schedule'] == "cosine":
            betas = cosine_beta_schedule(timesteps=self.timesteps, s=args['cosine_beta_s'])
            print("The schedule is cosine with s = {}".format(args['cosine_beta_s']))
        elif args['schedule'] == "linear":
            betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=args['linear_beta_start'], beta_end=args['linear_beta_end'])
            print("The schedule is linear with beta_start = {}, beta_end = {}".format(args['linear_beta_start'], args['linear_beta_end']))

        self.betas = betas
        self.alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)  # alpha_bar
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        # y_t = sqrt_alphas_cumprod* x_0 + sqrt_one_minus_alphas_cumprod * eps_t
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        #self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_variance = self.betas
        # y_t-1 = sqrt_recip_alphas * (y_t - betas_t * MODEL(x_t, t) / sqrt_one_minus_alphas_cumprod_t)
        #         + sqrt_one_minus_alphas_cumprod_t * eps_t

        self.loss_type = args["loss_type"]
        if self.loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif self.loss_type == 'l2':
            self.loss_fn = F.mse_loss
        elif self.loss_type == "huber":
            self.loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError()

    # forward diffusion (using the nice property)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_start, t, noise=None, condition=None):
        # L_CE <= L_VLB ~ Sum[eps_t - MODEL(x_t(x_0, eps_t), t) ]
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_t, t, condition=condition)
        loss = self.loss_fn(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def timewise_loss(self, denoise_model, x_start, t, noise=None, condition=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_t, t, condition=condition)
        loss = self.loss_fn(noise, predicted_noise)
        loss = torch.mean(loss, dim=[-2, -1]) #mean over all spatial dims
        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, condition=None):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_output = model(x, t) if condition is None else model(x, t, condition=condition)
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + posterior_variance_t * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, condition=None):
        device = next(model.parameters()).device
        print('sample device', device)
        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []
        if condition is not None:
            assert condition.shape[0] == shape[0]

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, condition=condition)
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=1, condition=None):
        return self.p_sample_loop(model, shape=(batch_size, image_size, channels), condition=condition)
    
class DDPM(pl.LightningModule):
    def __init__(self, model, sampler, **args):
        super().__init__()
        self.learning_rate = args['learning_rate']
        self.gamma = args['gamma']
        print("We are using Adam with lr = {}, gamma = {}".format(self.learning_rate, self.gamma))
        
        self.ifmask = args["mask"]
        if self.ifmask:
            self.masks = [[1, 1, 1, 1], [0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 0, 1]]
            print("We are using mask with gaussian noise")

        self.save_hyperparameters(ignore=['model'])
        self.diffusion = Diffusion(**args)
        self.model = model
        self.sampler = sampler

    def training_step(self, batch, batch_idx):
        cond, x = batch
        t = self.sampler.get_timesteps(x.shape[0], self.current_epoch)
        if self.ifmask:
            mask = [random.choice(self.masks) for _ in range(x.shape[0])]
            noise = torch.randn_like(x)
            x_t = self.diffusion.q_sample(x_start=x, t=t, noise=noise)
            x_masked = torch.cat([hr_or_sample(x[i:i+1, :, :], x_t[i:i+1, :, :], mask[i]) for i in range(x.shape[0])], dim=0)
            predicted_noise = self.model(x_masked, t, condition=cond)

            corr_noise = torch.cat([extract_masked_region(noise[i:i+1, :, :], mask[i]) for i in range(x.shape[0])], dim=1)
            corr_pred = torch.cat([extract_masked_region(predicted_noise[i:i+1, :, :], mask[i]) for i in range(x.shape[0])], dim=1)
            loss = self.diffusion.loss_fn(corr_noise, corr_pred)
        else:
            loss = self.diffusion.p_losses(self.model, x, t, condition=cond)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        cond, x = batch
        t = self.sampler.get_timesteps(x.shape[0], self.current_epoch)
        if self.ifmask:
            mask = [random.choice(self.masks) for _ in range(x.shape[0])]
            noise = torch.randn_like(x)
            x_t = self.diffusion.q_sample(x_start=x, t=t, noise=noise)
            x_masked = torch.cat([hr_or_sample(x[i:i+1, :, :], x_t[i:i+1, :, :], mask[i]) for i in range(x.shape[0])], dim=0)
            predicted_noise = self.model(x_masked, t, condition=cond)

            corr_noise = torch.cat([extract_masked_region(noise[i:i+1, :, :], mask[i]) for i in range(x.shape[0])], dim=1)
            corr_pred = torch.cat([extract_masked_region(predicted_noise[i:i+1, :, :], mask[i]) for i in range(x.shape[0])], dim=1)
            loss = self.diffusion.loss_fn(corr_noise, corr_pred)
        else:
            loss = self.diffusion.p_losses(self.model, x, t, condition=cond)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler} 