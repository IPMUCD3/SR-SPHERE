'''
Code snippets ported from:
https://huggingface.co/blog/annotated-diffusion
https://github.com/lucidrains/denoising-diffusion-pytorch
https://github.com/hojonathanho/diffusion
'''

import sys
import torch
import pickle
from glob import glob
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.append('/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/scripts')
from diffusion.diffusionclass import Diffusion
from diffusion.schedules import TimestepSampler, linear_beta_schedule
from diffusion.ResUnet_timeembed import Unet
from maploader.maploader import get_data, get_minmaxnormalized_data, get_loaders
from utils.run_utils import initialize_config, setup_trainer

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
    config_dict = initialize_config(config_file)

    pl.seed_everything(1234)

    ### get training data
    lrmaps_dir = "/gpfs02/work/akira.tokiwa/gpgpu/FastPM/healpix/nc128/"
    hrmaps_dir = "/gpfs02/work/akira.tokiwa/gpgpu/FastPM/healpix/nc256/"
    n_maps = len(glob(lrmaps_dir + "*.fits"))
    nside = 512
    order = 4

    CONDITIONAL = True
    BATCH_SIZE = 24
    TRAIN_SPLIT = 0.8

    config_dict['train']['batch_size'] = BATCH_SIZE
    config_dict["data"]["conditional"] = CONDITIONAL

    lr = get_data(lrmaps_dir, n_maps, nside, order, issplit=True)
    hr = get_data(hrmaps_dir, n_maps, nside, order, issplit=True)

    lr, inverse_transforms_lr, range_min_lr, range_max_lr = get_minmaxnormalized_data(lr)
    print("LR data loaded. min: {}, max: {}".format(range_min_lr, range_max_lr))

    hr, inverse_transforms_hr, range_min_hr, range_max_hr = get_minmaxnormalized_data(hr)
    print("HR data loaded. min: {}, max: {}".format(range_min_hr, range_max_hr))

    data_input, data_condition = hr-lr, lr
    train_loader, val_loader = get_loaders(data_input, data_condition, TRAIN_SPLIT, BATCH_SIZE)

    #get sampler type
    sampler = TimestepSampler(timesteps=int(config_dict['diffusion']['timesteps']), **config_dict['diffusion']['sampler_args'])

    #get model
    model = Unet_pl(Unet, config_dict, sampler = sampler)

    logger = TensorBoardLogger(save_dir='/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/ckpt_logs/diffusion', name='HR_LR_normalized')
    trainer = setup_trainer(logger=logger, fname=None, save_top_k=1, max_epochs=config_dict['train']['num_epochs'])
    trainer.fit(model, train_loader, val_loader)
    

