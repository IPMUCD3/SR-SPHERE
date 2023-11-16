'''
Code snippets ported from:
https://huggingface.co/blog/annotated-diffusion
https://github.com/lucidrains/denoising-diffusion-pytorch
https://github.com/hojonathanho/diffusion
'''

import argparse
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from scripts.utils.cheby_shev import SphericalChebConv
from scripts.diffusion.schedules import TimestepSampler
from scripts.diffusion.ResUnet_timeembed import Unet, ResnetBlock_t
from scripts.maploader.maploader import get_data, get_minmaxnormalized_data, get_loaders
from scripts.utils.run_utils import setup_trainer
from run.diffusion.run_diffusion import set_params, DDPM

# inherit set_params from run_diffusion.py
def set_params_concat(target="difference"):
    params = set_params(target=target)
    params["architecture"]["dim_in"]: int = 2
    params["architecture"]["dim_out"]: int = 1
    params["architecture"]["conditioning"]: str = "concat"
    params["train"]['log_name']: str = f"{params['train']['target']}_{params['diffusion']['schedule']}_o{params['data']['order']}_b{params['train']['batch_size']}_concat"
    return params

class Unet_concat(Unet):
    def __init__(self, params):
        super().__init__(params)
        self.dim_out = params["architecture"]["dim_out"]
        self.final_conv = nn.Sequential(
            ResnetBlock_t(self.dim, self.dim, self.laps[-1], self.kernel_size, self.time_dim), 
            SphericalChebConv(self.dim, self.dim_out, self.laps[-1], self.kernel_size)
        )

    def forward(self, x, time, condition=None):
        t = self.time_mlp(time) if self.time_mlp is not None else None

        if condition is not None:
            x = torch.cat([x, condition], dim=2)

        x = self.init_conv(x)

        skip_connections = []

        # downsample
        for block1, block2, downsample in self.down_blocks:
            x = block1(x, t)
            x = block2(x, t)
            skip_connections.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, upsample  in self.up_blocks:
            tmp_connection = skip_connections.pop()
            x = torch.cat([x, tmp_connection], dim=2)
            x = block2(block1(x, t), t)
            x = upsample(x)

        return self.final_conv(x)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--target', type=str, default='difference')
    args = args.parse_args()

    pl.seed_everything(1234)
    parms = set_params_concat(target=args.target)

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
    

