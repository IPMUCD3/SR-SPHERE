'''
Code snippets ported from:
https://huggingface.co/blog/annotated-diffusion
https://github.com/lucidrains/denoising-diffusion-pytorch
https://github.com/hojonathanho/diffusion
'''

import pytorch_lightning as pl

from scripts.diffusion.models.Unet_base import Unet
from scripts.diffusion.DDPM import DDPM, TimestepSampler
from scripts.maploader.datamodules import DataModule
from scripts.utils.run_utils import setup_trainer, get_parser
from scripts.utils.params import set_params

if __name__ == '__main__':
    args = get_parser()
    args = args.parse_args()

    pl.seed_everything(1234)
    params = set_params(**vars(args))

    ### get training data
    dm = DataModule(**params['data'])
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    #get sampler type
    sampler = TimestepSampler(timesteps=params['diffusion']['timesteps'])

    #get model
    unet = Unet(params['data']["nside"], params['data']["order"], **params['architecture'])
    model = DDPM(unet, sampler, **params['diffusion'])

    trainer = setup_trainer(**params['train'])
    trainer.fit(model, train_loader, val_loader)