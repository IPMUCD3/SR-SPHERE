'''
Code snippets ported from:
https://huggingface.co/blog/annotated-diffusion
https://github.com/lucidrains/denoising-diffusion-pytorch
https://github.com/hojonathanho/diffusion
'''

import pytorch_lightning as pl

from scripts.diffusion.models.Unet_base import Unet
from scripts.diffusion.DDPM import DDPM, TimestepSampler
from scripts.maploader.maploader import get_loaders_from_params
from scripts.utils.run_utils import setup_trainer, get_parser
from scripts.utils.params import set_params

if __name__ == '__main__':
    args = get_parser()
    args = args.parse_args()

    pl.seed_everything(1234)
    parms = set_params(**vars(args))

    ### get training data
    train_loader, val_loader = get_loaders_from_params(parms)

    #get sampler type
    sampler = TimestepSampler(
        timesteps=int(parms['diffusion']['timesteps']), 
        sampler_type=parms['diffusion']['sampler_type'])
    print("sampler type: {}, timesteps: {}".format(parms['diffusion']['sampler_type'], parms['diffusion']['timesteps']))

    #get model
    model = DDPM(Unet, parms, sampler = sampler)

    trainer = setup_trainer(parms)
    trainer.fit(model, train_loader, val_loader)