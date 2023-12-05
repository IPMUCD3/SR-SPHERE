'''
Code snippets ported from:
https://huggingface.co/blog/annotated-diffusion
https://github.com/lucidrains/denoising-diffusion-pytorch
https://github.com/hojonathanho/diffusion
'''

import pytorch_lightning as pl

from scripts.diffusion.models.Unet_base import Unet
from scripts.diffusion.DSM import DSM, VESDE
from scripts.maploader.maploader import get_loaders_from_params
from scripts.utils.run_utils import setup_trainer, get_parser
from scripts.utils.params import set_params

if __name__ == '__main__':
    args = get_parser()
    args = args.parse_args()

    pl.seed_everything(1234)
    params = set_params(**vars(args))
    params['diffusion']['sigmas'] = [0.01, 50]
    params['diffusion']['sigma_N'] = 1000
    params['diffusion']['continuous'] = True
    params['diffusion']['likelihood_weighting'] = False
    params['diffusion']['eps'] = 1e-5
    params["train"]['log_name'] = "SDE_test"

    ### get training data
    train_loader, val_loader = get_loaders_from_params(params)

    #get sampler
    sde = VESDE(sigma_min=params['diffusion']['sigmas'][0], sigma_max=params['diffusion']['sigmas'][1], N=params['diffusion']['timesteps'])

    #get model
    model = DSM(Unet, params, sde)

    trainer = setup_trainer(params)
    trainer.fit(model, train_loader, val_loader)