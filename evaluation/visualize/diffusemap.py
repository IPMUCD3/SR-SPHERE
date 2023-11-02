import torch
import pytorch_lightning as pl
import torch.utils.data as data
from tqdm.auto import tqdm
import os
import healpy as hp
import numpy as np

from scripts.diffusion.diffusionclass import Diffusion
from scripts.diffusion.schedules import TimestepSampler, linear_beta_schedule
from scripts.diffusion.ResUnet_timeembed import Unet
from run.run_diffusion import Unet_pl
from scripts.utils.run_utils import initialize_config
from scripts.maploader.maploader import get_data, get_minmaxnormalized_data, get_loaders

if __name__ == '__main__':
    config_file = "/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/config/config_diffusion.yaml"
    config_dict = initialize_config(config_file)

    pl.seed_everything(1234)

    ### get training data
    config_dict['data']['lrmaps_dir'] = "/gpfs02/work/akira.tokiwa/gpgpu/FastPM/healpix/nc128/"
    config_dict['data']['hrmaps_dir'] = "/gpfs02/work/akira.tokiwa/gpgpu/FastPM/healpix/nc256/"
    config_dict['data']['nside_lr'] = 512
    config_dict['data']['nside_hr'] = 512
    config_dict['data']["normalize"] = False
    config_dict['data']['order'] = 4
    config_dict['train']['batch_size'] = 48

    CONDITIONAL = bool(config_dict['data']['conditional'])
    BATCH_SIZE = config_dict['train']['batch_size']
    PATCH_SIZE = 12 * (config_dict['data']['order'])**2

    lr = get_data(config_dict['data']['lrmaps_dir'], config_dict['data']['n_maps'], config_dict['data']['nside_lr'], config_dict['data']['order'], issplit=True)
    hr = get_data(config_dict['data']['hrmaps_dir'], config_dict['data']['n_maps'], config_dict['data']['nside_hr'], config_dict['data']['order'], issplit=True)

    #lr, transforms_lr, nverse_transforms_lr, range_min_lr, range_max_lr = get_minmaxnormalized_data(lr)
    #print("LR data loaded. min: {}, max: {}".format(range_min_lr, range_max_lr))

    hr, transforms_hr, inverse_transforms_hr, range_min_hr, range_max_hr = get_minmaxnormalized_data(hr)
    print("HR data loaded. min: {}, max: {}".format(range_min_hr, range_max_hr))

    lr = transforms_hr(lr)
    print("LR data normalized by HR range. min: {}, max: {}".format(lr.min(), lr.max()))

    data_input, data_condition = hr-lr, lr

    # set up model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ckpt_path= "/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/ckpt_logs/diffusion/HR_normalized/version_1/checkpoints/Run_10-31_11-16epoch=06-val_loss=0.04.ckpt"

    #get sampler type
    timesteps = int(config_dict['diffusion']['timesteps'])
    sampler = TimestepSampler(timesteps=timesteps, **config_dict['diffusion']['sampler_args'])

    #get model
    model = Unet_pl(Unet, config_dict, sampler = sampler).to(device)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    beta_func = linear_beta_schedule
    beta_args = config_dict['diffusion']['schedule_args']
    betas = beta_func(timesteps=timesteps, **beta_args)
    tmp_diffusion = Diffusion(betas)

    map_dir = "/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/results/imgs/diffusion/HR_normalized/"
    if not os.path.exists(map_dir):
        os.makedirs(map_dir)

    print("Start Diffusion")
    for i in range(int(PATCH_SIZE/BATCH_SIZE)):
        tmp_sample =data_input[BATCH_SIZE*i:BATCH_SIZE*(i+1)].to(device)
        tmp_lr = data_condition[BATCH_SIZE*i:BATCH_SIZE*(i+1)].to(device)
        q_sample = tmp_diffusion.q_sample(tmp_sample, torch.full((BATCH_SIZE,), timesteps-1, device=device))
        img = torch.randn(tmp_sample.shape, device=device)
        with torch.no_grad():
            for j in reversed(range(0, timesteps)):
                t = torch.full((BATCH_SIZE,), j, device=device, dtype=torch.long)
                loss = model.diffusion.p_losses(model.model, tmp_sample, t, tmp_lr)
                img = tmp_diffusion.p_sample(model.model, img, t, tmp_lr, j)
                print('Step {}, Loss {}'.format(j, loss), flush=True)
                if (j % 10 == 0):
                    diffmap = np.hstack(img.detach().cpu().numpy()[:BATCH_SIZE, : , 0])
                    np.save(map_dir+f"diffused_step_{str(j).zfill(3)}_batch_{i}.npy", diffmap)