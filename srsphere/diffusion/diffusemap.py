import sys
import torch
import pytorch_lightning as pl
from tqdm.auto import tqdm
import healpy as hp
import numpy as np

sys.path.append('/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE')
from srsphere.diffusion.diffusion import Diffusion
from srsphere.diffusion.schedules import TimestepSampler, linear_beta_schedule
from srsphere.diffusion.ResUnet_timeembed import Unet
from srsphere.data.maploader import get_datasets, transform_combine
from srsphere.diffusion.main import initialize, Unet_pl


if __name__ == '__main__':
    pl.seed_everything(1234)

    # set up model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ckpt_path= "/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/diffusion/lightning_logs/version_5/checkpoints/Run_09-04_18-05epoch=286-val_loss=0.03.ckpt"

    config_file = "/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/diffusion/params.yaml"
    config_dict = initialize(config_file)

    timesteps = int(config_dict['diffusion']['timesteps'])
    sampler = TimestepSampler(timesteps=timesteps, **config_dict['diffusion']['sampler_args'])

    ckpt = torch.load(ckpt_path)
    model = Unet_pl(Unet, config_dict, sampler = sampler).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    beta_func = linear_beta_schedule
    beta_args = config_dict['diffusion']['schedule_args']
    betas = beta_func(timesteps=timesteps, **beta_args)
    tmp_diffusion = Diffusion(betas)

    ### get training data
    n_maps = config_dict['data']['n_maps']
    map_dirs = [config_dict['data']['hrmaps_dir'], config_dict['data']['lrmaps_dir']]
    nsides = [config_dict['data']['nside_hr'], config_dict['data']['nside_lr']]
    data_lr, data_hr = get_datasets(map_dirs, n_maps, nsides, config_dict['data']['order'], config_dict['data']['issplit'], config_dict['data']['normalize'])
    combined_dataset = transform_combine(data_hr - data_lr, data_lr)

    for i in range(config_dict['data']['n_maps']):
        print(i)
        tmp_sample =combined_dataset.tensors[0][48*i:48*(i+1)].to(device)
        tmp_lr = combined_dataset.tensors[1][48*i:48*(i+1)].to(device)
        q_sample = tmp_diffusion.q_sample(tmp_sample, torch.full((48,), timesteps-1, device=device))
        img = torch.randn(tmp_sample.shape, device=device)
        for j in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
            t = torch.full((48,), j, device=device, dtype=torch.long)
            img = tmp_diffusion.p_sample(model.model, img, t, tmp_lr, j)
        diffmap = np.hstack(img.detach().cpu().numpy()[:48, : , 0])
        map_dir = "/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/diffusion/result/"
        hp.write_map(map_dir+"diffused_{}.fits".format(i), diffmap, overwrite=True)
