
import os
import torch
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm
from glob import glob

from scripts.diffusion.schedules import TimestepSampler
from scripts.diffusion.models.Unet_base import Unet
from scripts.diffusion.models.DDPM import DDPM
from scripts.maploader.maploader import get_data_from_params, get_normalized_from_params
from scripts.utils.run_utils import get_parser
from scripts.utils.params import set_params

def initialize_model(denoising_model, ckpt_path, config_dict, device):
    timesteps = int(config_dict['diffusion']['timesteps'])
    sampler = TimestepSampler(timesteps=timesteps, sampler_type=config_dict['diffusion']['sampler_type'])
    model = DDPM(denoising_model, config_dict, sampler=sampler).to(device)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model

def run_diffusion_oneloop(model,
                        img,
                        tmp_condition,
                        batch_size,
                        save_dir, 
                        timesteps, 
                        device, 
                        savestep=10):
    with torch.no_grad():
        for j in tqdm(reversed(range(0, timesteps)), desc="Diffusion", total=timesteps):
            t = torch.full((batch_size,), j, device=device, dtype=torch.long)
            img = model.diffusion.p_sample(model.model, img, t, t_index=j, condition=tmp_condition)
            if j % savestep == 0:
                diffmap = img.detach().cpu().numpy()
                np.save(f"{save_dir}/step_{str(j).zfill(3)}.npy", diffmap)

def run_diffusion(model, params, data_condition, map_dir,
                device, savestep=10, start_mult=1, batch_size=None):
    batch_size = params['train']['batch_size']*2 if batch_size is None else batch_size
    patch_size = 12 * (params['data']['order'])**2
    timesteps = int(params['diffusion']['timesteps'] * start_mult)
    

    print("Start Diffusion")
    for i in range(int(patch_size/batch_size)):
        print(f"Running diffusion on patch {i+1}/{int(patch_size/batch_size)}")
        tmp_lr = data_condition[batch_size*i:batch_size*(i+1)].to(device)
        tmp_save_dir = f"{map_dir}/patch_{i+1}"
        if not os.path.exists(tmp_save_dir):
            os.makedirs(tmp_save_dir)
        img = torch.randn(tmp_lr.shape, device=device)
        run_diffusion_oneloop(model, img, tmp_lr, batch_size,
                            tmp_save_dir, timesteps, device, savestep=savestep)
    print("Diffusion Finished")


if __name__ == '__main__':
    args = get_parser()
    args = args.parse_args()

    pl.seed_everything(1234)
    params = set_params(**vars(args))

    ckpt_dir = sorted(glob(f"{params['train']['save_dir']}{params['train']['log_name']}/version_*"), key=lambda x: float(x.split('_')[-1]))[-1]
    ckpt_path = sorted(glob(f"{ckpt_dir}/checkpoints/*.ckpt"), key=lambda x: float(x.split('=')[-1].rsplit('.', 1)[0]))[-1]

    ### get training data
    lr, hr = get_data_from_params(params)
    data_input, data_condition, _, _, _, _, _, _, _, _ = get_normalized_from_params(lr, hr, params)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = initialize_model(Unet, ckpt_path, params, device)

    map_dir = f"{args.base_dir}/results/imgs/diffusion/{params['train']['log_name']}"
    if not os.path.exists(map_dir):
        os.makedirs(map_dir)

    run_diffusion(model, params, data_condition, map_dir, device, start_mult=5/8)