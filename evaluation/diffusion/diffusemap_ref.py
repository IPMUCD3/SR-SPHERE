
import torch
import pytorch_lightning as pl
from tqdm import tqdm 
import os
import numpy as np
import argparse
from glob import glob

from scripts.diffusion.schedules import TimestepSampler
from scripts.diffusion.ResUnet_timeembed import Unet_bg
from scripts.maploader.maploader import get_data_from_params, get_normalized_from_params
from scripts.utils.run_utils import set_params
from run.diffusion.run_diffusion import DDPM

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run diffusion process on maps.')
    parser.add_argument('--target', type=str, default='difference', choices=['difference', 'HR'],
                        help='Target for the diffusion process. Can be "difference" or "HR".')
    parser.add_argument('--scheduler', type=str, default='linear', choices=['linear', 'cosine'],
                        help='Schedule for the diffusion process. Can be "linear" or "cosine".')
    parser.add_argument('--normtype', type=str, default='sigmoid', choices=['sigmoid', 'minmax', 'both'],
                        help='Normalization type for the data. Can be "sigmoid" or "minmax" or "both".')
    parser.add_argument('--version', type=int, default=1, help='Version of the model to load.')
    return parser.parse_args()

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

def run_diffusion(model, params, data_condition,
                device, version, savestep=10, start_mult=1, batch_size=None):
    batch_size = params['train']['batch_size']*2 if batch_size is None else batch_size
    patch_size = 12 * (params['data']['order'])**2
    timesteps = int(params['diffusion']['timesteps'] * start_mult)
    map_dir = f"{base_dir}/results/imgs/diffusion/{params['train']['log_name']}/version_{version}"
    if not os.path.exists(map_dir):
        os.makedirs(map_dir)

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
    args = parse_arguments()
    base_dir = "/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE"
    params = set_params(base_dir, args.target, "diffusion", args.scheduler)

    ckpt_dir = f"{params['train']['save_dir']}{params['train']['log_name']}/version_{args.version}"
    ckpt_path = sorted(glob(f"{ckpt_dir}/checkpoints/*.ckpt"), key=lambda x: float(x.split('=')[-1].rsplit('.', 1)[0]))[-1]

    pl.seed_everything(1234)

    lr, hr = get_data_from_params(params)
    data_input, data_condition, transforms_lr, inverse_transforms_lr, transforms_hr, inverse_transforms_hr, range_min_lr, range_max_lr, range_min_hr, range_max_hr = get_normalized_from_params(lr, hr, params)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = initialize_model(Unet_bg, ckpt_path, params, device)

    run_diffusion(model, params, data_condition, device, args.version)