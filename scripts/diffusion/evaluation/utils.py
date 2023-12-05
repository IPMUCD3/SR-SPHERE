
import os
import torch
import healpy as hp
import numpy as np
from glob import glob
from tqdm import tqdm
from scripts.diffusion.DDPM import DDPM, TimestepSampler

def initialize_model(denoising_model, ckpt_path, params, device):
    timesteps = int(params['diffusion']['timesteps'])
    sampler = TimestepSampler(timesteps=timesteps, sampler_type=params['diffusion']['sampler_type'])
    model = DDPM(denoising_model, params, sampler=sampler).to(device)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model

def run_diffusion_oneloop(model,
                        y,
                        cond,
                        save_dir, 
                        timesteps, 
                        savestep=10, 
                        verbose=True):
    with torch.no_grad():
        for j in tqdm(reversed(range(0, timesteps)), desc="Diffusion", total=timesteps):
            t = torch.full((y.shape[0],), j, device=y.device, dtype=torch.long)
            y = model.diffusion.p_sample(model.model, y, t, t_index=j, condition=cond)
            if j % savestep == 0:
                diffmap = y.detach().cpu().numpy()
                np.save(f"{save_dir}/step_{str(j).zfill(3)}.npy", diffmap)
                print(f"Saved step {j} to {save_dir}/step_{str(j).zfill(3)}.npy") if verbose else None

def run_diffusion(model, cond, map_dir, batch_size, num_patches, timesteps,savestep=10, verbose=True):
    device = model.device
    for i in range(num_patches):
        print(f"Running diffusion on patch {i+1}/{int(num_patches)}") if verbose else None
        tmp_cond = cond[batch_size*i:batch_size*(i+1)].to(device)
        tmp_save_dir = f"{map_dir}/patch_{i+1}"
        os.makedirs(tmp_save_dir, exist_ok=True)
        y = torch.randn(tmp_cond.shape, device=device)
        run_diffusion_oneloop(model, y, tmp_cond, tmp_save_dir, timesteps, savestep=savestep)

def read_maps(map_dir, diffsteps=100, batch_size=4):
    maps = sorted(glob(map_dir + "/patch_*/step_*.npy"), key=lambda x: (int(x.split("/")[-2].split("_")[-1]), int(x.split("/")[-1].split(".")[0].split("_")[-1])))
    map_diffused = []
    for i in range(diffsteps):
        map_steps = []
        for j in range(batch_size):
            map_steps.append(np.load(maps[i*batch_size+j]))
        map_steps = np.hstack(map_steps)
        map_diffused.append(map_steps)
    map_diffused = np.array(map_diffused)
    return map_diffused

def t2hpr(x):
    x_hp = hp.pixelfunc.reorder(x, n2r=True)
    return x_hp