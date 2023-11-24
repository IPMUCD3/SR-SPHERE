import torch
import pytorch_lightning as pl
import yaml
import os
import numpy as np
import argparse

from scripts.diffusion.schedules import TimestepSampler
from scripts.diffusion.models.ResUnet_timeembed import Unet
from scripts.maploader.maploader import get_data, get_minmaxnormalized_data
from run.diffusion.run_diffusion import DDPM

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--target', type=str, default='difference')
    args = args.parse_args()

    base_dir = "/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE"
    ckpt_dir = f"{base_dir}/ckpt_logs/diffusion/HR/cosine_o4_b24/version_0"

    with open(f"{ckpt_dir}/hparams.yaml", 'r') as stream:
        config_dict = yaml.safe_load(stream)

    pl.seed_everything(1234)

    BATCH_SIZE = 48#config_dict['train']['batch_size']
    PATCH_SIZE = 12 * (config_dict['data']['order'])**2

    lr = get_data(config_dict['data']['LR_dir'], config_dict['data']['n_maps'], config_dict['data']['nside'], config_dict['data']['order'], issplit=True)
    hr = get_data(config_dict['data']['HR_dir'], config_dict['data']['n_maps'], config_dict['data']['nside'], config_dict['data']['order'], issplit=True)

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

    # set up model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ckpt_path= f"{ckpt_dir}/checkpoints/Run_11-06-12-05epoch=24-val_loss=0.07.ckpt"

    #get sampler type
    timesteps = int(config_dict['diffusion']['timesteps'])
    sampler = TimestepSampler(timesteps=timesteps, sampler_type=config_dict['diffusion']['sampler_type'])

    #get model
    model = DDPM(Unet, config_dict, sampler = sampler).to(device)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    print("Model loaded")

    #get diffusion
    tmp_diffusion = model.diffusion

    map_dir = f"{base_dir}/results/imgs/diffusion/HR/{config_dict['train']['log_name']}"
    if not os.path.exists(map_dir):
        os.makedirs(map_dir)

    print("Start Diffusion")
    for i in range(int(PATCH_SIZE/BATCH_SIZE)):
        tmp_sample =data_input[BATCH_SIZE*i:BATCH_SIZE*(i+1)].to(device)
        tmp_lr = data_condition[BATCH_SIZE*i:BATCH_SIZE*(i+1)].to(device)
        img = torch.randn(tmp_sample.shape, device=device)
        with torch.no_grad():
            for j in reversed(range(0, timesteps-1)):
                t = torch.full((BATCH_SIZE,), j, device=device, dtype=torch.long)
                loss = tmp_diffusion.p_losses(model.model, tmp_sample, t, condition=tmp_lr, loss_type=config_dict['diffusion']['loss_type'])
                img = tmp_diffusion.p_sample(model.model, img, t, t_index=j, condition=tmp_lr)
                print('Current step {}, Loss {}'.format(j, loss), flush=True)
                if (j % 10 == 0):
                    diffmap = np.hstack(img.detach().cpu().numpy()[:BATCH_SIZE, : , 0])
                    np.save(map_dir+f"/diffused_step_{str(j).zfill(3)}_patch_{str(i).zfill(2)}.npy", diffmap)