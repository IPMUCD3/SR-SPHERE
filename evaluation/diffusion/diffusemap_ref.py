
import torch
import pytorch_lightning as pl
import yaml
import os
import numpy as np
import healpy as hp
import argparse
import itertools
from glob import glob
from scripts.diffusion.schedules import TimestepSampler
from scripts.diffusion.ResUnet_timeembed import Unet, Unet_ref
from scripts.maploader.maploader import get_data, get_normalized_data, get_log2linear_transform
from run.diffusion.run_diffusion import DDPM


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run diffusion process on maps.')
    parser.add_argument('--target', type=str, default='difference', choices=['difference', 'HR'],
                        help='Target for the diffusion process. Can be "difference" or "HR".')
    parser.add_argument('--schedule', type=str, default='linear', choices=['linear', 'cosine'],
                        help='Schedule for the diffusion process. Can be "linear" or "cosine".')
    parser.add_argument('--normtype', type=str, default='sigmoid', choices=['sigmoid', 'minmax'],
                        help='Normalization type for the data. Can be "sigmoid" or "minmax".')
    parser.add_argument('--version', type=int, default=1, help='Version of the model to load.')
    parser.add_argument('--ifref', type=bool, default=True, help='If the model is ref model.')
    return parser.parse_args()


def load_configuration(checkpoint_dir):
    with open(f"{checkpoint_dir}/hparams.yaml", 'r') as stream:
        return yaml.safe_load(stream)


def initialize_model(ckpt_dir, config_dict, device):
    ckpt_path = sorted(glob(f"{ckpt_dir}/checkpoints/*.ckpt"), key=lambda x: float(x.split('=')[-1].rsplit('.', 1)[0]))[-1]
    timesteps = int(config_dict['diffusion']['timesteps'])
    sampler = TimestepSampler(timesteps=timesteps, sampler_type=config_dict['diffusion']['sampler_type'])
    model = DDPM(Unet_ref, config_dict, sampler=sampler).to(device)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model


def run_diffusion(model, config_dict, data_input, data_condition, target, patch_size, batch_size, device, version, savestep=10):
    timesteps = int(config_dict['diffusion']['timesteps'])
    map_dir = f"{base_dir}/results/imgs/diffusion/{target}/{config_dict['train']['log_name']}/version_{version}"
    if not os.path.exists(map_dir):
        os.makedirs(map_dir)

    print("Start Diffusion")
    for i in range(int(patch_size/batch_size)):
        tmp_sample = data_input[batch_size*i:batch_size*(i+1)].to(device)
        tmp_lr = data_condition[batch_size*i:batch_size*(i+1)].to(device)
        img = torch.randn(tmp_sample.shape, device=device)
        with torch.no_grad():
            for j in reversed(range(0, timesteps)):
                t = torch.full((batch_size,), j, device=device, dtype=torch.long)
                loss = model.diffusion.p_losses(model.model, tmp_sample, t, condition=tmp_lr,
                                                loss_type=config_dict['diffusion']['loss_type'])
                img = model.diffusion.p_sample(model.model, img, t, t_index=j, condition=tmp_lr)
                print('Current step: {}, Loss: {}'.format(j, loss.item()), flush=True)
                if j % savestep == 0:
                    diffmap = np.hstack(img.detach().cpu().numpy()[:, :, 0])
                    np.save(f"{map_dir}/diffused_step_{str(j).zfill(3)}_patch_{str(i).zfill(2)}.npy", diffmap)

def run_diffusion_oneloop(model,
                        img,
                        tmp_sample, 
                        tmp_condition, 
                        save_dir, 
                        timesteps, 
                        device, 
                        loss_type,
                        savestep=10,
                        mask_bin=None):
    batch_size = tmp_sample.shape[0]
    with torch.no_grad():
        for j in reversed(range(0, timesteps)):
            t = torch.full((batch_size,), j, device=device, dtype=torch.long)
            img = model.diffusion.p_sample(model.model, img, t, t_index=j, condition=tmp_condition)
            if j % savestep == 0:
                loss = model.diffusion.p_losses(model.model, tmp_sample, t, condition=tmp_condition,
                                            loss_type=loss_type)
                print('Current step: {}, Loss: {}'.format(j, loss.item()), flush=True)
                diffmap = img.detach().cpu().numpy()
                if mask_bin is None:
                    np.save(f"{save_dir}/{str(j).zfill(3)}.npy", diffmap)
                else:
                    np.save(f"{save_dir}/{str(j).zfill(3)}_{mask_bin}.npy", diffmap)

def mask_with_gaussian(x, mask, device, num_chunk=4):
    x = torch.chunk(x, num_chunk, dim=1)
    x = [x[i] if mask[i] else torch.randn_like(x[i], device=device) for i in range(num_chunk)]
    x = torch.cat(x, dim=1)
    return x

def run_mask_diffusion(model, config_dict, data_input, data_condition, batch_size, device, version, savestep=10, onlyfirst=True):
    timesteps = int(config_dict['diffusion']['timesteps'])
    patch_size = int(12 * config_dict['data']['order']**2)

    map_dir = f"{config_dict['train']['save_dir']}{config_dict['train']['log_name']}/version_{version}"
    if not os.path.exists(map_dir):
        os.makedirs(map_dir)

    print("Start first Diffusion, generate map from noise")
    for i in range(int(patch_size/batch_size)):
        tmp_sample = data_input[batch_size*i:batch_size*(i+1)].to(device)
        tmp_lr = data_condition[batch_size*i:batch_size*(i+1)].to(device)
        img = torch.randn(tmp_sample.shape, device=device)
        tmp_save_dir = f"{map_dir}/gen_0/patch_{str(i).zfill(2)}"
        if not os.path.exists(tmp_save_dir):
            os.makedirs(tmp_save_dir)
        run_diffusion_oneloop(model, img, tmp_sample, tmp_lr, tmp_save_dir,
                        timesteps, device, config_dict['diffusion']['loss_type'], savestep=savestep)
    
    if onlyfirst:
        return 0
        
    """
    print("Start second Diffusion, generate 3 map patch from map with 1 patch already generated")
    # make a list of masks, possible pattern of True/False for 4 small patches
    masks = list(itertools.product([True, False], repeat=4))
    mask1 = [mask for mask in masks if sum(mask) == 1]
    for i in range(int(patch_size/batch_size)):
        tmp_sample = data_input[batch_size*i:batch_size*(i+1)].to(device)
        tmp_lr = data_condition[batch_size*i:batch_size*(i+1)].to(device)
        
        # randomly choose take 1 mask from mask3, record the number of the mask
        mask = mask1[np.random.choice(len(mask1))]
        mask_bin = ''.join('1' if val else '0' for val in mask)

        img = torch.from_numpy(np.load(f"{map_dir}/gen_0/patch_{str(i).zfill(2)}/000.npy")).to(device)
        img = mask_with_gaussian(img, mask, device)
        tmp_save_dir = f"{map_dir}/gen_1/patch_{str(i).zfill(2)}"
        run_diffusion_oneloop(model, img, tmp_sample, tmp_lr, tmp_save_dir,
                        timesteps, device, config_dict['diffusion']['loss_type'], savestep=savestep, mask_bin=mask_bin)
    
    print("Start third Diffusion, generate 2 map patch from map with 2 patch already generated")
    mask2 = [mask for mask in masks if sum(mask) == 2]
    for i in range(int(patch_size/batch_size)):
        tmp_sample = data_input[batch_size*i:batch_size*(i+1)].to(device)
        tmp_lr = data_condition[batch_size*i:batch_size*(i+1)].to(device)

        tmp_basemap = glob(f"{map_dir}/gen_1/patch_{str(i).zfill(2)}/000_*.npy")[0]
        prev_mask = tmp_basemap.split('_')[-1].split('.')[0]
        # randomly choose take 1 mask from mask2 except the ones has False in the same position as the previous step, record the number of the mask
        tmp_mask2 = [mask for mask in mask2 if ''.join('1' if val else '0' for val in mask) != prev_mask]
        mask = tmp_mask2[np.random.choice(len(tmp_mask2))]
        mask_bin = ''.join('1' if val else '0' for val in mask)

        img = torch.from_numpy(np.load(tmp_basemap)).to(device)
        img = mask_with_gaussian(img, mask, device)
        tmp_save_dir = f"{map_dir}/gen_2/patch_{str(i).zfill(2)}"
        run_diffusion_oneloop(model, img, tmp_sample, tmp_lr, tmp_save_dir,
                        timesteps, device, config_dict['diffusion']['loss_type'], savestep=savestep, mask_bin=mask_bin)
        
    print("Start fourth Diffusion, generate 1 map patch from map with 3 patch already generated")
    mask3 = [mask for mask in masks if sum(mask) == 3]
    for i in range(int(patch_size/batch_size)):
        tmp_sample = data_input[batch_size*i:batch_size*(i+1)].to(device)
        tmp_lr = data_condition[batch_size*i:batch_size*(i+1)].to(device)

        tmp_basemap = glob(f"{map_dir}/gen_2/patch_{str(i).zfill(2)}/000_*.npy")[0]
        prev_mask = tmp_basemap.split('_')[-1].split('.')[0]
        # randomly choose take 1 mask from mask3 except the one used in the previous step, record the number of the mask
        tmp_mask3 = [mask for mask in mask3 if ''.join('1' if val else '0' for val in mask) != prev_mask]
        mask = tmp_mask3[np.random.choice(len(tmp_mask3))]
        mask_bin = ''.join('1' if val else '0' for val in mask)

        img = torch.from_numpy(np.load(tmp_basemap)).to(device)
        img = mask_with_gaussian(img, mask, device)
        tmp_save_dir = f"{map_dir}/gen_3/patch_{str(i).zfill(2)}"
        run_diffusion_oneloop(model, img, tmp_sample, tmp_lr, tmp_save_dir,
                        timesteps, device, config_dict['diffusion']['loss_type'], savestep=savestep, mask_bin=mask_bin)
        """


def load_data(parms, issplit=True):
    lr = get_data(parms["data"]["LR_dir"], parms["data"]["n_maps"], parms["data"]["nside"], parms["data"]["order"], issplit=issplit)
    print("LR data loaded from {}.  Number of maps: {}".format(parms["data"]["LR_dir"], parms["data"]["n_maps"]))

    hr = get_data(parms["data"]["HR_dir"], parms["data"]["n_maps"], parms["data"]["nside"], parms["data"]["order"], issplit=issplit)
    print("HR data loaded from {}.  Number of maps: {}".format(parms["data"]["HR_dir"], parms["data"]["n_maps"]))
    return lr, hr

def preprocess_data(lr, hr, parms):
    lr, transform_lr, inverse_transform_lr, range_min_lr, range_max_lr = get_normalized_data(lr, transform_type=parms["data"]["transform_type"])
    print("LR data normalized by {} transform.".format(parms["data"]["transform_type"]))

    if parms["train"]["target"] == 'difference':
        log2linear_transform, inverse_log2linear_transform = get_log2linear_transform()
        diff = log2linear_transform(hr) - log2linear_transform(inverse_transform_lr(lr))*(parms["data"]["upsample_scale"]**3)
        print("Difference data calculated from HR - LR*upsample_scale^3. min: {}, max: {}".format(diff.min(), diff.max()))
        diff, transforms_diff, inverse_transforms_diff, range_min_diff, range_max_diff = get_normalized_data(diff, transform_type=parms["data"]["transform_type"])
        print("Difference data normalized by {} transform.".format(parms["data"]["transform_type"]))
        data_input, data_condition = diff, lr
    elif parms["train"]["target"] == 'HR':
        hr, transforms_hr, inverse_transforms_hr, range_min_hr, range_max_hr = get_normalized_data(hr, transform_type=parms["data"]["transform_type"])
        print("HR data normalized by {} transform.".format(parms["data"]["transform_type"]))
        data_input, data_condition = hr, lr
    else:
        raise ValueError("target must be 'difference' or 'HR'")
    
    print("data nside: {}, divided into {} patches, each patch has {} pixels.".format(parms["data"]["nside"], 12 * parms["data"]["order"]**2, hp.nside2npix(parms["data"]["nside"])//(12 * parms["data"]["order"]**2)))
    return data_input, data_condition

if __name__ == '__main__':
    args = parse_arguments()
    base_dir = "/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE"
    ckpt_dir = f"{base_dir}/ckpt_logs/diffusion/{args.target}/{args.target}_{args.schedule}_{args.normtype}_o2_b6{'_ref' if args.ifref else ''}/version_{args.version}"

    config_dict = load_configuration(ckpt_dir)
    pl.seed_everything(1234)

    BATCH_SIZE = config_dict['train']['batch_size']*2
    PATCH_SIZE = 12 * (config_dict['data']['order'])**2

    lr, hr = load_data(config_dict)
    data_input, data_condition = preprocess_data(lr, hr, config_dict)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = initialize_model(ckpt_dir, config_dict, device)

    run_diffusion(model, config_dict, data_input, data_condition, args.target, PATCH_SIZE, BATCH_SIZE, device, args.version)