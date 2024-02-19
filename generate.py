
import torch
import os
import pytorch_lightning as pl
from glob import glob

from srsphere.models.Unet import Unet
from srsphere.models.ddpm import DDPM
from srsphere.diffusion.scheduler import TimestepSampler
from srsphere.params import set_params

from srsphere.dataset.fits_dataset import MapDataset
from srsphere.generation.gen_strategy import GenerationStrategy

def parse_fname(fname):
    # Remove the 'n' prefix and split the string into parts
    parts = fname.split('_')

    # Create a dictionary to hold the parameters
    params = {}

    # Parse each part and add it to the dictionary
    params['n_maps'] = int(parts[0][1:])
    params['nside'] = int(parts[1][1:])
    params['order'] = int(parts[2][1:])
    params['batch_size'] = int(parts[3][1:])
    params['difference'] = parts[4][1:] == 'True'
    params['conditioning'] = parts[5]
    params['norm_type'] = parts[6]
    params['act_type'] = parts[7]
    params['use_attn'] = parts[8][1:] == 'True'
    params['mask'] = parts[9][1:] == 'True'
    params['scheduler'] = parts[10]
    params['timesteps'] = int(parts[11][1:])

    return params

if __name__ == '__main__':
    base_dir = "/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE"
    ckpt_dir = f"{base_dir}/ckpt_logs/n100_s512_o8_b32_dTrue_concat_batch_silu_aFalse_mTrue_cosine_t2000/version_0"
    ckpt_path = sorted(glob(f"{ckpt_dir}/checkpoints/*.ckpt"), key=lambda x: float(x.split('=')[-1].rsplit('.', 1)[0]))[-1]

    pl.seed_everything(1234)
    args = parse_fname(ckpt_dir.split('/')[-2])

    # for validation
    params = set_params(**args)
    params['valid'] = {}
    params['valid']['save_dir'] = f"{base_dir}/results/imgs/diffusion/{ckpt_dir.split('/')[-2]}"
    os.makedirs(params['valid']['save_dir'], exist_ok=True)
    params['valid']['timesteps'] = 1200
    
    dataset = MapDataset(lrdir=params['data']['LR_dir'], hrdir=params['data']['HR_dir'], n_maps=1, norm=params['data']['norm'], order=None, difference=params['data']['difference'], upsample_scale=params['data']['upsample_scale']**3)
    cond = dataset.lrmaps[0].reshape(-1)

    #get sampler type
    sampler = TimestepSampler(timesteps=params['diffusion']['timesteps'])
    #get model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    unet = Unet(params['data']["nside"], params['data']["order"], **params['architecture'])
    model = DDPM(unet, sampler, **params['diffusion']).to(device)
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict['state_dict'], strict=False)
    model.eval()

    strategy = GenerationStrategy(model, params['data']["nside"], params['data']["order"], params['valid'], cond)
    strategy.generation_task()
    print("Map generation completed.")