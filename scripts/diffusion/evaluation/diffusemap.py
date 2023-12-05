
import os
import torch
import pytorch_lightning as pl
from glob import glob

from scripts.diffusion.models.Unet_base import Unet
from scripts.maploader.maploader import get_condition_from_params
from scripts.diffusion.evaluation.utils import initialize_model, run_diffusion
from scripts.utils.run_utils import get_parser
from scripts.utils.params import set_params

if __name__ == '__main__':
    args = get_parser()
    args = args.parse_args()

    pl.seed_everything(1234)
    params = set_params(**vars(args))

    ckpt_dir = sorted(glob(f"{params['train']['save_dir']}{params['train']['log_name']}/version_*"), key=lambda x: float(x.split('_')[-1]))[-1]
    ckpt_path = sorted(glob(f"{ckpt_dir}/checkpoints/*.ckpt"), key=lambda x: float(x.split('=')[-1].rsplit('.', 1)[0]))[-1]

    ### get training data
    lr, data_condition,inverse_transform_lr = get_condition_from_params(params)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = initialize_model(Unet, ckpt_path, params, device)

    map_dir = f"{args.base_dir}/results/imgs/diffusion/{params['train']['log_name']}"
    os.makedirs(map_dir, exist_ok=True)

    num_patches = int(data_condition.shape[0] / params['diffusion']['batch_size'])
    run_diffusion(model, data_condition, map_dir, params['diffusion']['batch_size'], num_patches, 
                int(params['diffusion']['timesteps']), savestep=10, verbose=True)