
import os
import sys
import glob
import torch

sys.path.append('/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/')
from srsphere.data.maploader import MapDataset, get_minmax_transform
from srsphere.tests.params import get_params
from srsphere.run.train_models import selected_model
from srsphere.statistics.utils import powerspec

def prepare_data(params):
    lr = MapDataset(params['lrmaps_dir'], params['n_maps'],  params['nside_lr'], params['order'], 
                    params['issplit'], params["normalize"]).__getitem__(0)
    hr = MapDataset(params['hrmaps_dir'], params['n_maps'],  params['nside_hr'], params['order'],
                    params['issplit'], params["normalize"]).__getitem__(0)
    RANGE_MIN, RANGE_MAX = lr.min().clone().detach(), lr.max().clone().detach()
    transforms, inverse_transforms = get_minmax_transform(RANGE_MIN, RANGE_MAX)
    lr = transforms(lr)
    hr = transforms(hr)
    return lr, hr, inverse_transforms

if __name__ == '__main__':
    params = get_params()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    lr, hr, inverse_transforms = prepare_data(params)
    logdir = "/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/log"
    save_mapdir = "/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/visualization/"
    result_dir = "/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/statistics/result/"
    ckpt_paths = glob.glob(f"{logdir}/*/version_*/checkpoints/*.ckpt")

    n_maps, patch_size, lmax = params["n_maps"], 12*params["order"]**2, params["nside_lr"]*4
    for ckpt_path in ckpt_paths:
        model_name = ckpt_path.split('/')[-4].split('_')[0]
        if model_name == 'resUnet':
            model_name = 'Unet'
        if len(ckpt_path.split('/')[-4].split('_')) == 3:
            if ckpt_path.split('/')[-4].split('_')[2] == 'wL1':
                loss_fn = 'l1'
            elif ckpt_path.split('/')[-4].split('_')[2] == 'wMSE':
                loss_fn = 'mse'
            loss_fn = ckpt_path.split('/')[-4].split('_')[1] + '_' + loss_fn
        else:
            loss_fn = ckpt_path.split('/')[-4].split('_', 1)[1]
        print(f"model: {model_name}, loss_fn: {loss_fn}", flush=True)
        tmp_save_mapdir = save_mapdir + model_name + '_' + loss_fn + '/'
        if not os.path.exists(tmp_save_mapdir):
            os.makedirs(tmp_save_mapdir)
        tmp_result_dir = result_dir + model_name + '_' + loss_fn + '/'
        if not os.path.exists(tmp_result_dir):
            os.makedirs(tmp_result_dir)
        model = selected_model(params, model=model_name, loss_fn=loss_fn).to(device)
        model.load_state_dict(torch.load(ckpt_path)["state_dict"], strict=False)
        powerspec(model, lr, hr, inverse_transforms, n_maps, patch_size, device, lmax,
                  result_dir=tmp_result_dir, save_first=True, save_mapdir=tmp_save_mapdir)
