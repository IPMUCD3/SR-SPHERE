
import os
import numpy as np
from glob import glob

def set_data_params(params=None, 
                    n_maps=None, 
                    order=2, 
                    transform_type="sigmoid"):
    if params is None:
        params = {}
    if "data" not in params.keys():
        params["data"] = {}
    params["data"]["HR_dir"]: str = "/gpfs02/work/akira.tokiwa/gpgpu/FastPM/healpix/nc256/"
    params["data"]["LR_dir"]: str = "/gpfs02/work/akira.tokiwa/gpgpu/FastPM/healpix/nc128/"
    params["data"]["n_maps"]: int = n_maps if n_maps is not None else len(glob(params["data"]["LR_dir"] + "*.fits"))
    params["data"]["nside"]: int = 512
    params["data"]["order"]: int = order
    params["data"]["transform_type"]: str = transform_type
    params["data"]["upsample_scale"]: float = 2.0
    return params

def set_diffusion_params(params=None, scheduler="linear"):
    if params is None:
        params = {}
    if "diffusion" not in params.keys():
        params["diffusion"] = {}
    params['diffusion']['timesteps']: int = 2000
    params['diffusion']['loss_type']: str = "huber"
    params['diffusion']['schedule']: str = scheduler
    if params['diffusion']['schedule'] == "linear":
        params['diffusion']['linear_beta_start']: float = 10**(-6)
        params['diffusion']['linear_beta_end']: float = 10**(-2)
    elif params['diffusion']['schedule'] == "cosine":
        params['diffusion']['cosine_beta_s']: float = 0.015
    params['diffusion']['sampler_type']: str = "uniform"
    return params

def set_architecture_params(params=None, 
                            model="diffusion", 
                            conditioning="concat",
                            norm_type="batch",
                            act_type="mish",
                            block="biggan",
                            mask=False):
    if params is None:
        params = {}
    if "architecture" not in params.keys():
        params["architecture"] = {}
    params["architecture"]["model"]: str = model
    if params["architecture"]["model"] != "threeconv":
        params["architecture"]["mults"] = [1, 2, 4, 4]
        params["architecture"]["block"]: str = block
        params["architecture"]["skip_factor"]: float = 1/np.sqrt(2)
    if params["architecture"]["model"] == "diffusion":
        params["architecture"]["conditional"]: bool = True
        params["architecture"]["conditioning"]: str = conditioning 
        params["architecture"]["mask"]: bool = mask
    params["architecture"]["kernel_size"]: int = 8 
    params["architecture"]["dim_in"]: int = 1 if params["architecture"]["conditioning"] != "concat" else 2
    params["architecture"]["dim_out"]: int = 1
    params["architecture"]["inner_dim"]: int = 64
    params["architecture"]["norm_type"]: str = norm_type
    params["architecture"]["act_type"]: str = act_type
    return params

def set_train_params(params=None, base_dir=None, target="HR", batch_size=4):
    if params is None:
        params = {}
    if "train" not in params.keys():
        params["train"] = {}
    params["train"]['target']: str = target
    params["train"]['train_rate']: float = 0.825
    params["train"]['batch_size']: int = batch_size
    params["train"]['learning_rate'] = 10**-4
    params["train"]['n_epochs']: int = 1000
    params["train"]['gamma']: float = 0.9999
    params["train"]['save_dir']: str = f"{base_dir}/ckpt_logs/{params['architecture']['model']}/"
    os.makedirs(params["train"]['save_dir'], exist_ok=True)
    if "diffusion" in params.keys():
        params["train"]['log_name']: str = f"{params['train']['target']}_{params['data']['transform_type']}_{params['diffusion']['schedule']}_{params['architecture']['conditioning']}_b{params['train']['batch_size']}_o{params['data']['order']}"
    else:
        params["train"]['log_name']: str = f"{params['train']['target']}_{params['data']['transform_type']}_{params['architecture']['conditioning']}_b{params['train']['batch_size']}_o{params['data']['order']}"
    params["train"]['patience']: int = 30
    params["train"]['save_top_k']: int = 1
    params["train"]['early_stop']: bool = True
    return params

def set_params(
        base_dir="/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE",
        n_maps=None,
        order=2,
        transform_type="sigmoid",
        model="diffusion",
        conditioning="concat",
        norm_type="batch",
        act_type="mish",
        block="biggan",
        mask=False,
        scheduler="linear",
        target="HR", 
        batch_size=4
        ):
    params = {}
    params = set_data_params(params, n_maps=n_maps, order=order, transform_type=transform_type)
    params = set_architecture_params(params, model=model, conditioning=conditioning, 
                                    norm_type=norm_type, act_type=act_type, block=block, mask=mask)
    if model == "diffusion":
        params = set_diffusion_params(params, scheduler=scheduler)
    params = set_train_params(params, base_dir=base_dir, target=target, batch_size=batch_size)
    return params