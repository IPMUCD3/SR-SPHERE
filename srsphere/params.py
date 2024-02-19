
import numpy as np

def set_data_params(params:dict=None, 
                    n_maps:int=None,
                    nside:int=512,
                    order:int=None,
                    batch_size:int=32,
                    difference:bool=True):
    if params is None:
        params = {}
    if "data" not in params.keys():
        params["data"] = {}
    params["data"]["HR_dir"] = "/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/data/nc256"
    params["data"]["LR_dir"] = "/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/data/nc128"
    params["data"]["n_maps"] = n_maps
    params["data"]["nside"] = nside
    params["data"]["order"] = order
    params["data"]["batch_size"] = batch_size
    params["data"]["rate_split"] = [0.8, 0.1, 0.1]
    params["data"]["norm"] = True
    params["data"]["difference"] = difference
    params["data"]["upsample_scale"] = 2
    return params

def set_diffusion_params(params=None, 
                        mask=False,
                        timesteps=2000,
                        scheduler="linear", 
                        loss_type="huber"):
    if params is None:
        params = {}
    if "diffusion" not in params.keys():
        params["diffusion"] = {}
    params['diffusion']['sampler_type'] = "uniform"

    params["diffusion"]['learning_rate'] = 10**-4
    params["diffusion"]['gamma'] = 0.9999
    params["diffusion"]['mask'] = mask

    params['diffusion']['timesteps'] = timesteps
    params['diffusion']['schedule'] = scheduler
    if params['diffusion']['schedule'] == "linear":
        params['diffusion']['linear_beta_start'] = 10**(-6)
        params['diffusion']['linear_beta_end'] = 10**(-2)
    elif params['diffusion']['schedule'] == "cosine":
        params['diffusion']['cosine_beta_s'] = 0.015
    params['diffusion']['loss_type'] = loss_type
    return params

def set_architecture_params(params=None, 
                            conditioning="concat",
                            use_attn=False,
                            norm_type="batch",
                            act_type="silu",
                            use_conv=False,
                            use_scale_shift_norm=True,
                            num_resblocks=[2, 2, 2, 2]):
    if params is None:
        params = {}
    if "architecture" not in params.keys():
        params["architecture"] = {}
    params["architecture"]["conditioning"]: str = conditioning 
    params["architecture"]["dim_in"]: int = 2 if params["architecture"]["conditioning"] == "concat" else 1
    params["architecture"]["dim_out"]: int = 1
    params["architecture"]["inner_dim"]: int = 64
    params["architecture"]["mults"] = [1, 2, 4, 8]
    params["architecture"]["num_resblocks"]: list = num_resblocks
    params["architecture"]["skip_factor"]: float = 1/np.sqrt(2)
    params["architecture"]["use_attn"]: bool = use_attn
    if params["architecture"]["use_attn"]:
        params["architecture"]["attn_type"]: str = "self"
    params["architecture"]["norm_type"]: str = norm_type
    params["architecture"]["act_type"]: str = act_type
    params["architecture"]["use_conv"]: bool = use_conv
    params["architecture"]["use_scale_shift_norm"]: bool = use_scale_shift_norm
    params["architecture"]["kernel_size"]: int = 20 
    return params

def set_train_params(params=None, save_dir="/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/ckpt_logs", log_name="test"):
    if params is None:
        params = {}
    if "train" not in params.keys():
        params["train"] = {}
    params["train"]['save_dir']: str = save_dir
    params["train"]['log_name']: str = log_name
    params["train"]['n_epochs']: int = 100
    params["train"]['patience']: int = 10
    params["train"]['save_top_k']: int = 3
    params["train"]['early_stop']: bool = True
    return params

def set_params(
        n_maps=None,
        nside=512,
        order=2,
        batch_size=4,
        difference=True,
        conditioning="concat",
        norm_type="batch",
        act_type="silu",
        use_attn=False,
        mask=False,
        scheduler="linear",
        timesteps=2000,
        log_name="test",
        ):
    params = {}
    params = set_data_params(params, n_maps=n_maps, nside=nside, order=order, batch_size=batch_size, difference=difference)
    params = set_architecture_params(params, conditioning=conditioning, norm_type=norm_type, act_type=act_type, use_attn=use_attn)
    params = set_diffusion_params(params, mask=mask, scheduler=scheduler, timesteps=timesteps)
    params = set_train_params(params, log_name=log_name)
    return params