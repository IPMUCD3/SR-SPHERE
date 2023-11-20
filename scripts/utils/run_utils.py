
from glob import glob
import numpy as np
import pytorch_lightning as pl
import datetime
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

def setup_trainer(params,fname=None):
    logger = TensorBoardLogger(save_dir=params["train"]['save_dir'], name=params["train"]['log_name'])
    print("data saved in {}".format(params["train"]['save_dir']))
    print("data name: {}".format(params["train"]['log_name']))

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=params["train"]['patience'],
        verbose=0,
        mode="min"
    )

    if fname is not None:
        name = fname
    else:
        dt = datetime.datetime.now()
        name = dt.strftime('Run_%m-%d-%H-%M')

    checkpoint_callback = ModelCheckpoint(
        filename= name + "{epoch:02d}-{val_loss:.2f}",
        save_top_k=params["train"]['save_top_k'],
        monitor="val_loss",
        save_last=False,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=params["train"]['n_epochs'],
        callbacks=[checkpoint_callback, early_stop_callback] if params["train"]['early_stop'] else [checkpoint_callback],
        num_sanity_val_steps=0,
        accelerator='gpu', devices=1,
        logger=logger
    )
    return trainer

def set_train_params(params=None, target="HR", model="diffusion", batch_size=None,  base_dir=None):
    if params is None:
        params = {}
    if "train" not in params.keys():
        params["train"] = {}
    params["train"]['target']: str = target
    params["train"]['train_rate']: float = 0.825
    params["train"]['batch_size']: int = 6 if batch_size is None else batch_size
    params["train"]['learning_rate'] = 10**-4
    params["train"]['n_epochs']: int = 500
    params["train"]['gamma']: float = 0.9999
    if base_dir is None:
        base_dir = "/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE"
    params["train"]['save_dir']: str = f"{base_dir}/ckpt_logs/{model}/"
    if "diffusion" in params.keys():
        params["train"]['log_name']: str = f"{params['train']['target']}_{params['data']['transform_type']}_{params['diffusion']['schedule']}_b{params['train']['batch_size']}_o{params['data']['order']}"
    else:
        params["train"]['log_name']: str = f"{params['train']['target']}_{params['data']['transform_type']}_b{params['train']['batch_size']}_o{params['data']['order']}"
    params["train"]['patience']: int = 30
    params["train"]['save_top_k']: int = 1
    params["train"]['early_stop']: bool = True
    return params

def set_diffusion_params(params=None, scheduler="linear"):
    if params is None:
        params = {}
    if "diffusion" not in params.keys():
        params["diffusion"] = {}
    params['diffusion']['timesteps']: int = 2000
    params['diffusion']['loss_type']: str = "huber"
    if scheduler == "linear":
        params['diffusion']['schedule']: str = "linear"
        params['diffusion']['linear_beta_start']: float = 10**(-6)
        params['diffusion']['linear_beta_end']: float = 10**(-2)
    elif scheduler == "cosine":
        params['diffusion']['schedule']: str = "cosine"
        params['diffusion']['cosine_beta_s']: float = 0.015
    else:
        raise ValueError("scheduler must be 'linear' or 'cosine'")
    params['diffusion']['sampler_type']: str = "uniform"
    return params

def set_architecture_params(params=None, model="diffusion"):
    if params is None:
        params = {}
    if "architecture" not in params.keys():
        params["architecture"] = {}
    params["architecture"]["kernel_size"]: int = 8 
    params["architecture"]["dim_in"]: int = 1
    params["architecture"]["dim_out"]: int = 1
    params["architecture"]["inner_dim"]: int = 64
    if model != "threeconv":
        params["architecture"]["mults"] = [1, 2, 4, 8, 8]
        params["architecture"]["skip_factor"]: float = 1/np.sqrt(2)
    if model == "diffusion":
        params["architecture"]["conditional"]: bool = True
        params["architecture"]["mask"]: bool = False
    return params

def set_data_params(params=None, n_maps=None, order=2):
    if params is None:
        params = {}
    if "data" not in params.keys():
        params["data"] = {}
    params["data"]["HR_dir"]: str = "/gpfs02/work/akira.tokiwa/gpgpu/FastPM/healpix/nc256/"
    params["data"]["LR_dir"]: str = "/gpfs02/work/akira.tokiwa/gpgpu/FastPM/healpix/nc128/"
    params["data"]["n_maps"]: int = len(glob(params["data"]["LR_dir"] + "*.fits")) if n_maps is None else n_maps
    params["data"]["nside"]: int = 512
    params["data"]["order"]: int = 2 if order is None else order
    params["data"]["transform_type"]: str = "both"
    params["data"]["upsample_scale"]: float = 2.0
    return params

def set_params(
        base_dir="/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE",
        target="HR", 
        model="diffusion", 
        scheduler=None,
        order=None,
        n_maps=None,
        batch_size=None
        ):
    params = {}
    params = set_data_params(params, n_maps=n_maps, order=order)
    params = set_architecture_params(params)
    if model == "diffusion":
        params = set_diffusion_params(params, scheduler=scheduler if scheduler is not None else "linear")
    params = set_train_params(params, target=target, model=model, batch_size=batch_size, base_dir=base_dir)
    return params
