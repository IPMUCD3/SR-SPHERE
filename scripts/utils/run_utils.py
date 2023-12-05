
import argparse
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

def get_parser(): #list of args: [base_dir, n_maps, order, transform_type, model, conditioning, norm_type, act_type, block, scheduler, target, batch_size]
    parser = argparse.ArgumentParser(description='Run diffusion process on maps.')
    parser.add_argument('--base_dir', type=str, default="/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE",
                        help='Base directory for the project.')
    parser.add_argument('--n_maps', type=int, default=None,
                        help='Number of maps to use.')
    parser.add_argument('--order', type=int, default=2,
                        help='Order of the data. Should be power of 2.')
    parser.add_argument('--transform_type', type=str, default='sigmoid', choices=['sigmoid', 'minmax', 'both'],
                        help='Normalization type for the data. Can be "sigmoid" or "minmax" or "both".')
    parser.add_argument('--model', type=str, default='diffusion', choices=['diffusion', 'threeconv', 'unet'],
                        help='Model to use. Can be "diffusion" or "threeconv" or "unet".')
    parser.add_argument('--conditioning', type=str, default='concat', choices=['concat', 'addconv'],
                        help='Conditioning type for the diffusion process. Can be "concat" or "addconv".')
    parser.add_argument('--norm_type', type=str, default='group', choices=['batch', 'group'],
                        help='Normalization type for the model. Can be "batch" or "group".')
    parser.add_argument('--act_type', type=str, default='silu', choices=['mish', 'silu', 'lrelu'],
                        help='Activation type for the model. Can be "mish" or "silu" or "lrelu".')
    parser.add_argument('--block', type=str, default='biggan', choices=['biggan', 'resnet'],
                        help='Block type for the model. Can be "biggan" or "resnet".')
    parser.add_argument('--mask', type=bool, default=False,
                        help='Whether to use mask for the diffusion process.')
    parser.add_argument('--use_attn', type=bool, default=False,
                        help='Whether to use attention for the diffusion process.')
    parser.add_argument('--scheduler', type=str, default='linear', choices=['linear', 'cosine'],
                        help='Schedule for the diffusion process. Can be "linear" or "cosine".')
    parser.add_argument('--target', type=str, default='HR', choices=['difference', 'HR'],
                        help='Target for the diffusion process. Can be "difference" or "HR".')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size to use.')
    return parser