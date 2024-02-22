import argparse
import datetime
import logging
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from srsphere.models.Unet import Unet
from srsphere.models.ddpm import DDPM
from srsphere.diffusion.scheduler import TimestepSampler
from srsphere.dataset.datamodules import DataModule
from srsphere.params import set_params

def setup_trainer(**args):
    logger = TensorBoardLogger(save_dir=args['save_dir'], name=args['log_name'])
    logging.info("data saved in {}".format(args['save_dir']))
    logging.info("data name: {}".format(args['log_name']))

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=args['patience'],
        verbose=0,
        mode="min"
    )

    dt = datetime.datetime.now()
    name = dt.strftime('Run_%m-%d-%H-%M')

    checkpoint_callback = ModelCheckpoint(
        filename= name + "{epoch:02d}-{val_loss:.2f}",
        save_top_k=args['save_top_k'],
        monitor="val_loss",
        save_last=False,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=args['n_epochs'],
        callbacks=[checkpoint_callback, early_stop_callback] if args['early_stop'] else [checkpoint_callback],
        num_sanity_val_steps=0,
        accelerator="gpu", devices=1,
        logger=logger
    )
    return trainer

def get_parser():
    parser = argparse.ArgumentParser(description='Run diffusion process on maps.')
    parser.add_argument('--n_maps', type=int, default=None,
                        help='Number of maps to use.')
    parser.add_argument('--nside', type=int, default=512,
                        help='Nside parameter for the maps.')
    parser.add_argument('--order', type=int, default=2,
                        help='Order of the data. Should be power of 2.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size to use.')
    parser.add_argument('--difference', type=bool, default=True,
                        help='Whether to use difference for the diffusion process.')
    parser.add_argument('--conditioning', type=str, default='concat', choices=['concat', 'addconv'],
                        help='Conditioning type for the diffusion process. Can be "concat" or "addconv".')
    parser.add_argument('--norm_type', type=str, default='batch', choices=['batch', 'group'],
                        help='Normalization type for the model. Can be "batch" or "group".')
    parser.add_argument('--act_type', type=str, default='silu', choices=['mish', 'silu', 'lrelu'],
                        help='Activation type for the model. Can be "mish" or "silu" or "lrelu".')
    parser.add_argument('--use_attn', type=bool, default=False,
                        help='Whether to use attention for the diffusion process.')
    parser.add_argument('--mask', type=bool, default=False,
                        help='Whether to use mask for the diffusion process.')
    parser.add_argument('--scheduler', type=str, default='linear', choices=['linear', 'cosine'],
                        help='Schedule for the diffusion process. Can be "linear" or "cosine".')
    parser.add_argument('--timesteps', type=int, default=2000,
                        help='Number of timesteps for the diffusion process.')
    parser.add_argument('--log_name', type=str, default="test",
                        help='Name of the log file.')
    return parser

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename='./log/train.log')
    args = get_parser()
    args = args.parse_args()

    pl.seed_everything(1234)
    params = set_params(**vars(args))

    ### get training data
    dm = DataModule(**params['data'])
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    #get sampler type
    sampler = TimestepSampler(timesteps=params['diffusion']['timesteps'])

    #get model
    unet = Unet(params['data']["nside"], params['data']["order"], **params['architecture'])
    model = DDPM(unet, sampler, **params['diffusion'])

    trainer = setup_trainer(**params['train'])
    trainer.fit(model, train_loader, val_loader)