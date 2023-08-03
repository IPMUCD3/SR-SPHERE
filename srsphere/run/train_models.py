# Refactored code
import datetime
import argparse
import sys
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Local module imports
sys.path.append('/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/')
from srsphere.data.maploader import get_loaders_from_params
from srsphere.tests.params import get_params
from srsphere.ploss.perceptualloss import PerceptualLoss, PerceptualLoss_plus
from srsphere.models.Unet import SphericalUNet
from srsphere.models.threeconv import threeconv
from srsphere.models.model_template import template

def setup_trainer(params, logger=None, patience=100):
    """Set up the PyTorch Lightning trainer."""
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        verbose=0,
        mode="min"
    )

    current_time = datetime.datetime.now().strftime("%Y%m%d")

    checkpoint_callback = ModelCheckpoint(
        filename=f"{current_time}-" + "{epoch:02d}",
        save_top_k=1,
        monitor="val_loss",
        save_last=False,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=params['num_epochs'],
        #callbacks=[checkpoint_callback],
        callbacks=[checkpoint_callback, early_stop_callback],
        num_sanity_val_steps=0,
        accelerator='gpu', devices=1,
        logger=logger
    )
    return trainer


def train_model(model_class, params, logger=None, **args):
    """Train the model."""
    train_loader, val_loader = get_loaders_from_params(params)
    model = model_class(params, **args)
    if logger is None:
        logger = TensorBoardLogger(save_dir='.')
    trainer = setup_trainer(params, logger)
    trainer.fit(model, train_loader, val_loader)
    return model

class selected_model(template):    
    def __init__(self, params, **args):
        super().__init__(params)
        if args.get('model') is not None:
            if args.get('model') == "Unet":
                self.model=SphericalUNet(params)
            elif args.get('model') == "threeconv":
                self.model=threeconv(params)
            else:
                raise ValueError("model must be specified")
        else:
            print("model is not specified. Use threeconv as default")
            self.model=threeconv(params)
        
        if args.get('loss_fn') is not None:
            if args.get('loss_fn') == "mse":
                self.loss_fn=nn.MSELoss()
            elif args.get('loss_fn') == "l1":
                self.loss_fn=nn.L1Loss()
            elif args.get('loss_fn') == "ploss":
                self.loss_fn=PerceptualLoss()
            elif args.get('loss_fn') == 'ploss_mse':
                self.loss_fn = PerceptualLoss_plus(add_loss='mse', model="VGG16", lambda_=0.1)
            elif args.get('loss_fn') == 'ploss_l1':
                self.loss_fn = PerceptualLoss_plus(add_loss='l1', model="VGG16", lambda_=0.1)
            else:
                raise ValueError("loss_fn must be specified")
        else:
            print("loss_fn is not specified. Use mse as default")
            self.loss_fn=nn.MSELoss()

    def forward(self, x):
        return self.model(x)
    
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--model', type=str, default='threeconv')
    args.add_argument('--loss_fn', type=str, default='mse')
    args.add_argument('--num_epochs', type=int, default=1000)
    args.add_argument('--batch_size', type=int, default=32)
    args = args.parse_args()

    pl.seed_everything(1234)
    base_logdir = '/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/log/'
    logger = TensorBoardLogger(save_dir=base_logdir, name=args.model+'_'+args.loss_fn)

    params = get_params()
    params['num_epochs'] = args.num_epochs
    params['batch_size'] = args.batch_size
    model=train_model(selected_model, params, logger, model=args.model, loss_fn=args.loss_fn)
