# Refactored code
import datetime
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Local module imports
sys.path.append('/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/')
from srsphere.data.maploader import get_loaders_from_params
from srsphere.tests.params import get_params
from srsphere.models.VGG16 import PerceptualLoss
from srsphere.models.ResUnet import Unet


def setup_trainer(params, logger=None, patience=30):
    """Set up the PyTorch Lightning trainer."""
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        verbose=0,
        mode="min"
    )

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    checkpoint_callback = ModelCheckpoint(
        filename=f"{current_time}-" + "{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        save_last=True,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=params['num_epochs'],
        callbacks=[checkpoint_callback, early_stop_callback],
        num_sanity_val_steps=0,
        accelerator='gpu', devices=1,
        logger=logger
    )
    return trainer


def train_model(model_class, params=None, logger=None, ckpt_path=None):
    """Train the model."""
    if params is None:
        params = get_params()
    train_loader, val_loader = get_loaders_from_params(params)
    model = model_class(params, ckpt_path=ckpt_path)
    if logger is None:
        logger = TensorBoardLogger(save_dir='.')
    trainer = setup_trainer(params, logger)
    trainer.fit(model, train_loader, val_loader)
    return model


class Unet_ploss(Unet):
    """U-Net model with perceptual loss."""
    def __init__(self, params, ckpt_path=None):
        super().__init__(params)
        if ckpt_path is not None:
            try:
                self.loss_fn = PerceptualLoss(params, ckpt_path=ckpt_path)
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                self.loss_fn = PerceptualLoss(params)


if __name__ == '__main__':
    logger = TensorBoardLogger(save_dir='/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/srsphere/run/')
    model = train_model(Unet_ploss, logger=logger)