import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from srsphere.data.maploader import get_loaders_from_params
from srsphere.tests.params import get_params

def setup_trainer(params, logger=None):
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=30,
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


def train_model(model_class, params=None, logger=None):
    if params is None:
        params = get_params()
    train_loader, val_loader = get_loaders_from_params(params)
    model = model_class(params)
    if logger is None:
        logger = TensorBoardLogger(save_dir='.')
    trainer = setup_trainer(params, logger)
    trainer.fit(model, train_loader, val_loader)
    return model

