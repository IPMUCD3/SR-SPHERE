
import pytorch_lightning as pl
import yaml
import datetime
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

def initialize_config(config_file):
    with open(config_file, 'r') as stream:
        config_dict = yaml.safe_load(stream)

    config_dict["train"].update({"steps_per_epoch": int(config_dict['train']['num_epochs']) * int(config_dict["data"]['n_maps']) // int(config_dict['train']['batch_size'])})
    return config_dict

def setup_trainer(logger=None, fname=None, save_top_k=1, max_epochs=1000):
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=0,
        mode="min"
    )

    if fname is not None:
        name = fname
    else:
        dt = datetime.datetime.now()
        name = dt.strftime('Run_%m-%d_%H-%M')

    checkpoint_callback = ModelCheckpoint(
        filename= name + "{epoch:02d}-{val_loss:.2f}",
        save_top_k=save_top_k,
        monitor="val_loss",
        save_last=False,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        num_sanity_val_steps=0,
        accelerator='gpu', devices=1,
        logger=logger
    )
    return trainer

