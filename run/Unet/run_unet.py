
import pytorch_lightning as pl

from scripts.models.ResUnet import Unet
from scripts.maploader.maploader import get_loaders_from_params
from scripts.utils.run_utils import setup_trainer, set_params

if __name__ == '__main__':
    pl.seed_everything(1234)
    base_dir = "/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE"
    params = set_params(base_dir, model="unet")
    params["train"]["batch_size"] = 2

    ### get training data
    train_loader, val_loader = get_loaders_from_params(params)

    #get model
    model = Unet(params)

    trainer = setup_trainer(params)
    trainer.fit(model, train_loader, val_loader)