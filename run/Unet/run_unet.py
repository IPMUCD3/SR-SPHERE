
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from scripts.models.ResUnet import Unet
from scripts.maploader.maploader import get_data, get_minmaxnormalized_data, get_loaders
from scripts.utils.run_utils import setup_trainer

if __name__ == '__main__':
    pl.seed_everything(1234)
    base_dir = "/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE"

    config_file = f"{base_dir}/config/config_unet.yaml"
    with open(config_file, 'r') as stream:
        config_dict = yaml.safe_load(stream)

    ### get training data
    lrmaps_dir = config_dict['data']['lrmaps_dir'] 
    hrmaps_dir = config_dict['data']['hrmaps_dir']
    n_maps = int(config_dict['data']['n_maps'])
    nside = int(config_dict['data']['nside'])
    order = int(config_dict['data']['order'])
    issplit = config_dict['data']['issplit']

    BATCH_SIZE = config_dict['train']['batch_size']
    RATE_TRAIN = config_dict['data']['rate_train']

    learning_rate = float(config_dict['train']['learning_rate'])
    num_epochs = int(config_dict['train']['num_epochs'])

    mults = config_dict['model']['mults']
    inner_channels = config_dict['model']['inner_channels']

    lr = get_data(lrmaps_dir, n_maps, nside, order, issplit)
    hr = get_data(hrmaps_dir, n_maps, nside, order, issplit)

    lr, transforms_lr, inverse_transforms_lr, range_min_lr, range_max_lr = get_minmaxnormalized_data(lr)
    print("LR data loaded. min: {}, max: {}".format(range_min_lr, range_max_lr))

    hr, transforms_hr, inverse_transforms_hr, range_min_hr, range_max_hr = get_minmaxnormalized_data(hr)
    print("HR data loaded. min: {}, max: {}".format(range_min_hr, range_max_hr))

    train_loader, val_loader = get_loaders(lr, hr, RATE_TRAIN, BATCH_SIZE)

    #get model
    model = Unet(
                in_channels=1,
                inner_channels=inner_channels,
                mults=mults,
                nside=nside,
                order=order,
                learning_rate=learning_rate,)

    logger = TensorBoardLogger(save_dir=f'{base_dir}/ckpt_logs/unet', name='HR_LR_normalized')
    trainer = setup_trainer(logger=logger, fname=None, save_top_k=3, max_epochs=num_epochs)
    trainer.fit(model, train_loader, val_loader)