
import pytorch_lightning as pl

from scripts.diffusion.models.Unet_base import Unet
from scripts.maploader.maploader import get_condition_from_params
from scripts.diffusion.evaluation.utils import initialize_model, run_diffusion
from scripts.utils.run_utils import get_parser
from scripts.utils.params import set_valid_params

if __name__ == '__main__':
    args = get_parser()
    args = args.parse_args()

    pl.seed_everything(1234)
    params = set_valid_params(**vars(args))

    ### get training data
    _, data_cond, _ = get_condition_from_params(params)
    
    model = initialize_model(Unet, params)

    run_diffusion(model, data_cond, params, verbose=True)