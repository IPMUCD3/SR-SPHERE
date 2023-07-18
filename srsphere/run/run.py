from pytorch_lightning.loggers import TensorBoardLogger
import sys
sys.path.append('/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE-1/')

from srsphere.models.Unet import SphericalUNet
from srsphere.run.train import train_model

if __name__ == '__main__':
    logger = TensorBoardLogger(save_dir='/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE-1/srsphere/run/')
    model = train_model(SphericalUNet, logger=logger)