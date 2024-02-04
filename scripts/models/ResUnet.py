
from torch import nn
from torch import optim
import torch
import pytorch_lightning as pl

from scripts.layers.cheby_shev import SphericalChebConv
from scripts.utils import get_partial_laplacians, Healpix

class Block(nn.Module):
    """
    Basic building block for the Unet architecture.
    """
    def __init__(self, in_channels, out_channels, laplacian, kernel_size=8):
        super().__init__()
        self.conv = SphericalChebConv(in_channels, out_channels, laplacian, kernel_size)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.Mish() if out_channels > 1 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(self.norm(x.permute(0, 2, 1)).permute(0, 2, 1))
        return x


class ResnetBlock(nn.Module):
    """
    Residual block composed of two basic blocks. https://arxiv.org/abs/1512.03385
    """
    def __init__(self, in_channels, out_channels, laplacian, kernel_size=8):
        super().__init__()
        self.block1 = Block(in_channels, out_channels, laplacian, kernel_size)
        self.block2 = Block(out_channels, out_channels, laplacian, kernel_size)
        self.res_conv = SphericalChebConv(in_channels, out_channels, laplacian, kernel_size) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.block2(self.block1(x)) + self.res_conv(x)


class Unet(pl.LightningModule):
    """
    Full Unet architecture composed of an encoder (downsampler), a bottleneck, and a decoder (upsampler).
    """
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)
        self.dim_in = params["architecture"]["dim_in"]
        self.dim_out = params["architecture"]["dim_out"]
        self.dim = params["architecture"]["inner_dim"]
        self.dim_mults = [self.dim * factor for factor in params["architecture"]["mults"]]
        self.kernel_size = params["architecture"]["kernel_size"]
        self.nside = params["data"]["nside"]
        self.order = params["data"]["order"]
        self.loss_fn = nn.MSELoss()
        self.learning_rate = params["train"]["learning_rate"]
        self.gamma = params["train"]["gamma"]

        self.pooling = Healpix().pooling
        self.unpooling = Healpix().unpooling

        self.depth = len(self.dim_mults)
        self.laps = get_partial_laplacians(self.nside, self.depth, self.order, 'normalized')

        self.init_conv = SphericalChebConv(self.dim_in, self.dim, self.laps[-1], self.kernel_size)

        self.down_blocks = nn.ModuleList([])
        for dim_in, dim_out, lap in zip([self.dim] + self.dim_mults[:-1], self.dim_mults, reversed(self.laps)):
            is_last = dim_out == self.dim_mults[-1]

            self.down_blocks.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, lap),
                        ResnetBlock(dim_out, dim_out, lap),
                        self.pooling if not is_last else nn.Identity(),
                    ]
                )
            )

        self.mid_block1 = ResnetBlock(self.dim_mults[-1], self.dim_mults[-1], self.laps[0])
        self.mid_block2 = ResnetBlock(self.dim_mults[-1], self.dim_mults[-1], self.laps[0])

        self.up_blocks = nn.ModuleList([])
        for dim_in, dim_out, lap in zip(reversed(self.dim_mults), reversed([self.dim] + self.dim_mults[:-1]), self.laps):
            is_last = dim_in == self.dim_mults[0]
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in*2, dim_out, lap),
                        ResnetBlock(dim_out, dim_out, lap),
                        self.unpooling if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            ResnetBlock(self.dim, self.dim, self.laps[-1]), 
            SphericalChebConv(self.dim, self.dim_out, self.laps[-1], self.kernel_size)
        )

    def forward(self, x):
        x = self.init_conv(x)

        skip_connections = []

        # downsample
        for block1, block2, downsample in self.down_blocks:
            x = block1(x)
            x = block2(x)
            skip_connections.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block2(self.mid_block1(x))

        # upsample
        for block1, block2, upsample  in self.up_blocks:
            tmp_connection = skip_connections.pop()
            x = torch.cat([x, tmp_connection], dim=2)
            x = block2(block1(x))
            x = upsample(x)

        return self.final_conv(x)
    
    def training_step(self, batch, batch_idx):
        y, x = batch
        y_hat = self(x)
        loss =  self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y, x = batch
        y_hat = self(x)
        loss =  self.loss_fn(y_hat, y)
        self.log('val_loss', loss, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}  