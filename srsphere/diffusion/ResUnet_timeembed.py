import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np
from einops import rearrange
import sys

sys.path.append('/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE')
from srsphere.utils.cheby_shev import SphericalChebConv
from srsphere.utils.partial_laplacians import get_partial_laplacians
from srsphere.utils.healpix_pool_unpool import Healpix
from srsphere.models.ResUnet import Block
from srsphere.diffusion.utils import exists, default

class ResnetBlock_t(nn.Module):
    """
    Residual block composed of two basic blocks. https://arxiv.org/abs/1512.03385
    """
    def __init__(self, in_channels, out_channels, laplacian, kernel_size, time_emb_dim):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
            if exists(time_emb_dim)
            else None
        )
        self.block1 = Block(in_channels, out_channels, laplacian, kernel_size)
        self.block2 = Block(out_channels, out_channels, laplacian, kernel_size)
        self.res_conv = SphericalChebConv(in_channels, out_channels, laplacian, kernel_size) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b 1 c") + h

        h = self.block2(h)
        return h + self.res_conv(x)

class SinusoidalPositionEmbeddings(nn.Module):
    #embeds time in the phase
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :] #t1: [40, 1], t2: [1, 32]. Works on cpu, not on mps
        #^ is matmul: torch.allclose(res, torch.matmul(t1.float(), t2)): True when cpu
        #NM: added float for mps
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings #Bx64

class Unet(pl.LightningModule):
    """
    Full Unet architecture composed of an encoder (downsampler), a bottleneck, and a decoder (upsampler).
    """
    def __init__(self, params):
        super().__init__()

        self.dim = 64
        self.dim_factor_mults = [1, 2, 4, 8]
        self.dim_mults = [self.dim * factor for factor in self.dim_factor_mults]
        self.kernel_size = params["architecture"]["kernel_size"]
        self.nside = params["data"]["nside_lr"]
        self.order = params["data"]["order"]
        self.pooling = Healpix().pooling
        self.unpooling = Healpix().unpooling

        self.depth = len(self.dim_mults)
        self.laps = get_partial_laplacians(self.nside, self.depth, self.order, 'normalized')

        # time embeddings
        self.time_dim = 64
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.dim),
            nn.Linear(self.dim, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        self.init_conv = SphericalChebConv(1, self.dim, self.laps[-1], self.kernel_size)
        self.init_conv_lr = SphericalChebConv(1, self.dim, self.laps[-1], self.kernel_size)

        self.down_blocks = nn.ModuleList([])
        for dim_in, dim_out, lap in zip([self.dim] + self.dim_mults[:-1], self.dim_mults, reversed(self.laps)):
            is_last = dim_out == self.dim_mults[-1]

            self.down_blocks.append(
                nn.ModuleList(
                    [
                        ResnetBlock_t(dim_in, dim_out, lap, self.kernel_size, self.time_dim),
                        ResnetBlock_t(dim_out, dim_out, lap, self.kernel_size, self.time_dim),
                        self.pooling if not is_last else nn.Identity(),
                    ]
                )
            )

        self.mid_block1 = ResnetBlock_t(self.dim_mults[-1], self.dim_mults[-1], self.laps[0], self.kernel_size, self.time_dim)
        self.mid_block2 = ResnetBlock_t(self.dim_mults[-1], self.dim_mults[-1], self.laps[0], self.kernel_size, self.time_dim)

        self.up_blocks = nn.ModuleList([])
        for dim_in, dim_out, lap in zip(reversed(self.dim_mults), reversed([self.dim] + self.dim_mults[:-1]), self.laps):
            is_last = dim_in == self.dim_mults[0]
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        ResnetBlock_t(dim_in*2, dim_out, lap, self.kernel_size, self.time_dim),
                        ResnetBlock_t(dim_out, dim_out, lap, self.kernel_size, self.time_dim),
                        self.unpooling if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            ResnetBlock_t(self.dim, self.dim, self.laps[-1], self.kernel_size, self.time_dim), 
            SphericalChebConv(self.dim, 1, self.laps[-1], self.kernel_size)
        )

    def forward(self, x, time, x_lr, label=None):
        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        x = x + self.init_conv_lr(x_lr) #residual connection from low-res input

        skip_connections = []

        # downsample
        for block1, block2, downsample in self.down_blocks:
            x = block1(x, t)
            x = block2(x, t)
            skip_connections.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block2(self.mid_block1(x, t), t)

        # upsample
        for block1, block2, upsample  in self.up_blocks:
            tmp_connection = skip_connections.pop()
            x = torch.cat([x, tmp_connection], dim=2)
            x = block2(block1(x, t), t)
            x = upsample(x)

        return self.final_conv(x)