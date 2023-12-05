import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np
from einops import rearrange

from scripts.layers.cheby_shev import SphericalChebConv
from scripts.utils.partial_laplacians import get_partial_laplacians
from scripts.utils.healpix_pool_unpool import Healpix
from scripts.models.ResUnet import Block
from scripts.utils.diffusion_utils import exists, default

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

class ResnetBlock_BigGAN(nn.Module):
    """
    Upsampling Residual block of BigGAN. https://arxiv.org/abs/1809.11096
    """
    def __init__(self, in_channels, out_channels, laplacian, kernel_size, time_emb_dim, pooling):
        super().__init__()
        self.mlp1 = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, in_channels))
            if exists(time_emb_dim)
            else None
        )
        self.mlp2 = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
            if exists(time_emb_dim)
            else None
        )
    
        self.norm1 = nn.BatchNorm1d(in_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.act = nn.Mish() if out_channels > 1 else nn.Identity()
        if pooling == "pooling":
            self.pooling = Healpix().pooling
        elif pooling == "unpooling":
            self.pooling = Healpix().unpooling
        elif pooling == "identity":
            self.pooling = nn.Identity()
        else:
            raise ValueError("must be pooling, unpooling or identity")
        self.conv_res = SphericalChebConv(in_channels, out_channels, laplacian, kernel_size)
        self.conv1 = SphericalChebConv(in_channels, out_channels, laplacian, kernel_size)
        self.conv2 = SphericalChebConv(out_channels, out_channels, laplacian, kernel_size)

    def forward(self, x, time_emb=None):
        h = x
        if exists(self.mlp1) and exists(time_emb):
            time_emb1 = self.mlp1(time_emb)
            h = rearrange(time_emb1, "b c -> b 1 c") + h

        h = self.norm1(h.permute(0, 2, 1)).permute(0, 2, 1)
        h = self.act(h)
        h = self.pooling(h)
        h = self.conv1(h)

        if exists(self.mlp2) and exists(time_emb):
            time_emb2 = self.mlp2(time_emb)
            h = rearrange(time_emb2, "b c -> b 1 c") + h
        h = self.norm2(h.permute(0, 2, 1)).permute(0, 2, 1)
        h = self.act(h)
        h = self.conv2(h)

        return h + self.conv_res(self.pooling(x))
    
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

        self.dim_in = params["architecture"]["dim_in"]    
        self.dim_out = params["architecture"]["dim_out"]
        self.dim = params["architecture"]["inner_dim"]
        self.dim_mults = [self.dim * factor for factor in params["architecture"]["mults"]]
        self.kernel_size = params["architecture"]["kernel_size"]
        self.nside = params["data"]["nside"]
        self.order = params["data"]["order"]
        self.pooling = Healpix().pooling
        self.unpooling = Healpix().unpooling

        self.depth = len(self.dim_mults)
        self.laps = get_partial_laplacians(self.nside, self.depth, self.order, 'normalized')

        # time embeddings
        self.time_dim = self.dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.dim),
            nn.Linear(self.dim, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        self.init_conv = SphericalChebConv(self.dim_in, self.dim, self.laps[-1], self.kernel_size)
        self.init_conv_cond = SphericalChebConv(self.dim_in, self.dim, self.laps[-1], self.kernel_size)

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
            SphericalChebConv(self.dim, self.dim_out, self.laps[-1], self.kernel_size)
        )

    def forward(self, x, time, condition=None):
        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        if condition is not None:
            cond = self.init_conv_cond(condition)
            x = x + cond

        skip_connections = []

        # downsample
        for block1, block2, downsample in self.down_blocks:
            x = block1(x, t)
            x = block2(x, t)
            skip_connections.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, upsample  in self.up_blocks:
            tmp_connection = skip_connections.pop()
            x = torch.cat([x, tmp_connection], dim=2)
            x = block2(block1(x, t), t)
            x = upsample(x)

        return self.final_conv(x)
    
class Unet_bg(pl.LightningModule):
    """
    Full Unet architecture composed of an encoder (downsampler), a bottleneck, and a decoder (upsampler).
    following arxiv:2311.05217
    """
    def __init__(self, params):
        super().__init__()

        self.dim_in = params["architecture"]["dim_in"]    
        self.dim_out = params["architecture"]["dim_out"]
        self.dim = params["architecture"]["inner_dim"]
        self.dim_mults = [self.dim * factor for factor in params["architecture"]["mults"]]
        self.kernel_size = params["architecture"]["kernel_size"]
        self.nside = params["data"]["nside"]
        self.order = params["data"]["order"]
        self.skip_factor = params["architecture"]["skip_factor"]

        self.depth = len(self.dim_mults)
        self.laps = get_partial_laplacians(self.nside, self.depth, self.order, 'normalized')

        # time embeddings
        self.time_dim = self.dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.dim),
            nn.Linear(self.dim, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        self.init_conv = SphericalChebConv(self.dim_in, self.dim, self.laps[-1], self.kernel_size)
        self.init_conv_cond = SphericalChebConv(self.dim_in, self.dim, self.laps[-1], self.kernel_size)

        self.down_blocks = nn.ModuleList([])
        for dim_in, dim_out, lap_id, lap_down in zip(self.dim_mults[:-1], self.dim_mults[1:], reversed(self.laps[1:]), reversed(self.laps[:-1])):
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        ResnetBlock_BigGAN(dim_in, dim_out, lap_id, self.kernel_size, self.time_dim, "identity"),
                        ResnetBlock_BigGAN(dim_out, dim_out, lap_down, self.kernel_size, self.time_dim, "pooling")
                    ]
                )
            )

        self.mid_block = ResnetBlock_BigGAN(self.dim_mults[-1], self.dim_mults[-1], self.laps[0], self.kernel_size, self.time_dim, "identity")

        self.up_blocks = nn.ModuleList([])
        for dim_in, dim_out, lap in zip(reversed(self.dim_mults[1:]), reversed(self.dim_mults[:-1]), self.laps[1:]):
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        ResnetBlock_BigGAN(dim_in, dim_in, lap, self.kernel_size, self.time_dim, "unpooling"),
                        ResnetBlock_BigGAN(2 * dim_in, dim_out, lap, self.kernel_size, self.time_dim, "identity")
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            ResnetBlock_BigGAN(self.dim_mults[0], self.dim, self.laps[-1], self.kernel_size, self.time_dim, "identity"), 
            SphericalChebConv(self.dim, self.dim_out, self.laps[-1], self.kernel_size)
        )

    def forward(self, x, time, condition=None):
        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        if condition is not None:
            cond = self.init_conv_cond(condition)
            x = x + cond

        skip_connections = []

        # downsample
        for block_id, block_down in self.down_blocks:
            x = block_id(x, t)
            skip_connections.append(x)
            x = block_down(x, t)

        # bottleneck
        x = self.mid_block(x, t)

        # upsample
        for block_up, block_id  in self.up_blocks:
            x = block_up(x, t)
            tmp_connection = skip_connections.pop() * self.skip_factor
            x = torch.cat([x, tmp_connection], dim=2)
            x = block_id(x, t)

        return self.final_conv(x)