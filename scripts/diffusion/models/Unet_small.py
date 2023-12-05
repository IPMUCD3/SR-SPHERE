
import torch
from torch import nn
import pytorch_lightning as pl
from functools import partial

from scripts.layers.timeembedding import SinusoidalPositionEmbeddings
from scripts.layers.cheby_shev import SphericalChebConv

from scripts.blocks.Resnetblock import Block, ResnetBlock, ResnetBlock_BG
from scripts.utils.partial_laplacians import get_partial_laplacians
from scripts.utils.healpix_pool_unpool import Healpix
from scripts.utils.diffusion_utils import exists
    
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
        self.skip_factor = params["architecture"]["skip_factor"]
        self.conditioning = params["architecture"]["conditioning"]

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
        if self.conditioning == "addconv":
            self.init_conv_cond = SphericalChebConv(self.dim_in, self.dim, self.laps[-1], self.kernel_size)

        self.block_type = params["architecture"]["block"]
        if self.block_type == "resnet":
            block_ud = partial(ResnetBlock, kernel_size=params["architecture"]["kernel_size"], time_emb_dim=self.time_dim, norm_type=params["architecture"]["norm_type"], act_type=params["architecture"]["act_type"])
        elif self.block_type == "biggan":
            block_ud = partial(ResnetBlock_BG, kernel_size=params["architecture"]["kernel_size"], time_emb_dim=self.time_dim, norm_type=params["architecture"]["norm_type"], act_type=params["architecture"]["act_type"])
        block_id = partial(ResnetBlock, kernel_size=params["architecture"]["kernel_size"], time_emb_dim=self.time_dim, norm_type=params["architecture"]["norm_type"], act_type=params["architecture"]["act_type"])

        self.down_blocks = nn.ModuleList([])
        for dim_in, dim_out, lap_id, lap_down in zip(self.dim_mults[:-1], self.dim_mults[1:], reversed(self.laps[1:]), reversed(self.laps[:-1])):
            self.down_blocks.append(
                nn.ModuleList([
                    block_id(dim_in, dim_out, lap_id),
                    block_ud(dim_out, dim_out, lap_down, pooling = "pooling")
                    ])
                )
        
        self.mid_block1 = block_id(self.dim_mults[-1], self.dim_mults[-1], self.laps[0])
        self.mid_block2 = block_id(self.dim_mults[-1], self.dim_mults[-1], self.laps[0])
        
        self.up_blocks = nn.ModuleList([])
        for dim_in, dim_out, lap in zip(reversed(self.dim_mults[1:]), reversed(self.dim_mults[:-1]), self.laps[1:]):
            self.up_blocks.append(
                nn.ModuleList([
                    block_ud(dim_in, dim_out, lap, pooling = "unpooling"),
                    block_id(2*dim_out, dim_out, lap)
                    ])
                )

        self.final_conv = nn.Sequential(
            block_id(self.dim_mults[0], self.dim, self.laps[-1]), 
            Block(self.dim, self.dim_out, self.laps[-1], self.kernel_size))

    def forward(self, x, time, condition=None):
        skip_connections = []
        if condition is not None:
            if self.conditioning == "concat":
                x = self.init_conv(torch.cat([x, condition], dim=2))
            elif self.conditioning == "addconv":
                x = self.init_conv(x) + self.init_conv_cond(condition)
        else:
            x = self.init_conv(x)
        skip_connections.append(x)
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # downsample
        for block1, block2 in self.down_blocks:
            x = block1(x, t)
            skip_connections.append(x)
            x = block2(x, t)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2  in self.up_blocks:
            x = block1(x, t)
            tmp_connection = skip_connections.pop() * self.skip_factor
            x = torch.cat([x, tmp_connection], dim=2)
            x = block2(x, t)

        return self.final_conv(x)