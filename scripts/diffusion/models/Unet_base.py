
import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np
from einops import rearrange
from functools import partial

from scripts.utils.cheby_shev import SphericalChebConv
from scripts.utils.partial_laplacians import get_partial_laplacians
from scripts.utils.healpix_pool_unpool import Healpix
from scripts.utils.diffusion_utils import exists

class Norms(nn.Module):
    def __init__(self, dim, norm_type, num_groups=8):
        super().__init__()
        if norm_type == "batch":
            self.norm = nn.BatchNorm1d(dim)
        elif norm_type == "group":
            self.norm = nn.GroupNorm(num_groups, dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
    
class Acts(nn.Module):
    def __init__(self, act_type):
        super().__init__()
        if act_type == "mish":
            self.act = nn.Mish() 
        elif act_type == "silu":
            self.act = nn.SiLU() 
        elif act_type == "lrelu":
            self.act = nn.LeakyReLU(0.1) 
        else:
            self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, laplacian, kernel_size=8, norm_type="batch", act_type="mish"):
        super().__init__()
        self.conv = SphericalChebConv(in_channels, out_channels, laplacian, kernel_size)
        self.norm = Norms(out_channels, norm_type)
        self.act = Acts(act_type) if out_channels > 1 else nn.Identity()

    def forward(self, x):
        return self.conv(self.act(self.norm(x)))
    
class time_embed(nn.Module):
    def __init__(self, time_emb_dim, in_channels):
        super().__init__()
        self.proj = nn.Linear(time_emb_dim, in_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.proj(x))
        return rearrange(x, "b c -> b 1 c")

class ResnetBlock(nn.Module):
    """
    Residual block composed of two basic blocks. https://arxiv.org/abs/1512.03385
    """
    def __init__(self, in_channels, out_channels, laplacian, kernel_size=8, time_emb_dim=None, norm_type="batch", act_type="mish"):
        super().__init__()
        if exists(time_emb_dim):
            self.mlp = time_embed(time_emb_dim, out_channels)
        self.block1 = Block(in_channels, out_channels, laplacian, kernel_size, norm_type, act_type)
        self.block2 = Block(out_channels, out_channels, laplacian, kernel_size, norm_type, act_type)
        self.res_conv = SphericalChebConv(in_channels, out_channels, laplacian, kernel_size) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(time_emb):
            h = self.mlp(time_emb) + h

        h = self.block2(h)
        return h + self.res_conv(x)

class ResnetBlock_BigGAN(nn.Module):
    """
    Upsampling Residual block of BigGAN. https://arxiv.org/abs/1809.11096
    """
    def __init__(self, in_channels, out_channels, laplacian, pooling="identity", kernel_size=8, time_emb_dim=None, norm_type="batch", act_type="mish"):
        super().__init__()
        if exists(time_emb_dim):
            self.mlp1 = time_embed(time_emb_dim, in_channels)
            self.mlp2 = time_embed(time_emb_dim, out_channels)

        self.norm1 = Norms(in_channels, norm_type)
        self.norm2 = Norms(out_channels, norm_type)

        self.act1 = Acts(act_type) 
        self.act2 = Acts(act_type) if out_channels > 1 else nn.Identity()

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
        if exists(time_emb):
            h = self.mlp1(time_emb) + h
        h = self.norm1(h)
        h = self.act1(h)
        h = self.pooling(h)

        h = self.conv1(h)
        if exists(time_emb):
            h = self.mlp2(time_emb) + h
        h = self.norm2(h)
        h = self.act2(h)
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
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        if self.block_type == "resnet":
            block_klass = partial(ResnetBlock, kernel_size=params["architecture"]["kernel_size"], time_emb_dim=self.time_dim, norm_type=params["architecture"]["norm_type"], act_type=params["architecture"]["act_type"])
            for dim_in, dim_out, lap_id, lap_down in zip(self.dim_mults[:-1], self.dim_mults[1:], reversed(self.laps[1:]), reversed(self.laps[:-1])):
                self.down_blocks.append(nn.ModuleList([block_klass(dim_in, dim_out, lap_id), self.pooling, block_klass(dim_out, dim_out, lap_down)]))
            for dim_in, dim_out, lap in zip(reversed(self.dim_mults[1:]), reversed(self.dim_mults[:-1]), self.laps[1:]):
                self.up_blocks.append(nn.ModuleList([self.unpooling, block_klass(dim_in, dim_in, lap),block_klass(2 * dim_in, dim_out, lap)]))
        elif self.block_type == "biggan":
            block_klass = partial(ResnetBlock_BigGAN, kernel_size=params["architecture"]["kernel_size"], time_emb_dim=self.time_dim, norm_type=params["architecture"]["norm_type"], act_type=params["architecture"]["act_type"])
            for dim_in, dim_out, lap_id, lap_down in zip(self.dim_mults[:-1], self.dim_mults[1:], reversed(self.laps[1:]), reversed(self.laps[:-1])):
                self.down_blocks.append(nn.ModuleList([block_klass(dim_in, dim_out, lap_id, "identity"),nn.Identity(), block_klass(dim_out, dim_out, lap_down, "pooling")]))
            for dim_in, dim_out, lap in zip(reversed(self.dim_mults[1:]), reversed(self.dim_mults[:-1]), self.laps[1:]):
                self.up_blocks.append(nn.ModuleList([nn.Identity(),block_klass(dim_in, dim_in, lap, "unpooling"),block_klass(2*dim_in, dim_out, lap, "identity")]))

        self.mid_block1 = block_klass(self.dim_mults[-1], self.dim_mults[-1], self.laps[0])
        self.mid_block2 = block_klass(self.dim_mults[-1], self.dim_mults[-1], self.laps[0])
        
        self.final_conv = nn.Sequential(block_klass(self.dim_mults[0], self.dim, self.laps[-1]), SphericalChebConv(self.dim, self.dim_out, self.laps[-1], self.kernel_size))

    def forward(self, x, time, condition=None):
        if condition is not None:
            if self.conditioning == "concat":
                x = torch.cat([x, condition], dim=2)

        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        if condition is not None:
            if self.conditioning == "addconv":
                cond = self.init_conv_cond(condition)
                x = x + cond

        skip_connections = []

        # downsample
        for block1, downsample , block2 in self.down_blocks:
            x = block1(x, t)
            skip_connections.append(x)
            if self.block_type == "resnet":
                x = downsample(x)
            x = block2(x, t)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        # upsample
        
        for upsample, block1, block2  in self.up_blocks:
            tmp_connection = skip_connections.pop() * self.skip_factor
            if self.block_type == "resnet":
                x = upsample(x)
            x = block1(x, t)
            x = torch.cat([x, tmp_connection], dim=2)
            x = block2(x, t)

        return self.final_conv(x)