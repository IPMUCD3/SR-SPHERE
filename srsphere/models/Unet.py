
import torch
from torch import nn
import pytorch_lightning as pl
from functools import partial

from srsphere.models.modules.timeembedding import SinusoidalPositionEmbeddings
from srsphere.models.modules.normalization import Norms
from srsphere.models.modules.activation import Acts
#from srsphere.models.modules.attention import Attentions
from srsphere.models.modules.Resnetblock import ResnetBlock

from deepsphere.cheby_shev import SphericalChebConv
from deepsphere.partial_laplacians import get_partial_laplacians
    
class Unet(pl.LightningModule):
    """
    Full Unet architecture composed of an encoder (downsampler), a bottleneck, and a decoder (upsampler).
    The architecture is inspired by the one used in the (https://arxiv.org/abs/2311.05217).
    """
    def __init__(self, nside, order, **args):
        super().__init__()

        # params for basic architecture
        self.dim_in = args["dim_in"]    
        self.dim_out = args["dim_out"]
        self.inner_dim = args["inner_dim"]
        self.dim_mults = [self.inner_dim * factor for factor in args["mults"]]
        self.num_resblocks = args["num_resblocks"]

        # params for further customization
        self.skip_factor =args["skip_factor"]
        self.conditioning = args["conditioning"]
        self.use_attn = args["use_attn"]
        if self.use_attn:
            self.attn_type = args["attn_type"]
        self.norm_type = args["norm_type"]
        self.act_type = args["act_type"]
        self.use_conv = args["use_conv"]
        self.use_scale_shift_norm = args["use_scale_shift_norm"]

        # params for Healpix
        self.kernel_size = args["kernel_size"]
        self.nside = nside
        self.order = order
        self.depth = len(self.dim_mults)
        self.laps = get_partial_laplacians(self.nside, self.depth, self.order, 'normalized')

        # time embeddings
        self.time_dim = self.inner_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.inner_dim),
            nn.Linear(self.inner_dim, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        self.init_conv = SphericalChebConv(self.dim_in, self.inner_dim, self.laps[-1], kernel_size=self.kernel_size)
        if self.conditioning == "addconv":
            self.init_conv_cond = SphericalChebConv(self.dim_in, self.inner_dim, self.laps[-1], kernel_size=self.kernel_size)

        block_partial = partial(ResnetBlock, 
                        kernel_size=self.kernel_size,
                        time_emb_dim=self.time_dim, norm_type=self.norm_type,
                        act_type=self.act_type, 
                        use_conv=self.use_conv,
                        use_scale_shift_norm=self.use_scale_shift_norm)

        self.down_blocks = nn.ModuleList([])
        for dim_in, dim_out, lap_id, lap_down, num_resblock in zip(self.dim_mults[:-1], self.dim_mults[1:], reversed(self.laps[1:]), reversed(self.laps[:-1]), reversed(self.num_resblocks)):
            tmp = nn.ModuleList([])
            for jj in range(num_resblock):
                tmp.append(block_partial(dim_in, dim_in, lap_id))
                #if (jj == 0)&(self.use_attn):
                    #tmp.append(Attentions(self.attn_type, dim_in, lap_id))
            tmp.append(block_partial(dim_in, dim_out, lap_down, down = True))
            self.down_blocks.append(tmp)

        self.mid_block1 = block_partial(self.dim_mults[-1], self.dim_mults[-1], self.laps[0])
        #if self.use_attn:
        #    self.mid_attn = Attentions(self.attn_type, self.dim_mults[-1], self.laps[0])
        self.mid_block2 = block_partial(self.dim_mults[-1], self.dim_mults[-1], self.laps[0])

        self.up_blocks = nn.ModuleList([])
        for dim_in, dim_out, lap_id, lap_up, num_resblock in zip(reversed(self.dim_mults[1:]), reversed(self.dim_mults[:-1]), self.laps[:-1], self.laps[1:], self.num_resblocks):
            tmp = nn.ModuleList([])
            tmp.append(block_partial(2*dim_in, dim_in, lap_id))
            #if self.use_attn:
            #    tmp.append(Attentions(self.attn_type, dim_in, lap_id, self.kernel_size, n_head=1, norm_type=self.norm_type))
            tmp.append(block_partial(dim_in, dim_out, lap_up, up = True))
            self.up_blocks.append(tmp)

        self.out_block = block_partial(2 * self.dim_mults[0], self.inner_dim, self.laps[-1])
        self.out_norm = Norms(self.inner_dim, self.norm_type)
        self.out_act = Acts(self.act_type)
        self.final_conv = nn.Linear(self.inner_dim, self.dim_out)

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
        t = self.time_mlp(time)

        # downsample
        for downs in self.down_blocks:
            for block in downs:
                x = block(x, t)
            skip_connections.append(x)

        # bottleneck
        x = self.mid_block1(x, t)
        #if self.use_attn:
        #    x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for ups in self.up_blocks:
            tmp_connection = skip_connections.pop() * self.skip_factor
            x = torch.cat([x, tmp_connection], dim=2)
            for block in ups:
                x = block(x, t)

        tmp_connection = skip_connections.pop() * self.skip_factor
        x = torch.cat([x, tmp_connection], dim=2)
        x = self.out_block(x, t)
        x = self.out_norm(x)
        x = self.out_act(x)
        x = self.final_conv(x)
        return x