
import torch
from torch import nn
import numpy as np
from einops import rearrange

'''
The purpose of this code is to implement sinusoidal positional embeddings for time-based data.
It defines two classes: SinusoidalPositionEmbeddings and time_embed.
SinusoidalPositionEmbeddings generates embeddings using sinusoidal functions, which is particularly
useful in models where the relative or absolute position of time steps is important.
time_embed is a module that transforms these embeddings through a linear layer followed by an activation function.
'''

class SinusoidalPositionEmbeddings(nn.Module):
    # Embeds time in the phase using sinusoidal functions
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :] 
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings 

class TimeEmbed(nn.Module):
    # Projects and activates the time embeddings
    def __init__(self, time_emb_dim, in_channels):
        super().__init__()
        self.proj = nn.Linear(time_emb_dim, in_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(self.act(x))
        return rearrange(x, "b c -> b 1 c")
