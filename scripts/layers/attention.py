import torch
import torch.nn as nn
import math
from scripts.layers.cheby_shev import SphericalChebConv
from scripts.layers.normalization import Norms
from scripts.layers.activation import Acts
from scripts.blocks.Resnetblock import Block

'''
This code defines a self-attention mechanism for neural networks, particularly useful in graph-based
neural networks where the convolutional operations are defined on the graph structure.
It employs spherical Chebyshev convolution for query, key, value transformations, and normalization
techniques for stabilizing the learning process. The self-attention mechanism is designed to process
graph-structured data, making it suitable for applications like HEALPix map.
'''

class SelfAttention(nn.Module):
    def __init__(self, in_channel, laplacian, kernel_size, n_head=1, norm_type="group"):
        super().__init__()

        self.n_head = n_head

        # Normalization layer
        self.norm = Norms(in_channel, norm_type)
        # Spherical Chebyshev convolution layers for query, key, and value
        self.qkv = SphericalChebConv(in_channel, in_channel * 3, laplacian, kernel_size)
        # Output spherical Chebyshev convolution layer
        self.out = SphericalChebConv(in_channel, in_channel, laplacian, kernel_size)

    def forward(self, input_):
        batch, patch_length, channel = input_.shape
        head_dim = channel // self.n_head

        # Normalization and transformation
        norm = self.norm(input_)
        qkv = self.qkv(norm).view(batch, patch_length, self.n_head, head_dim * 3)
        # Splitting query, key, value
        query, key, value = qkv.chunk(3, dim=3)

        # Attention calculation
        attn = torch.einsum("blnc, bmnc -> blmn", query, key).contiguous() / math.sqrt(channel)
        attn = torch.softmax(attn.view(batch, patch_length, self.n_head, -1), -1)
        attn = attn.view(batch, patch_length, patch_length, self.n_head)

        # Output calculation
        out = torch.einsum("blmn, bmnc -> blnc", attn, value).contiguous()
        out = self.out(out.view(batch, patch_length, channel))

        return out + input_
    
class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    # https://github.com/openai/consistency_models.git
    """
    def __init__(self, in_channel, laplacian, kernel_size=8, n_head=1, norm_type="group", act_type="silu"):
        super().__init__()

        self.n_head = n_head
        self.qkv = Block(in_channel, in_channel * 3, laplacian, kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        batch, length, channel = x.shape
        assert channel % self.n_heads == 0
        ch = channel // self.n_head
        qkv = self.qkv(x)
        query, key, value = qkv.chunk(3, dim=-1)

        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "btc,bsc->bts",
            (query * scale).view(batch * self.n_head, length, ch),
            (key * scale).view(batch * self.n_head, -1, ch),
        ) 

        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bsc->btc", weight, value.reshape(batch * self.n_head, -1, ch))
        return a.reshape(batch, length, -1)
