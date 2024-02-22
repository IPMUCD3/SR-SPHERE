# NOT YET 
import torch
import torch.nn as nn
import math
from deepsphere.cheby_shev import SphericalChebConv

'''
This code defines a self-attention mechanism for neural networks, particularly useful in graph-based
neural networks where the convolutional operations are defined on the graph structure.
It employs spherical Chebyshev convolution for query, key, value transformations, and normalization
techniques for stabilizing the learning process. The self-attention mechanism is designed to process
graph-structured data, making it suitable for applications like HEALPix map.
'''

class SelfAttention(nn.Module):
    def __init__(self, in_channel, laplacian, kernel_size, n_head=1):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.BatchNorm1d(in_channel)
        self.qkv =  SphericalChebConv(in_channel, in_channel * 3, laplacian, kernel_size)
        self.out = SphericalChebConv(in_channel, in_channel, laplacian, kernel_size)

    def forward(self, input, time=None):
        batch, length, channel = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input.permute(0, 2, 1)).permute(0, 2, 1)
        qkv = self.qkv(norm).view(batch, length, n_head, head_dim * 3)
        query, key, value = qkv.chunk(3, dim=3)  # blnc

        attn = torch.einsum(
            "blnc, bknc -> blkn", query, key
        ).contiguous() / math.sqrt(channel)
        attn = torch.softmax(attn, 2)
        out = torch.einsum("blkn, bknc -> blnc", attn, value).contiguous()
        out = self.out(out.view(batch, length, channel))

        return out + input