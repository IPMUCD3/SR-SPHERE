
import torch.nn as nn

'''
This code defines a normalization module for neural networks. The Norms class allows the selection
between different types of normalization: batch normalization and group normalization. It is a flexible
component that can be incorporated into various neural network architectures to stabilize and speed up
training, as well as to prevent overfitting.
'''

class Norms(nn.Module):
    def __init__(self, dim, norm_type, num_groups=32):
        super().__init__()

        # Select the normalization type based on the 'norm_type' parameter
        if norm_type == "batch":
            self.norm = nn.BatchNorm1d(dim)  # Batch normalization
        elif norm_type == "group":
            self.norm = nn.GroupNorm(num_groups, dim, eps=1e-6, affine=True)  # Group normalization
        else:
            self.norm = nn.Identity()  # Identity (no normalization)

    def forward(self, x, time_emb=None):
        # Apply the chosen normalization
        # Permute the dimensions of x for compatibility with nn.BatchNorm1d and nn.GroupNorm
        return self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)

