
import numpy as np
import torch
from inspect import isfunction

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    shape = (batch_size, *((1,) * (len(x_shape) - 1)))
    return out.reshape(shape).to(t.device)

def mask_with_gaussian(x, mask=None, masks=None):
    device = x.device
    x = torch.chunk(x, 4, dim=1)
    if masks is None:
        masks = [[1, 1, 1, 1], [0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 0, 1]]
    if mask is None:
        mask = masks[np.random.randint(len(masks))]
    x = [torch.randn_like(x[i], device=device) if mask[i] else x[i] for i in range(4)]
    x = torch.cat(x, dim=1)
    return x

def extract_masked_region(x, mask):
    # get the chunks where mask is 1
    x = torch.chunk(x, 4, dim=1)
    x = [x[i] for i in range(4) if mask[i]]
    x = torch.cat(x, dim=1)
    return x