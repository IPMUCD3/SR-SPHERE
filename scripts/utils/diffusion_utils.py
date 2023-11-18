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

def mask_with_gaussian(x, device, num_chunk=4):
    x = torch.chunk(x, num_chunk, dim=1)
    flags = torch.randint(2, size=(num_chunk,)).bool()
    # make sure at least one chunk is masked
    while flags.all():
        flags = torch.randint(2, size=(num_chunk,)).bool()
    x = [x[i] if flags[i] else torch.randn_like(x[i], device=device) for i in range(num_chunk)]
    x = torch.cat(x, dim=1)
    return x