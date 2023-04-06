import torch
import numpy as np
import healpy as hp
from spherical_unet_utils import SphericalCheb

class healpix_pixelshuffle(torch.nn.Module):
    def __init__(self, n_hr, n_lr, unpooling):
        super().__init__()
        self.t = hp.pixelfunc.ud_grade(np.arange(hp.nside2npix(n_lr)), n_hr)
        self.neighbors = np.array([np.where(self.t==i)[0] for i in range(hp.nside2npix(n_lr))])
        self.unpooling = unpooling

    def forward(self, x):
        chunks = torch.chunk(x, 4, dim=2)
        x1 = self.unpooling(chunks[0])
        for i in range(1,4):
            x1[:,self.neighbors[:,i],:] =  chunks[i]
        return x1  
    
class PSUB(torch.nn.Module):
    def __init__(self, n_hr, n_lr, unpooling, nf, lap0, lap1, kernel_size=10):
        super().__init__()
        self.sc_in = SphericalCheb(nf, nf * 4, lap0, kernel_size)
        self.sc_out = SphericalCheb(nf, nf, lap1, kernel_size)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.pixelshuffle = healpix_pixelshuffle(n_hr, n_lr, unpooling)
        
    def forward(self, x):
        x = self.lrelu(self.sc_in(x))
        x = self.pixelshuffle(x)
        x = self.lrelu(self.sc_out(x))
        return x