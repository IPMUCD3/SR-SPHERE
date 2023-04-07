import torch
import numpy as np
import healpy as hp
from spherical_unet_utils import SphericalCheb, PatchSphericalCheb
from maploader import hp_split

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
        x = self.sc_in(x)
        x = self.pixelshuffle(x)
        x = self.sc_out(x)
        return x
    
class patch_healpix_pixelshuffle(torch.nn.Module):
    def __init__(self, npix_hr, npix_lr, unpooling, split_npix=12*2**2):
        super().__init__()
        self.mat_order = hp_split(npix_lr, split_npix=split_npix).mat_order
        self.t = hp.pixelfunc.ud_grade(np.arange(npix_lr), hp.npix2nside(npix_hr))
        self.neighbors = np.array([np.where(self.t==i)[0] for i in range(npix_lr)])
        self.unpooling = unpooling

    def forward(self, x, label):
        tmp_neighbors = self.neighbors[self.mat_order[label]]
        chunks = torch.chunk(x, 4, dim=2)
        x1 = self.unpooling(chunks[0])
        for i in range(1,4):
            x1[:,tmp_neighbors[:,i],:] =  chunks[i]
        return x1
    
class patch_PSUB(torch.nn.Module):
    def __init__(self, npix_hr, npix_lr, unpooling, nf, kernel_size=10, split_npix=12*2**2):
        super().__init__()
        self.sc_in = PatchSphericalCheb(nf, nf * 4, kernel_size)
        self.sc_out = PatchSphericalCheb(nf, nf, kernel_size)
        self.pixelshuffle = patch_healpix_pixelshuffle(npix_hr, npix_lr, unpooling, split_npix=split_npix)
        
    def forward(self, x, label, lap0, lap1):
        x = self.sc_in(x, lap0)
        x = self.pixelshuffle(x, label)
        x = self.sc_out(x, lap1)
        return x