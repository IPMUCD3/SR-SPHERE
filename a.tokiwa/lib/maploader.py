import numpy as np
import healpy as hp
from torchvision import transforms
import torch.utils.data as data
import glob
import sys

sys.path.append('/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE-1/a.tokiwa/lib/')
from partiallib import hp_split

class MapDataset(data.Dataset):
    def __init__(self, mapdir, n_maps, nside, issplit=False, order = 2):
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.mapdir = mapdir
        self.n_maps = n_maps
        self.order = order
        self.split_npix = 12*self.order**2
        self.len = self.n_maps 
        self.maps = sorted(glob.glob(self.mapdir+'*.fits'))[:self.n_maps]
        self.ringorder = hp.nest2ring(self.nside, np.arange(self.npix))
        self.issplit = issplit
        if self.issplit:
            self.len = self.n_maps  * self.split_npix


    def __getitem__(self):
        dmaps = [hp.read_map(dmap) for dmap in self.maps]
        data = np.vstack([dmap[self.ringorder] for dmap in dmaps])
        
        if self.issplit:
            data = np.vstack([hp_split(el, order=self.order) for el in data])
            dataset = transforms.ToTensor()(data).view(self.n_maps*self.split_npix, self.npix//self.split_npix, 1).float()
            return dataset
        else:
            dataset = transforms.ToTensor()(data).view(self.n_maps, self.npix, 1).float()
            return dataset
    
    def __len__(self):
        return self.len

def get_loaders(hrmaps_dir, lrmaps_dir, n_maps, nside_hr, nside_lr, rate_train, batch_size, issplit=False, order=2):
    dataset_hr = MapDataset(hrmaps_dir, n_maps, nside_hr, issplit=issplit, order=2)
    dataset_lr = MapDataset(lrmaps_dir, n_maps, nside_lr, issplit=issplit, order=2)
    len_train = int(dataset_hr.len * rate_train)
    len_val = int(dataset_hr.len - len_train)
    dlr = dataset_lr.__getitem__()
    dhr = dataset_hr.__getitem__()
    dataset = data.TensorDataset(dlr, dhr)
    train, val = data.random_split(dataset, [len_train, len_val])
    train_loader = data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=80)
    val_loader = data.DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=80)
    return train_loader, val_loader
        