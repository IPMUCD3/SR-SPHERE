import numpy as np
import healpy as hp
from torchvision import transforms
import torch.utils.data as data
import os
import glob

class hp_split():
    def __init__(self, npix_hr, split_npix):
        self.npix_hr = npix_hr
        self.nside_hr = hp.npix2nside(npix_hr)
        self.split_npix = split_npix
        self.t = hp.pixelfunc.ud_grade(np.arange(self.split_npix), self.nside_hr)        
        self.mat_split = np.array([self.t==i for i in range(self.split_npix)])
        self.sample = np.arange(self.npix_hr)
        self.mat_order = np.array([self.sample[el] for el in self.mat_split])

    def split(self, img):
        return np.array([img[el] for el in self.mat_split])


class MapDataset(data.Dataset):
    def __init__(self, mapdir, n_maps, nside, issplit=False, split_npix = 12*2**2):
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.mapdir = mapdir
        self.n_maps = n_maps
        self.split_npix = split_npix
        self.len = self.n_maps 
        self.maps = sorted(glob.glob(self.mapdir+'*.fits'))[:self.n_maps]
        self.ringorder = hp.nest2ring(self.nside, np.arange(self.npix))
        self.issplit = issplit

        if self.issplit:
            self.split = hp_split(self.npix, split_npix = self.split_npix)
            self.len = self.n_maps  * self.split_npix

    def __getitem__(self):
        dmaps = [hp.read_map(dmap) for dmap in self.maps]
        data = np.vstack([dmap[self.ringorder] for dmap in dmaps])
        if self.issplit:
            label = np.repeat(np.arange(self.split_npix)[None, :], self.n_maps, axis=0).ravel().reshape(-1,1)
            dataset =  np.vstack([self.split.split(el) for el in data])
            return label, transforms.ToTensor()(dataset).view(self.n_maps*self.split_npix, self.npix//self.split_npix, 1)
        else:
            dataset = transforms.ToTensor()(data).view(self.n_maps, self.npix, 1)
            return dataset
    
    def __len__(self):
        return self.len

def get_loaders(hrmaps_dir, lrmaps_dir, n_maps, nside_hr, nside_lr, rate_train, batch_size, issplit=False, split_npix = 12*2**2):
    dataset_hr = MapDataset(hrmaps_dir, n_maps, nside_hr, issplit=issplit, split_npix=split_npix)
    dataset_lr = MapDataset(lrmaps_dir, n_maps, nside_lr, issplit=issplit, split_npix=split_npix)
    len_train = int(dataset_hr.len * rate_train)
    len_val = int(dataset_hr.len - len_train)
    dataset = data.TensorDataset(dataset_hr, dataset_lr)
    train, val = data.random_split(dataset, [len_train, len_val])
    train_loader = data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

"""
class MapDataset(data.Dataset):
    def __init__(self, hrmaps_dir, lrmaps_dir, n_maps, nside_hr, nside_lr, rate_train, split_npix = 12*2**2):
        self.nside_hr = nside_hr
        self.nside_lr = nside_lr
        self.npix_hr = hp.nside2npix(nside_hr)
        self.npix_lr = hp.nside2npix(nside_lr)

        self.hrmaps_dir = hrmaps_dir
        self.lrmaps_dir = lrmaps_dir
        self.n_maps = n_maps
        self.rate_train = rate_train
        self.split_npix = split_npix

        self.len_train = int(self.n_maps * self.rate_train)
        self.len_val = int((self.n_maps - self.len_train))
        
        # load maps
        hrmaps = glob.glob(hrmaps_dir + "*.fits")
        hrmaps.sort()#(key=lambda x: x.split("_")[-5])
        lrmaps = glob.glob(lrmaps_dir + "*.fits")
        lrmaps.sort()#(key=lambda x: x.split("_")[-1])
        
        d_lrmaps = [hp.read_map(lrmaps[i]) for i in range(self.n_maps)]
        d_hrmaps = [hp.read_map(hrmaps[i]) for i in range(self.n_maps)]

        ringorder_lr = hp.nest2ring(self.nside_lr, np.arange(self.npix_lr))
        ringorder_hr = hp.nest2ring(self.nside_hr, np.arange(self.npix_hr))
        
        self.data_train = np.vstack([d[ringorder_lr] for d in d_lrmaps]).astype("float")
        self.data_target = np.vstack([d[ringorder_hr] for d in d_hrmaps]).astype("float")
        self.trans = transforms.ToTensor()
        
        self.inputs = self.trans(self.data_train).view((self.n_maps , self.npix_lr, 1)).float()
        self.targets = self.trans(self.data_target).view((self.n_maps, self.npix_hr, 1)).float()

        self.split_hr = hp_split(self.npix_hr, split_npix=self.split_npix)
        self.split_lr = hp_split(self.npix_lr, split_npix=self.split_npix)
        
    def get_train_val_loaders(self, batch_size):
        self.dataset = data.TensorDataset(self.inputs, self.targets)
        self.train_dataset, self.val_dataset = data.random_split(self.dataset, [self.len_train, self.len_val])
        train_loader = data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
        val_loader = data.DataLoader(self.val_dataset, batch_size=batch_size, num_workers=os.cpu_count())
        return train_loader, val_loader
    
    def split_data(self, batch_size):
        label = np.repeat(np.arange(self.split_npix)[None, :], self.n_maps, axis=0).ravel().reshape(-1,1)
        self.split_train = self.trans(np.hstack([label, np.vstack([self.split_lr.split(el) for el in self.data_train])])).view((self.split_npix*self.n_maps , self.npix_lr//self.split_npix + 1, 1)).float()
        self.split_target = self.trans(np.vstack([self.split_hr.split(el) for el in self.data_target])).view((self.split_npix*self.n_maps , self.npix_hr//self.split_npix, 1)).float()
        self.dataset = data.TensorDataset(self.split_train, self.split_target)
        self.len_train = int(self.split_npix* self.n_maps * self.rate_train)
        self.len_val = int((self.split_npix*self.n_maps - self.len_train))

        self.train_dataset, self.val_dataset = data.random_split(self.dataset, [self.len_train, self.len_val])
        train_loader = data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
        val_loader = data.DataLoader(self.val_dataset, batch_size=batch_size, num_workers=os.cpu_count())
        return train_loader, val_loader
"""
        