import os
import glob
import numpy as np
import torch
import healpy as hp
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, Normalize


# Constants
PIXEL_AREA_MULTIPLIER = 12  # The pixel area is defined by 12*order^2


def get_minmax_transform(rangemin, rangemax):
    """
    Function to get a pair of transforms that normalize and denormalize tensors.
    """
    transform = Compose([lambda t: (t - rangemin) / (rangemax - rangemin) * 2 - 1])
    inverse_transform = Compose([lambda t: (t + 1) / 2 * (rangemax - rangemin) + rangemin])
    
    return transform, inverse_transform


def get_file_info(filename, index):
    """
    Function to get the information from the filename. seed: 1, redshift: 2
    """
    return filename.split('/')[-1].split('_')[index]


def hp_split(img, order, nest=True):
    """
    Function to split the image into multiple images based on the given order.
    """
    npix = len(img)
    nsample = PIXEL_AREA_MULTIPLIER * order**2
    
    if npix < nsample:
        raise ValueError('Order not compatible with data.')
    
    if not nest:
        raise NotImplementedError('Implement the change of coordinate.')
    
    return img.reshape([nsample, npix // nsample])


class MapDataset(data.Dataset):
    """
    Class for the map dataset.
    """
    def __init__(self, mapdir, n_maps, nside, order=2, issplit=True, normalize=False):
        self.nside = nside
        self.n_maps = n_maps
        self.order = order
        self.issplit = issplit
        self.normalize = normalize
        self.npix = hp.nside2npix(self.nside)
        self.maps = sorted(glob.glob(f'{mapdir}*.fits'), key=lambda x: (get_file_info(x, 1), get_file_info(x, 2)))[:self.n_maps]
        self.ringorder = hp.nest2ring(self.nside, np.arange(self.npix))
        self.len = self.n_maps * (PIXEL_AREA_MULTIPLIER*self.order**2 if self.issplit else 1)
        self.mean = 0
        self.std = 1

    def __getitem__(self, index):
        dmaps = [hp.read_map(dmap) for dmap in self.maps]
        data = np.vstack([dmap for dmap in dmaps])
        
        if self.issplit:
            data = np.vstack([hp_split(el, order=self.order) for el in data])
            shape = (self.n_maps*self.order**2*PIXEL_AREA_MULTIPLIER, self.npix//(self.order**2*PIXEL_AREA_MULTIPLIER), 1)
        else:
            shape = (self.n_maps, self.npix, 1)
        
        tensor_map = ToTensor()(data).view(*shape).float()
        
        if self.normalize:
            tensor_map = self.normalize_map(tensor_map)
        
        return tensor_map

    def normalize_map(self, tensor_map):
        self.mean = tensor_map.mean().item()
        self.std = tensor_map.std().item()
        
        return Normalize(mean=[self.mean], std=[self.std])(tensor_map)

    def __len__(self):
        return self.len

def get_datasets(map_dirs, n_maps, nsides, order=2, issplit=False, normalize=True):
    datasets = {x: MapDataset(loc, n_maps, ns, order, issplit, normalize) for x, loc, ns in zip(('hr', 'lr'), map_dirs, nsides)}
    data_hr = datasets['hr'].__getitem__(0)
    data_lr = datasets['lr'].__getitem__(0)
    return data_hr, data_lr

def transform_combine(data_lr, data_hr):
    RANGE_MIN, RANGE_MAX = data_lr.min().clone().detach(), data_lr.max().clone().detach()
    transforms, inverse_transforms = get_minmax_transform(RANGE_MIN, RANGE_MAX)
    combined_dataset = data.TensorDataset(transforms(data_lr), transforms(data_hr))
    return combined_dataset

def get_loaders(map_dirs, n_maps, nsides, rate_train, batch_size, order=2, issplit=False, normalize=True):
    """
    Function to get the loaders for training and validation datasets.
    """
    data_hr, data_lr = get_datasets(map_dirs, n_maps, nsides, order, issplit, normalize)
    len_train = int(rate_train * len(data_hr))
    len_val = len(data_hr) - len_train

    combined_dataset = transform_combine(data_lr, data_hr)
    train, val = data.random_split(combined_dataset, [len_train, len_val])
    loaders = {x: data.DataLoader(ds, batch_size=batch_size, shuffle=x=='train', num_workers=os.cpu_count()) for x, ds in zip(('train', 'val'), (train, val))}
    
    return loaders['train'], loaders['val']


def get_loaders_from_params(params):
    """
    Function to get the loaders for training and validation datasets using parameters.
    """
    map_dirs = [params['hrmaps_dir'], params['lrmaps_dir']]
    nsides = [params['nside_hr'], params['nside_lr']]
    
    return get_loaders(map_dirs, params['n_maps'], nsides, params['rate_train'], params['batch_size'], params['order'], params['issplit'], params['normalize'])
