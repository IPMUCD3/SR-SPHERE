import os
from glob import glob
from arrow import get
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

def get_sigmoid_transform():
    transform = Compose([lambda t: torch.sigmoid(t)])
    inverse_transform = Compose([lambda t: torch.logit((t))])
    return transform, inverse_transform

def get_both_transform(rangemin, rangemax):
    # perform sigmoid and then minmax
    transform = Compose([lambda t: torch.sigmoid(t), lambda t: (t - rangemin) / (rangemax - rangemin) * 2 - 1])
    inverse_transform = Compose([lambda t: (t + 1) / 2 * (rangemax - rangemin) + rangemin, lambda t: torch.logit((t))])
    return transform, inverse_transform

def get_log2linear_transform():
    transform = Compose([lambda t: 10**t - 1])
    inverse_transform = Compose([lambda t: torch.log10(t + 1)])
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
    def __init__(self, mapdir, n_maps, nside, order=2):
        self.nside = nside
        self.n_maps = n_maps
        self.order = order
        self.npix = hp.nside2npix(self.nside)
        self.data_shape = (self.n_maps, self.npix, 1)
        self.maps = sorted(glob(f'{mapdir}*.fits'))[:self.n_maps]
        self.patch_flag = False

    def get_numpymap(self):
        map_stacked = np.vstack([hp.read_map(dmap) for dmap in self.maps])
        return map_stacked
    
    def maps2patches(self, map_stacked):
        map_patches = np.vstack([hp_split(el, order=self.order) for el in map_stacked])
        self.patch_flag = True
        return map_patches

    def get_tensormap(self, map_stacked):
        if self.patch_flag:
            self.data_shape = (self.n_maps*self.order**2*PIXEL_AREA_MULTIPLIER, self.npix//(self.order**2*PIXEL_AREA_MULTIPLIER), 1)
        tensor_map = ToTensor()(map_stacked).view(*self.data_shape).float()
        return tensor_map    
    
def get_data(map_dir, n_map, nside, order=2, issplit=False):
    dataset = MapDataset(map_dir, n_map, nside, order)
    data_loaded_np = dataset.get_numpymap()
    if issplit:
        data_loaded_np = dataset.maps2patches(data_loaded_np)
    data_loaded = dataset.get_tensormap(data_loaded_np)
    return data_loaded

def get_normalized_data(data_loaded, transform_type='minmax'):
    if transform_type == 'minmax':
        data_normalized, transforms, inverse_transforms, range_min, range_max = get_minmaxnormalized_data(data_loaded)
    elif transform_type == 'sigmoid':
        data_normalized, transforms, inverse_transforms, range_min, range_max = get_sigmoidnormalized_data(data_loaded)
    elif transform_type == 'both':
        data_normalized, transforms, inverse_transforms, range_min, range_max = get_bothnormalized_data(data_loaded)
    else:
        raise NotImplementedError()
    return data_normalized, transforms, inverse_transforms, range_min, range_max

def get_sigmoidnormalized_data(data_loaded):
    range_min, range_max = data_loaded.min().clone().detach(), data_loaded.max().clone().detach()
    transforms, inverse_transforms = get_sigmoid_transform()
    data_normalized = transforms(data_loaded)
    return data_normalized, transforms, inverse_transforms, range_min, range_max

def get_minmaxnormalized_data(data_loaded):
    range_min, range_max = data_loaded.min().clone().detach(), data_loaded.max().clone().detach()
    transforms, inverse_transforms = get_minmax_transform(range_min, range_max)
    data_normalized = transforms(data_loaded)
    return data_normalized, transforms, inverse_transforms, range_min, range_max

def get_bothnormalized_data(data_loaded):
    range_min, range_max = torch.sigmoid(data_loaded).min().clone().detach(), torch.sigmoid(data_loaded).max().clone().detach()
    transforms, inverse_transforms = get_both_transform(range_min, range_max)
    data_normalized = transforms(data_loaded)
    return data_normalized, transforms, inverse_transforms, range_min, range_max

def get_loaders(data_input, data_condition, rate_train, batch_size):
    """
    Function to get the loaders for training and validation datasets.
    """
    combined_dataset = data.TensorDataset(data_input, data_condition)
    len_train = int(rate_train * len(data_input))
    len_val = len(data_input) - len_train
    train, val = data.random_split(combined_dataset, [len_train, len_val])
    loaders = {x: data.DataLoader(ds, batch_size=batch_size, shuffle=x=='train', num_workers=os.cpu_count()) for x, ds in zip(('train', 'val'), (train, val))}
    
    return loaders['train'], loaders['val']

def get_data_from_params(params):
    lr = get_data(params["data"]["LR_dir"], params["data"]["n_maps"], params["data"]["nside"], params["data"]["order"], issplit=True)   
    print("LR data loaded from {}.  Number of maps: {}".format(params["data"]["LR_dir"], params["data"]["n_maps"]))

    hr = get_data(params["data"]["HR_dir"], params["data"]["n_maps"], params["data"]["nside"], params["data"]["order"], issplit=True)
    print("HR data loaded from {}.  Number of maps: {}".format(params["data"]["HR_dir"], params["data"]["n_maps"]))

    print("data nside: {}, divided into {} patches, each patch has {} pixels.".format(params["data"]["nside"], 12 * params["data"]["order"]**2, hp.nside2npix(params["data"]["nside"])//(12 * params["data"]["order"]**2)))
    return lr, hr

def get_normalized_from_params(lr, hr, params):
    lr, transforms_lr, inverse_transform_lr, range_min_lr, range_max_lr = get_normalized_data(lr, transform_type=params["data"]["transform_type"])
    print("LR data normalized to [{},{}] by {} transform.".format(lr.min(), lr.max(), params["data"]["transform_type"]))

    if params["train"]['target'] == 'difference':
        diff = hr - inverse_transform_lr(lr)
        print("Difference data calculated from HR - LR. min: {}, max: {}".format(diff.min(), diff.max()))
        diff, transforms_diff, inverse_transforms_diff, range_min_diff, range_max_diff = get_normalized_data(diff, transform_type=params["data"]["transform_type"])
        print("Difference data normalized to [{},{}] by {} transform.".format(diff.min(), diff.max(), params["data"]["transform_type"]))
        data_input, data_condition = diff, lr
        return data_input, data_condition, transforms_lr, inverse_transform_lr, transforms_diff, inverse_transforms_diff, range_min_lr, range_max_lr, range_min_diff, range_max_diff
    elif params["train"]['target'] == 'HR':
        hr, transforms_hr, inverse_transforms_hr, range_min_hr, range_max_hr = get_normalized_data(hr, transform_type=params["data"]["transform_type"])
        print("HR data normalized to [{},{}] by {} transform.".format(hr.min(), hr.max(), params["data"]["transform_type"]))
        data_input, data_condition = hr, lr
        return data_input, data_condition, transforms_lr, inverse_transform_lr, transforms_hr, inverse_transforms_hr, range_min_lr, range_max_lr, range_min_hr, range_max_hr
    else:
        raise ValueError("target must be 'difference' or 'HR'")

def get_loaders_from_params(params):
    lr, hr = get_data_from_params(params)
    data_input, data_condition, _, _, _, _, _, _, _, _ = get_normalized_from_params(lr, hr, params)
    train_loader, val_loader = get_loaders(data_input, data_condition, params["train"]['train_rate'], params["train"]['batch_size'])
    print("train:validation = {}:{}, batch_size: {}".format(len(train_loader), len(val_loader), params["train"]['batch_size']))
    return train_loader, val_loader