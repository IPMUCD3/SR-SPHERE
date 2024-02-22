
import logging
import torch
import numpy as np
import healpy as hp
import torch.utils.data as data
from glob import glob

from srsphere.dataset.data_transformer import Transforms

linear2log = Transforms('linear2log')
sigmoid = Transforms('sigmoid') 
minmax = Transforms('minmax', range_min=0.5, range_max=1)

def normalize(map, ifminmax=True):
    map_normalized = sigmoid.transform(linear2log.transform(map))
    if ifminmax:
        map_normalized = minmax.transform(map_normalized)
    return map_normalized

def denormalize(map, ifminmax=True):
    if ifminmax:
        map_denormalized = minmax.inverse_transform(map)
        map_denormalized = linear2log.inverse_transform(sigmoid.inverse_transform(map_denormalized))
    else:
        map_denormalized = linear2log.inverse_transform(sigmoid.inverse_transform(map))

    return map_denormalized

def normalize_diff(lrmap, hrmap):
    map_normalized = sigmoid.transform(linear2log.transform(hrmap) - linear2log.transform(lrmap))
    return map_normalized

def denormalize_diff(lrmap, diffmap):
    map_denormalized = linear2log.inverse_transform(sigmoid.inverse_transform(diffmap) + linear2log.transform(lrmap))
    return map_denormalized

def read_one_map(path, upsample_scale=1, norm=True, reshape_shape=None):
    tmp = hp.read_map(path) if not norm else normalize(hp.read_map(path)/upsample_scale)
    # replace nan with 0
    tmp[np.isnan(tmp)] = 0
    if reshape_shape is not None:
        tmp = tmp.reshape(reshape_shape)
    return tmp

class MapDataset(data.Dataset):
    """
    Class for the map dataset.

    Args:
        mapdir (str): path to the map directory.
        n_maps (int): number of maps to load.

    Attributes:
        n_maps (int): number of maps to load.
        data_shape (tuple): shape of the data.
        maps (list): list of maps.
    """
    def __init__(self, lrdir, hrdir, n_maps=None, norm=True, order=None, difference=True, upsample_scale=2**3, verbose=True):
        if n_maps is None:
            logging.info("n_maps is not specified, loading all maps in the directory.") if verbose else None
            self.lrmap_path = sorted(glob(f'{lrdir}/*.fits'))
            self.hrmap_path = sorted(glob(f'{hrdir}/*.fits'))
            self.n_maps = np.min([len(self.lrmap_path), len(self.hrmap_path)])
        else:
            logging.info(f"loading {n_maps} maps in the directory.") if verbose else None
            self.n_maps = n_maps
            self.lrmap_path = sorted(glob(f'{lrdir}/*.fits'))[:self.n_maps]
            self.hrmap_path = sorted(glob(f'{hrdir}/*.fits'))[:self.n_maps]

        self.difference = difference
        if order is not None:
            self.order = order
            self.n_patches = 12 * order**2
            self.lrmaps = [read_one_map(path, norm=norm, reshape_shape=[self.n_patches, -1, 1]) for path in self.lrmap_path]
            self.hrmaps = [read_one_map(path, upsample_scale, norm=norm, reshape_shape=[self.n_patches, -1, 1]) for path in self.hrmap_path]
            self.lrmaps = np.vstack(self.lrmaps)
            self.lrmaps = [self.lrmaps[i] for i in range(self.lrmaps.shape[0])]
            self.hrmaps = np.vstack(self.hrmaps)
            self.hrmaps = [self.hrmaps[i] for i in range(self.hrmaps.shape[0])]
        else:
            self.lrmaps = [read_one_map(path, norm=norm, reshape_shape=[-1, 1]) for path in self.lrmap_path]
            self.hrmaps = [read_one_map(path, upsample_scale, norm=norm, reshape_shape=[-1, 1]) for path in self.hrmap_path]
            
        if difference:
            self.diffmaps = [normalize_diff(denormalize(self.lrmaps[i]), denormalize(self.hrmaps[i])) for i in range(len(self.lrmaps))]

        if verbose:
            logging.info(f"LR data loaded from {lrdir}.  Number of maps: {self.n_maps}")
            logging.info(f"HR data loaded from {hrdir}.  Number of maps: {self.n_maps}")
            logging.info(f"data divided into {self.n_patches} patches.") if order is not None else None

    def __len__(self):
        return len(self.lrmaps)

    def __getitem__(self, index):
        if self.difference:
            return torch.from_numpy(self.lrmaps[index]).float(), torch.from_numpy(self.diffmaps[index]).float()
        else:
            return torch.from_numpy(self.lrmaps[index]).float(), torch.from_numpy(self.hrmaps[index]).float()
