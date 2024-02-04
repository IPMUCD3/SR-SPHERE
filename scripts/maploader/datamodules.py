
import os
import torch.utils.data as data
import pytorch_lightning as pl

from scripts.maploader.fits_dataset import MapDataset

class DataModule(pl.LightningDataModule):
    """
    Data module for map dataset.

    Args:
        LR_dir (str): path to the map directory.
        HR_dir (str): path to the map directory.
        batch_size (int): batch size.
        rate_split (list): ratio of train, validation and test datasets.
        n_maps (int): number of maps to load.
        order (int): order of the map.
        norm (bool): normalize the map to [-1, 1] range.
        difference (bool): use difference map.
    """
    def __init__(self, **args):
        super().__init__()
        self.lr_dir = args['LR_dir']
        self.hr_dir = args['HR_dir']
        self.batch_size = args['batch_size']
        self.rate_split = args['rate_split']
        self.n_maps = args['n_maps']
        self.order = args['order']
        self.norm = args['norm']
        self.difference = args['difference']
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.dataset = MapDataset(self.lr_dir, self.hr_dir, n_maps=self.n_maps, norm=self.norm, order=self.order, difference=self.difference)
        self.len_train = int(self.rate_split[0] * self.dataset.__len__())
        self.len_val =  int(self.rate_split[1] * self.dataset.__len__())
        self.len_test = len(self.dataset) - self.len_train - self.len_val
        print("train:validation:test = {}:{}:{}, batch_size: {}".format(self.len_train, self.len_val, self.len_test, self.batch_size))
        self.train_dataset, self.val_dataset, self.test_dataset = data.random_split(self.dataset, [self.len_train, self.len_val, self.len_test])

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
        )