import pytorch_lightning as pl
from utils.transforms import *
import os
from glob import glob
import torchvision.transforms as tf
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from datasets.density_dataset import SimulationDataset

class DensityDataModule(pl.LightningDataModule):
    def __init__(self,
                 target = "temporal_density_gradient",
                 include_bounding_box = False,
                 data_dir: str = "path/to/dir",
                 batch_size: int = 32,
                 data_split: tuple = (0.7, 0.15, 0.15),
                 num_workers: int = 0,
                 shuffle: bool = False,
                 cache=False,
                 device = 'cpu'):

        super().__init__()
        if not os.path.exists(data_dir):
            raise Exception("Data directory does not exist")

        self.files = glob(os.path.join(data_dir, '*.zst'))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_split = data_split
        self.shuffle = shuffle
        self.device = device
        self.cache = cache
        self.dataset = {}
        self.total_dataset = None

        self.transform = tf.Compose([
            CorruptAttribute("pos", 0.005),
            CorruptAttribute("vel", 0.005),
        ])

        one_time_transforms = [AddDensity(include_box=False)]

        if target == "temporal_density_gradient":
            one_time_transforms.append(AddTemporalDensityGradient(include_box=include_bounding_box))
        elif target == "spatial_density_gradient":
            one_time_transforms.append(AddSpatialDensityGradient(include_box=include_bounding_box))
        else:
            raise Exception("Unknown target")

        one_time_transforms.append(NormalizeDensityData())

        self.transform_once = tf.Compose(one_time_transforms)

    def _collate_identity(self, x):
        if not isinstance(x, list):
            raise Exception("x must be a list")

        # check if for each element['pos'] in the list, the dimensions are 2
        return x


    def setup(self, stage: str):
        print("Setting up data module for stage ", stage)

        if bool(self.dataset):
            print("Dataset already set up")
            return

        self.total_dataset = SimulationDataset(
            files = self.files,
            transform = self.transform,
            transform_once = self.transform_once,
            cache = self.cache,
            device = self.device
        )

        dataset = {}
        dataset["train"], dataset["val"], dataset["test"] = random_split(self.total_dataset, self.data_split)

        self.dataset = dataset

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], collate_fn=self._collate_identity, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.dataset["val"], collate_fn=self._collate_identity, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], collate_fn=self._collate_identity, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

    def predict_dataloader(self):
        return DataLoader(self.dataset["val"], collate_fn=self._collate_identity, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

    def teardown(self, stage: str):
        pass

    def to(self, device):
        self.total_dataset.to(device)
        self.device = device
        return self


# ref: https://lightning.ai/docs/pytorch/stable/data/datamodule.html