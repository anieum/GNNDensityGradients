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

        self.transform = None

        self.transform_once = tf.Compose([
            AddDensity(include_box=False),
            AddSpatialDensityGradient(include_box=False),
            AddTemporalDensityGradient(include_box=False),
            NormalizeDensityData(),
        ])


    def setup(self, stage: str):
        print("Setting up data module for stage ", stage)

        if bool(self.dataset):
            print("Dataset already set up")
            return

        data = SimulationDataset(
            files = self.files,
            transform = self.transform,
            transform_once = self.transform_once,
            cache = self.cache,
            device = self.device
        )

        dataset = {}
        dataset["train"], dataset["eval"], dataset["test"] = random_split(data, self.data_split)

        self.dataset = dataset

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.dataset["eval"], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

    def predict_dataloader(self):
        return DataLoader(self.dataset["eval"], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

    def teardown(self, stage: str):
        pass


# ref: https://lightning.ai/docs/pytorch/stable/data/datamodule.html