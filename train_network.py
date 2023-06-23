from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import \
    ModelCheckpoint, EarlyStopping, LearningRateFinder, LearningRateMonitor, RichModelSummary
from utils.callbacks import *

from models.cconv import CConvModel
from datasets.vtk_dataset import VtkDataset
from datasets.density_data_module import DensityDataModule

from glob import glob
import os



hparams = {
    'dataset_dir': 'datasets/data/dpi_dam_break/train',
    'data_split': (0.7, 0.15, 0.15),
    'batch_size': 2,

    'num_training_nodes': 1, # number of nodes to train on
    'num_workers': 0, # number of workers for dataloader

    'num_epochs': 5,
    'log_every_n_steps': 1,
    'val_every_n_epoch': 10,

    # TODO: NOT YET IMPLEMENTED
    'num_features': 4,
    'lr': 1e-3,
    'save_every_n_epoch': 100,
    'save_path': 'checkpoints',
    'load_path': None,
    'model': 'cconv',
    'optimizer': 'adam',
    'scheduler': 'cosine',
}

if not os.path.exists(hparams['save_path']):
    os.makedirs(hparams['save_path'])

if not os.path.exists(hparams['dataset_dir']):
    raise Exception("Data directory does not exist")

hparams['dataset_dir'] = os.path.abspath(hparams['dataset_dir'])

# Model
# TODO: https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html
model = CConvModel()

# Datasets
density_data = DensityDataModule(
    data_dir = hparams['dataset_dir'],
    batch_size = hparams['batch_size'],
    data_split = hparams['data_split'],
    num_workers = hparams['num_workers'],
    device = model.device
)
density_data.setup("fit")

# TODO: callbacks: ModelCheckpoint, EarlyStopping, LearningRateFinder, LearningRateMonitor, RichModelSummary
callbacks = [
    VisualizePredictionCallback(model=model, dataset=density_data.train_dataloader(), dataset_type="train"),
    VisualizePredictionCallback(model=model, dataset=density_data.val_dataloader(), dataset_type="eval"),
    ActivationHistogramCallback(model=model)
]
callbacks.clear()

trainer = pl.Trainer(
    num_nodes = hparams['num_training_nodes'],
    max_epochs = hparams['num_epochs'],
    log_every_n_steps = hparams['log_every_n_steps'],
    check_val_every_n_epoch = hparams['val_every_n_epoch'],
    callbacks = callbacks,
)

trainer.fit(model=model, datamodule=density_data)


# Testing (Only right before publishing the thesis)
# trainer.test(dataloaders=test_dataloaders)