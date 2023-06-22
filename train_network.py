# Torch
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
from datasets.density_dataset import SimulationDataset

from glob import glob
import os



hparams = {
    'data_split': (0.7, 0.15, 0.15),
    'batch_size': 2,
    'lr': 1e-3,                    #todo

    'num_training_nodes': 1,
    'num_workers': 0,              #todo
    'num_features': 4,
    'num_epochs': 1000,

    'log_every_n_steps': 1,
    'val_every_n_epoch': 10,
    'save_every_n_epoch': 100,     #todo

    'save_path': 'checkpoints',    #todo
    'load_path': None,             #todo
    'dataset_dir': 'datasets/data/dpi_dam_break',

    'model': 'cconv',              #todo
    'optimizer': 'adam',           #todo
    'scheduler': 'cosine',         #todo
}


if not os.path.exists(hparams['save_path']):
    os.makedirs(hparams['save_path'])

# Datasets
files = glob(os.path.join(hparams['dataset_dir'], 'train', '*.zst'))
data = SimulationDataset(files=files, transforms = None)

dataset = {}
dataset["train"], dataset["eval"], dataset["test"] = random_split(data, hparams['data_split'])

# Dataloaders
train_loader = DataLoader(dataset["train"], batch_size=hparams['batch_size'], num_workers=hparams['num_workers'])
val_loader = DataLoader(dataset["eval"], batch_size=hparams['batch_size'], num_workers=hparams['num_workers'])
test_loader = DataLoader(dataset["test"], batch_size=hparams['batch_size'], num_workers=hparams['num_workers'])

# Model
model = None
if hparams['load_path'] is not None:
    # model = CConvModel.load_from_checkpoint(hparams['load_path'])
    raise NotImplementedError("Loading from checkpoint not implemented yet")
else:
    model = CConvModel()

# Trainer
# see https://lightning.ai/docs/pytorch/stable/common/trainer.html
trainer = pl.Trainer(
    num_nodes = hparams['num_training_nodes'],
    max_epochs = hparams['num_epochs'],
    log_every_n_steps = hparams['log_every_n_steps'],
    check_val_every_n_epoch = hparams['val_every_n_epoch']
)

# Training
# TODO: callbacks: ModelCheckpoint, EarlyStopping, LearningRateFinder, LearningRateMonitor, RichModelSummary
callbacks = [
    VisualizePredictionCallback(model=model, dataset=dataset["train"], dataset_type="train"),
    VisualizePredictionCallback(model=model, dataset=dataset["eval"], dataset_type="eval"),
    ActivationHistogramCallback(model=model)
]
trainer.fit(model=model, train_dataloader=train_loader, val_dataloaders=val_loader, callbacks=callbacks)

# Testing (Only right before publishing the thesis)
# trainer.test(dataloaders=test_dataloaders)