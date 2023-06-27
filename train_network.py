from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import \
    ModelCheckpoint, EarlyStopping, LearningRateFinder, LearningRateMonitor, RichModelSummary
from utils.train_helper import find_learning_rate
from utils.callbacks import *

from models.cconv import CConvModel
from datasets.vtk_dataset import VtkDataset
from datasets.density_data_module import DensityDataModule

from glob import glob
import os



hparams = {
    'load_checkpoint': False,

    'dataset_dir': 'datasets/data/dpi_dam_break/train',
    'data_split': (0.7, 0.15, 0.15),
    'batch_size': 30, # IF BATCHSIZE != 1, torch.stack() will fail when composing the batch, as the particle count differs between samples
    'shuffle': False,
    'cache': True, # Load dataset into memory

    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_training_nodes': 1, # number of nodes to train on (e.g. 1 GPU)
    'num_workers': 0, # number of workers for dataloader (e.g. 2 worker threads)

    'num_epochs': 30,
    'log_every_n_steps': 1,
    'val_every_n_epoch': 3,

    # TODO: NOT YET IMPLEMENTED
    'num_features': 4,
    'lr': 2e-3,
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
model = CConvModel(hparams)

# Datasets
density_data = DensityDataModule(
    data_dir = hparams['dataset_dir'],
    batch_size = hparams['batch_size'],
    data_split = hparams['data_split'],
    num_workers = hparams['num_workers'], # Note that cuda only allows 0 workers.
    shuffle = hparams['shuffle'],
    cache = hparams['cache'], # Load dataset into memory
    device = hparams['device'],
)
density_data.setup("fit")

# TODO: callbacks: ModelCheckpoint, EarlyStopping, LearningRateFinder, LearningRateMonitor, RichModelSummary
callbacks = [
    VisualizePredictionCallback(model=model, dataset=density_data.dataset['train'], dataset_type="train"),
    VisualizePredictionCallback(model=model, dataset=density_data.dataset['eval'], dataset_type="eval"),
    VisulizePrediction3DCallback(model=model, dataset=density_data.dataset['train'], dataset_type="train"),
    VisulizePrediction3DCallback(model=model, dataset=density_data.dataset['eval'], dataset_type="eval"),
    # LearningRateMonitor(logging_interval='step'),
    # ActivationHistogramCallback(model=model)
]

trainer = pl.Trainer(
    num_nodes = hparams['num_training_nodes'],
    max_epochs = hparams['num_epochs'],
    log_every_n_steps = hparams['log_every_n_steps'],
    check_val_every_n_epoch = hparams['val_every_n_epoch'],
    callbacks = callbacks,
)

# TODO: Implement checkpoints
if hparams['load_checkpoint']:
    raise Exception("Not yet implemented")
    model = CConvModel(hparams)
    trainer.resume_from_checkpoint(model, ckpt_path=hparams['load_path'])

# DISABLED, the resulting learning rate is way too low
# model.learning_rate = find_learning_rate(trainer, model, density_data)

trainer.fit(model=model, datamodule=density_data)


# Testing (Only right before publishing the thesis)
# trainer.test(dataloaders=test_dataloaders)