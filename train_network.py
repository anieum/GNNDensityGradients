import pytorch_lightning as pl
from pytorch_lightning.callbacks import \
    ModelCheckpoint, EarlyStopping, LearningRateFinder, LearningRateMonitor, RichModelSummary
from utils.train_helper import *
from utils.callbacks import *
from models.cconv import CConvModel
from datasets.vtk_dataset import VtkDataset
from datasets.density_data_module import DensityDataModule

hparams = {
    # Dataset
    'dataset_dir': 'datasets/data/dpi_dam_break/train',
    'data_split': (0.7, 0.15, 0.15),
    'batch_size': 30,
    'shuffle': True,
    'cache': True,            # Preprocess and preload dataset into memory

    # Training
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_training_nodes': 1,  # number of nodes to train on (e.g. 1 GPU)
    'num_workers': 0,         # number of workers for dataloader (e.g. 2 worker threads)
    'num_epochs': 0,
    'lr': 2e-3,

    # Logging
    'log_every_n_steps': 1,
    'val_every_n_epoch': 3,

    # Checkpoints
    'load_checkpoint': False,
    'save_path': 'lightning_logs',
    'load_path': 'lightning_logs/version_42.ckpt',
    'model': CConvModel,
}

# Validate hyperparameters
validate_hparams(hparams)

# Load model
m = hparams['model']
model = m.load_from_checkpoint(hparams['load_path']) if hparams['load_checkpoint'] else m(hparams)

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
density_data.setup("Initialize")

# TODO: callbacks: ModelCheckpoint, EarlyStopping, LearningRateFinder, LearningRateMonitor, RichModelSummary
callbacks = [
    VisualizePredictionCallback(model=model, dataset=density_data.dataset['train'], dataset_type="train"),
    VisualizePredictionCallback(model=model, dataset=density_data.dataset['val'], dataset_type="val"),
    VisulizePrediction3DCallback(model=model, dataset=density_data.dataset['train'], dataset_type="train"),
    VisulizePrediction3DCallback(model=model, dataset=density_data.dataset['val'], dataset_type="val"),
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

# DISABLED, the resulting learning rate is way too low
# model.learning_rate = find_learning_rate(trainer, model, density_data)

# Train
trainer.fit(model, datamodule=density_data, ckpt_path=hparams['load_path'])

# Save model
save_checkpoint(trainer, model, hparams['save_path'])

# Testing (Only right before publishing the thesis)
# trainer.test(dataloaders=test_dataloaders)