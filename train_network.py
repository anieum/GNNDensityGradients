import pytorch_lightning as pl
from pytorch_lightning.callbacks import \
    ModelCheckpoint, EarlyStopping, LearningRateFinder, LearningRateMonitor, RichModelSummary
from utils.train_helper import *
from utils.callbacks import *
from models.cconv import CConvModel
from datasets.vtk_dataset import VtkDataset
from datasets.density_data_module import DensityDataModule




hparams = {
    # General ---------------------------------------------------

    # Dataset
    'dataset_dir' : 'datasets/data/dam_break_preprocessed/train',
    'data_split'  : (0.7, 0.15, 0.15),
    'shuffle'     : True,
    'cache'       : True,                                         # Preprocess and preload dataset into memory

    # Training
    'batch_size'    : 2,
    'learning_rate' : 2e-3,
    'device'        : 'cuda' if torch.cuda.is_available() else 'cpu',

    'num_training_nodes'  : 1,    # number of nodes to train on (e.g. 1 GPU)
    'num_workers'         : 0,    # number of workers for dataloader (e.g. 2 worker threads)
    'num_epochs'          : 25,   # Per epoch with 1000 files containing 6600 samples expect 6 minutes on a 1080 Ti
    'limit_train_batches' : 0.5,  # Use only 10% of the training data per epoch; default is 1.0
    'limit_val_batches'   : 0.1,  # Use only 10% of the validation data per epoch; default is 1.0

    # Logging
    'log_every_n_steps' : 1,
    'val_every_n_epoch' : 3,

    # Checkpoints
    'load_checkpoint' : True,
    'save_path'       : 'lightning_logs/best',
    'load_path'       : '/home/jakob/ray_results3/LightningTrainer_2023-07-31_17-28-20/00093_93/rank_0/logs/srch/checkpoints/last.ckpt',
    'params_path'     : '/home/jakob/ray_results3/LightningTrainer_2023-07-31_17-28-20/00093_93/params.json',
    'model'           : CConvModel,

    # CConv parameters -------------------------------------------

    # CConv architecture
    "kernel_size"              : 4,  # Default is 4
    "num_hidden_layers"        : 2,  # Default is 2
    "input_layer_out_channels" : 32, # Default is 32
    "hidden_units"             : 64, # Default is 64
    "out_units"                : 1,  # 1 for temporal density gradient, 3 for spatial density gradient

    # CConv operation parameters
    "intermediate_activation_fn"        : "relu",                           # Default is ReLU;
    "interpolation"                     : "linear",                         # Default is linear;
    "align_corners"                     : True,                             # Default is True
    "normalize"                         : False,                            # Default is False
    "window_function"                   : "poly6",                          # Default is poly6
    "coordinate_mapping"                : "ball_to_cube_volume_preserving", # Default is ball_to_cube_volume_preserving
    "filter_extent"                     : 0.12574419077161608,              # Default is 0.025 * 6 * 1.5 = 0.225
    "radius_search_ignore_query_points" : False,                            # Default is False
    "use_dense_layer_for_centers"       : True,                             # Default is False
}

# Validate hyperparameters
validate_hparams(hparams)

# Load model
model = hparams['model'](hparams)

# Datasets
density_data = DensityDataModule(
    data_dir    = hparams['dataset_dir'],
    batch_size  = hparams['batch_size'],
    data_split  = hparams['data_split'],
    num_workers = hparams['num_workers'], # Note that cuda only allows 0 workers.
    shuffle     = hparams['shuffle'],
    cache       = hparams['cache'],       # Load dataset into memory
    device      = hparams['device'],
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
    num_nodes               = hparams['num_training_nodes'],
    max_epochs              = hparams['num_epochs'],
    log_every_n_steps       = hparams['log_every_n_steps'],
    check_val_every_n_epoch = hparams['val_every_n_epoch'],
    limit_train_batches     = hparams['limit_train_batches'],
    limit_val_batches       = hparams['limit_val_batches'],
    callbacks               = callbacks,
)

# DISABLED, the resulting learning rate is way too low
# model.learning_rate = find_learning_rate(trainer, model, density_data)

# Train
if hparams['load_checkpoint']:
    # Load hyperparameters from checkpoint; only overwrite model specific parameters
    print("Loading hyperparameters from checkpoint...")
    loaded_hparams = load_hparams(file_path=hparams['params_path'])
    hparams        = update_hparams(hparams=hparams, new_hparams=loaded_hparams)
    model          = hparams['model'].load_from_checkpoint(hparams['load_path'], hparams=hparams)
    print("Loaded hyperparameters from checkpoint:", hparams)

    print("Resuming training...")
    trainer.fit(model, datamodule=density_data, ckpt_path=hparams['load_path'])
else:
    trainer.fit(model, datamodule=density_data)

# Save model
save_checkpoint(trainer, model, hparams['save_path'])

# Testing (Only right before publishing the thesis)
# trainer.test(dataloaders=test_dataloaders)