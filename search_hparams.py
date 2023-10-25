from typing import Dict, List, Optional
import pytorch_lightning as pl
from ray.tune.experiment.trial import Trial
from ray.tune.stopper import Stopper
from utils.train_helper import *
from utils.callbacks import LogParametersCallback
from models.cconv import CConvModel
from datasets.density_data_module import DensityDataModule
from pytorch_lightning.loggers import TensorBoardLogger
import ray
from ray.train.lightning import LightningTrainer, LightningConfigBuilder
from ray import air, tune
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
import torch.cuda



hparams = {
    # Search parameters
    'num_epochs'  : 6, # TODO: 10
    'num_samples' : 100, # TODO: 100

    # Search space ---------------------------------------------

    # General
    'learning_rate' : tune.loguniform(2e-4, 6e-3),             # Default is 1e-3
    'batch_size'    : 2,                                       # Default is 10 tune.choice([1, 2, 4, 8]),
    # 'regularization': tune.choice([None, 'l1', 'l2']),         # NOT IMPLEMENTED
    # 'optimizer'     : tune.choice(['adam', 'sgd', 'rmsprop']), # NOT IMPLEMENTED

    # CConv architecture
    'kernel_size'              : tune.uniform(3, 7),   # Default is 4
    'num_hidden_layers'        : tune.uniform(1, 4),   # Default is 2
    'input_layer_out_channels' : tune.uniform(8, 33),  # Default is 32
    'hidden_units'             : tune.uniform(16, 65), # Default is 64

    # CConv operation parameters
    'intermediate_activation_fn'        : 'ReLU',                   # Default is ReLU; Alternives     : 'tanh',             'sigmoid', 'leaky_relu', 'GeLU'
    'interpolation'                     : 'linear',                 # Default is linear; Alternatives : 'nearest_neighbor', 'linear_border'
    'window_function'                   : 'None',                   # Default is poly6; Alternatives  : 'gaussian',         'cubic_spline'
    'coordinate_mapping'                : 'identity',               # Default is ball_to_cube_volume_preserving
    'filter_extent'                     : tune.uniform(0.12, 0.25), # Default is 0.025 * 6 * 1.5 = 0.225
    'radius_search_ignore_query_points' : False,                    # Default is False
    'use_dense_layer_for_centers'       : True,                     # Default is False

    # Static parameters -----------------------------------------
    'out_units'     : 1,     # 1 for temporal density gradient, 3 for spatial density gradient
    'align_corners' : True,  # Default is True
    'normalize'     : False, # Default is False

    # Dataset
    'dataset_dir' : 'datasets/data/dam_break_preprocessed/train',
    'data_split'  : (0.7, 0.15, 0.15),
    'shuffle'     : True,
    'cache'       : False,         # Preprocess and preload dataset into memory (GPU memory if cuda)
    'device'      : 'cuda',

    # Training
    'limit_train_batches' : 1000,  # Use only 10% of the training data per epoch; default is 1.0
    'limit_val_batches'   : 250,  # Use only 10% of the validation data per epoch; default is 1.0
}

# TODO: write Tune log into ram disk
datamodule = DensityDataModule(
    data_dir   = hparams['dataset_dir'],
    batch_size = hparams['batch_size'],
    data_split = hparams['data_split'],
    shuffle    = hparams['shuffle'],
    cache      = hparams['cache'],
    device     = hparams['device']
)
logger = TensorBoardLogger('logs', name='srch', version='.')

# DO NOT CACHE THE DATAMODULE IF IT IS PASSED DIRECTLY WITHOUT LOADERS.
# OTHERWISE RAY TUNE WILL SERIALIZE THE ENTIRE DATASET AND BLOW UP MEMORY AND DISK SPACE
datamodule.setup('fit')
# datamodule.to('cpu') # this potentially causes the datamodule to be serialized and sent to the workers, instead of the loaders
torch.cuda.empty_cache()

train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()


lightning_config = (
    LightningConfigBuilder()
    .module(cls=CConvModel, hparams=hparams)
    .trainer(
        max_epochs          = hparams['num_epochs'],
        logger              = logger,
        accelerator         = 'gpu',
        enable_progress_bar = False,
        limit_train_batches = hparams['limit_train_batches'],
        limit_val_batches   = hparams['limit_val_batches'],
        callbacks           = [LogParametersCallback()],
    )
    .fit_params(train_dataloaders=train_loader, val_dataloaders=val_loader)
    .checkpointing(monitor='val_loss', mode='min', save_top_k=1)
    .build()
)

ray.shutdown()
ray.init(num_cpus=6, num_gpus=1)

lightning_trainer = LightningTrainer(
    scaling_config = ScalingConfig(
        num_workers          = 1,
        use_gpu              = True,
        resources_per_worker = {'CPU': 1, 'GPU': 1. / 3}
    ),
    run_config = RunConfig(
        checkpoint_config = CheckpointConfig(
            num_to_keep                = 1,
            checkpoint_score_attribute = 'val_loss',
            checkpoint_score_order     = 'min',
        ),
    )
)

def tune_models(num_samples=100, num_epochs=25):
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=5, reduction_factor=2, time_attr='training_iteration')

    # Todo: search algorithm (the standard is either hyperopt, grid or random search)
    # https://docs.ray.io/en/latest/tune/api/suggestion.html
    # bayesopt = BayesOptSearch(metric="val_loss", mode="min", random_search_steps=10, patience=7)
    tuner = tune.Tuner(
        lightning_trainer,
        param_space = {
            'lightning_config': lightning_config,
            },
        tune_config = tune.TuneConfig(
            # search_alg   = bayesopt,
            metric       = 'val_loss',
            mode         = 'min',
            num_samples  = num_samples,
            scheduler    = scheduler,
        )
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric='val_loss', mode='min')
    print("Best result:", best_result)

print('Cuda is available:', torch.cuda.is_available())
tune_models(num_samples=hparams['num_samples'], num_epochs=hparams['num_epochs'])
 
tuner = tune.Tuner.restore(
    path      = "~/ray_results/LightningTrainer_2023-07-24_14-31-12",
    trainable = lightning_trainer
)
tuner.fit()