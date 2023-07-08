import pytorch_lightning as pl
from utils.train_helper import *
from models.cconv import CConvModel
from datasets.density_data_module import DensityDataModule
from pytorch_lightning.loggers import TensorBoardLogger
import ray
from ray.train.lightning import LightningTrainer, LightningConfigBuilder
from ray import air, tune
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.tune.schedulers import ASHAScheduler
import torch.cuda

# See https://docs.ray.io/en/latest/train/examples/lightning/lightning_mnist_example.html#lightning-mnist-example
# and https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html



hparams = {
    # Search space ---------------------------------------------
    'num_epochs'  : 10,
    'num_samples' : 10,

    # General
    'learning_rate' : tune.loguniform(1e-4, 1e-2),             # Default is 1e-3
    'batch_size'    : 2,                                       # Default is 10 tune.choice([1, 2, 4, 8]),
    # 'regularization': tune.choice([None, 'l1', 'l2']),         # NOT IMPLEMENTED
    # 'optimizer'     : tune.choice(['adam', 'sgd', 'rmsprop']), # NOT IMPLEMENTED

    # CConv architecture
    'kernel_size'             : tune.choice([3, 4, 8]),          # Default is 4; -1 is a magic value to use a dense layer
    'num_hidden_layers'       : tune.choice([0, 1, 2, 3, 4]),    # Default is 2
    'input_layer_out_channels': tune.choice([4, 8, 16, 32, 64]), # Default is 32
    'hidden_units'            : tune.choice([32, 64, 128, 256]), # Default is 64

    # CConv operation parameters
    'intermediate_activation_fn' : tune.choice(['relu', 'tanh', 'GeLU']),                                              # Default is ReLU; Alternives     : 'tanh',             'sigmoid', 'leaky_relu', 'GeLU'
    'interpolation'              : tune.choice(['linear']),                                                            # Default is linear; Alternatives : 'nearest_neighbor', 'linear_border'
    'align_corners'              : tune.choice([True, False]),                                                         # Default is True
    'normalize'                  : tune.choice([True, False]),                                                         # Default is False
    'window_function'            : tune.choice(['None', 'poly6']),                                                     # Default is poly6; Alternatives  : 'gaussian',         'cubic_spline'
    'coordinate_mapping'         : tune.choice(['ball_to_cube_volume_preserving', 'ball_to_cube_radial', 'identity']), # Default is ball_to_cube_volume_preserving
    'filter_extent'              : tune.loguniform(0.025 * 4 * 1.0, 0.025 * 9 * 1.5),                                  # Default is 0.025 * 6 * 1.5 = 0.225
    'radius_search_ignore_query_points' : tune.choice([True, False]),                                                 # Default is False

    # Static parameters -----------------------------------------
    'out_units'  : 1,           # 1 for temporal density gradient, 3 for spatial density gradient

    # Dataset
    'dataset_dir' : 'datasets/data/dam_break_preprocessed/train',
    'data_split'  : (0.7, 0.15, 0.15),
    'shuffle'     : True,
    'cache'       : False,         # Preprocess and preload dataset into memory (GPU memory if cuda)
    'device'      : 'cuda',

    # Training
    'limit_train_batches' : 0.1,  # Use only 10% of the training data per epoch; default is 1.0
    'limit_val_batches'   : 0.1,  # Use only 10% of the validation data per epoch; default is 1.0
}

datamodule = DensityDataModule(data_dir=hparams['dataset_dir'], batch_size=hparams['batch_size'], data_split=hparams['data_split'], shuffle=hparams['shuffle'], cache=hparams['cache'], device=hparams['device'])
logger = TensorBoardLogger('lightning_logs', name='cconv-hparam-search', version='.')

# DO NOT CACHE THE DATAMODULE IF IT IS PASSED DIRECTLY WITHOUT LOADERS.
# OTHERWISE RAY TUNE WILL SERIALIZE THE ENTIRE DATASET AND BLOW UP MEMORY AND DISK SPACE
datamodule.setup('fit')
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
    )
    .fit_params(train_dataloaders=train_loader, val_dataloaders=val_loader)
    .checkpointing(monitor='val_loss', mode='min', save_top_k=2)
    .build()
)

ray.shutdown()
ray.init(num_cpus=6, num_gpus=1)

lightning_trainer = LightningTrainer(
    scaling_config = ScalingConfig(
        num_workers          = 1,
        use_gpu              = True,
        resources_per_worker = {'CPU': 2, 'GPU': 0.5}
    ),
    run_config = RunConfig(
        checkpoint_config = CheckpointConfig(
            num_to_keep                = 2,
            checkpoint_score_attribute = 'val_loss',
            checkpoint_score_order     = 'min',
        ),
    )
)

def tune_models(num_samples=10, num_epochs=10):
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    # Todo: search algorithm (the standard is either hyperopt, grid or random search)
    # https://docs.ray.io/en/latest/tune/api/suggestion.html

    tuner = tune.Tuner(
        lightning_trainer,
        param_space = {'lightning_config': lightning_config},
        tune_config = tune.TuneConfig(
            metric      = 'val_loss',
            mode        = 'min',
            num_samples = num_samples,
            scheduler   = scheduler,
        )
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric='val_loss', mode='max')
    best_result

print('Cuda is available: ', torch.cuda.is_available())
tune_models(num_samples=hparams['num_samples'], num_epochs=hparams['num_epochs'])